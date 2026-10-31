"""Reusable Goal-conditioned vision model and inference preprocessing."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image
from torch import nn

COLOR_TO_ID = {"red_cube": 0, "blue_cube": 1, "green_cube": 2}
DESTINATION_TO_ID = {"left_bin": 0, "right_bin": 1}


@dataclass(frozen=True)
class TargetNormalizer:
    """Train-split-only min/max normalization for world XY targets."""

    minimum_xy_m: list[float]
    maximum_xy_m: list[float]

    @classmethod
    def fit(cls, targets: np.ndarray) -> TargetNormalizer:
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("Targets must have shape (N, 2)")
        minimum = targets.min(axis=0)
        maximum = targets.max(axis=0)
        if np.any(maximum <= minimum):
            raise ValueError("Target normalization range must be non-zero")
        return cls(minimum.tolist(), maximum.tolist())

    def normalize_array(self, targets: np.ndarray) -> np.ndarray:
        minimum = np.asarray(self.minimum_xy_m, dtype=np.float32)
        span = np.asarray(self.maximum_xy_m, dtype=np.float32) - minimum
        return 2.0 * (np.asarray(targets, dtype=np.float32) - minimum) / span - 1.0

    def denormalize_tensor(self, targets: torch.Tensor) -> torch.Tensor:
        minimum = targets.new_tensor(self.minimum_xy_m)
        maximum = targets.new_tensor(self.maximum_xy_m)
        return (targets + 1.0) * 0.5 * (maximum - minimum) + minimum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rgb_tensor(
    image_path: Path,
    image_size_wh: tuple[int, int],
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """Apply the exact resize and normalization shared by training and inference."""
    image = Image.open(image_path).convert("RGB")
    return preprocess_rgb_image(image, image_size_wh, mean, std)


def preprocess_rgb_array(
    rgb: np.ndarray,
    image_size_wh: tuple[int, int],
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """Preprocess an in-memory Genesis RGB frame identically to training images."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    return preprocess_rgb_image(image, image_size_wh, mean, std)


def preprocess_rgb_image(
    image: Image.Image,
    image_size_wh: tuple[int, int],
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """Shared PIL implementation for file-based training and array-based inference."""
    image = image.resize(image_size_wh, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


class SmallImageEncoder(nn.Module):
    """Compact spatial CNN suitable for a 4 GB GTX 1050."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 8 * 10, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(images))


class SpatialImageEncoder(nn.Module):
    """Preserve a dense feature map so a color query can select a small object."""

    output_channels = 48

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, self.output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class GoalConditionedGraspModel(nn.Module):
    """Predict normalized world XY from image features and structured goal IDs."""

    def __init__(
        self,
        mode: str = "full",
        image_feature_dim: int = 128,
        color_embedding_dim: int = 8,
        destination_embedding_dim: int = 4,
        dropout: float = 0.1,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"full", "instruction_only", "image_only"}:
            raise ValueError(f"Unsupported model mode: {mode}")
        self.mode = mode
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean or [0.0, 0.0, 0.0], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std or [1.0, 1.0, 1.0], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self.image_encoder = (
            SmallImageEncoder(image_feature_dim, dropout) if mode == "image_only" else None
        )
        self.spatial_encoder = SpatialImageEncoder() if mode == "full" else None
        self.color_embedding = (
            None if mode == "image_only" else nn.Embedding(len(COLOR_TO_ID), color_embedding_dim)
        )
        self.destination_embedding = (
            None
            if mode == "image_only"
            else nn.Embedding(len(DESTINATION_TO_ID), destination_embedding_dim)
        )
        input_dim = 0
        if self.spatial_encoder is not None:
            self.query_projection = nn.Linear(
                color_embedding_dim, SpatialImageEncoder.output_channels
            )
            input_dim += 4 + 2 * SpatialImageEncoder.output_channels
        else:
            self.query_projection = None
        if self.image_encoder is not None:
            input_dim += image_feature_dim
        if self.color_embedding is not None and self.destination_embedding is not None:
            input_dim += color_embedding_dim + destination_embedding_dim
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(
        self,
        images: torch.Tensor,
        color_ids: torch.Tensor,
        destination_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features: list[torch.Tensor] = []
        attention_xy: torch.Tensor | None = None
        if (
            self.spatial_encoder is not None
            and self.query_projection is not None
            and self.color_embedding is not None
        ):
            raw_images = torch.clamp(images * self.image_std + self.image_mean, 0.0, 1.0)
            red_contrast = raw_images[:, 0] - torch.maximum(raw_images[:, 1], raw_images[:, 2])
            blue_contrast = raw_images[:, 2] - torch.maximum(raw_images[:, 0], raw_images[:, 1])
            green_contrast = raw_images[:, 1] - torch.maximum(raw_images[:, 0], raw_images[:, 2])
            color_contrasts = torch.stack(
                (red_contrast, blue_contrast, green_contrast), dim=1
            )
            gather_index = color_ids.view(-1, 1, 1, 1).expand(
                -1, 1, images.shape[2], images.shape[3]
            )
            target_color_map = torch.relu(
                color_contrasts.gather(1, gather_index) - 0.18
            )
            image_height, image_width = images.shape[-2:]
            image_grid_y, image_grid_x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, image_height, device=images.device),
                torch.linspace(-1.0, 1.0, image_width, device=images.device),
                indexing="ij",
            )
            mask_weights = target_color_map[:, 0]
            mask_denominator = mask_weights.sum(dim=(1, 2)).clamp_min(1e-6)
            mask_xy = torch.stack(
                (
                    (mask_weights * image_grid_x).sum(dim=(1, 2)) / mask_denominator,
                    (mask_weights * image_grid_y).sum(dim=(1, 2)) / mask_denominator,
                ),
                dim=1,
            )
            feature_map = self.spatial_encoder(torch.cat((images, target_color_map), dim=1))
            color_embedding = self.color_embedding(color_ids)
            query = self.query_projection(color_embedding).unsqueeze(-1).unsqueeze(-1)
            logits = (feature_map * query).sum(dim=1) / math.sqrt(feature_map.shape[1])
            attention = torch.softmax(logits.flatten(1), dim=1).view_as(logits)
            height, width = logits.shape[-2:]
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, height, device=images.device),
                torch.linspace(-1.0, 1.0, width, device=images.device),
                indexing="ij",
            )
            attention_x = (attention * grid_x).sum(dim=(1, 2))
            attention_y = (attention * grid_y).sum(dim=(1, 2))
            attended_features = (feature_map * attention.unsqueeze(1)).sum(dim=(2, 3))
            global_features = feature_map.mean(dim=(2, 3))
            attention_xy = torch.stack((attention_x, attention_y), dim=1)
            features.extend(
                [
                    mask_xy,
                    attention_xy,
                    attended_features,
                    global_features,
                ]
            )
        if self.image_encoder is not None:
            features.append(self.image_encoder(images))
        if self.color_embedding is not None and self.destination_embedding is not None:
            features.append(self.color_embedding(color_ids))
            features.append(self.destination_embedding(destination_ids))
        prediction = self.regressor(torch.cat(features, dim=1))
        if return_attention:
            if attention_xy is None:
                raise ValueError("Attention output is only available in full mode")
            return prediction, attention_xy, logits
        return prediction


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def project_world_to_pixel(
    world_xyz: np.ndarray | list[float], camera: dict[str, Any]
) -> tuple[float, float]:
    """Project a world point with the fixed Genesis pinhole camera extrinsics/intrinsics."""
    position = np.asarray(camera["position_m"], dtype=np.float64)
    lookat = np.asarray(camera["lookat_m"], dtype=np.float64)
    up_hint = np.asarray(camera["up"], dtype=np.float64)
    forward = lookat - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    relative = np.asarray(world_xyz, dtype=np.float64) - position
    depth = float(np.dot(relative, forward))
    if depth <= 0:
        raise ValueError("World point lies behind the camera")
    width, height = camera["resolution"]
    focal = height / (2.0 * math.tan(math.radians(camera["fov_deg"]) / 2.0))
    pixel_x = width / 2.0 + focal * float(np.dot(relative, right)) / depth
    pixel_y = height / 2.0 - focal * float(np.dot(relative, camera_up)) / depth
    return pixel_x, pixel_y
