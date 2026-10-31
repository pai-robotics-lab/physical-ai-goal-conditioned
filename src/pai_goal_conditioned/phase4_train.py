"""Train and offline-evaluate Goal-conditioned grasp XY models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .goal_vision import (
    COLOR_TO_ID,
    DESTINATION_TO_ID,
    GoalConditionedGraspModel,
    TargetNormalizer,
    count_parameters,
    load_rgb_tensor,
    project_world_to_pixel,
)
from .runtime import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase4_training.json"


@dataclass
class TrainingResult:
    mode: str
    parameter_count: int
    best_epoch: int
    epochs_ran: int
    best_validation_loss: float
    training_seconds: float
    peak_gpu_allocated_bytes: int
    peak_gpu_reserved_bytes: int
    weight_path: str
    history_path: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dataset_version(dataset_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("labels.jsonl", "manifest.json", "phase3_config_snapshot.json"):
        digest.update((dataset_dir / name).read_bytes())
    return digest.hexdigest()


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_train_image_stats(
    records: list[dict[str, Any]], dataset_dir: Path, size_wh: tuple[int, int]
) -> tuple[list[float], list[float]]:
    """Calculate RGB mean/std from unique Train images only."""
    image_paths = sorted({record["image_path"] for record in records})
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for relative_path in image_paths:
        image = Image.open(dataset_dir / relative_path).convert("RGB")
        image = image.resize(size_wh, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float64) / 255.0
        channel_sum += array.sum(axis=(0, 1))
        channel_square_sum += np.square(array).sum(axis=(0, 1))
        pixel_count += array.shape[0] * array.shape[1]
    mean = channel_sum / pixel_count
    variance = channel_square_sum / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-8))
    return mean.tolist(), std.tolist()


class Phase4Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        dataset_dir: Path,
        image_size_wh: tuple[int, int],
        image_mean: list[float],
        image_std: list[float],
        target_normalizer: TargetNormalizer,
        image_cache: dict[str, torch.Tensor],
    ) -> None:
        self.records = records
        self.dataset_dir = dataset_dir
        self.image_size_wh = image_size_wh
        self.image_mean = image_mean
        self.image_std = image_std
        self.target_normalizer = target_normalizer
        self.image_cache = image_cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        relative_path = record["image_path"]
        if relative_path not in self.image_cache:
            self.image_cache[relative_path] = load_rgb_tensor(
                self.dataset_dir / relative_path,
                self.image_size_wh,
                self.image_mean,
                self.image_std,
            )
        target_world = np.asarray(record["grasp_position_xy_world_m"], dtype=np.float32)
        target_normalized = self.target_normalizer.normalize_array(target_world)
        pixel_x, pixel_y = project_world_to_pixel(
            record["target_truth_position_m"], record["camera"]
        )
        camera_width, camera_height = record["camera"]["resolution"]
        attention_target = np.asarray(
            [
                2.0 * pixel_x / (camera_width - 1) - 1.0,
                2.0 * pixel_y / (camera_height - 1) - 1.0,
            ],
            dtype=np.float32,
        )
        attention_column = int(
            np.clip(round((attention_target[0] + 1.0) * 0.5 * 39), 0, 39)
        )
        attention_row = int(
            np.clip(round((attention_target[1] + 1.0) * 0.5 * 29), 0, 29)
        )
        return {
            "image": self.image_cache[relative_path],
            "color_id": COLOR_TO_ID[record["target_class"]],
            "destination_id": DESTINATION_TO_ID[record["transport_target_id"]],
            "target_normalized": torch.from_numpy(target_normalized),
            "target_world": torch.from_numpy(target_world),
            "attention_target": torch.from_numpy(attention_target),
            "attention_cell_index": attention_row * 40 + attention_column,
            "sample_id": record["sample_id"],
        }


def build_model(
    config: dict[str, Any],
    mode: str,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> GoalConditionedGraspModel:
    return GoalConditionedGraspModel(
        mode=mode,
        image_feature_dim=config["image_feature_dim"],
        color_embedding_dim=config["color_embedding_dim"],
        destination_embedding_dim=config["destination_embedding_dim"],
        dropout=config["dropout"],
        image_mean=image_mean,
        image_std=image_std,
    )


def make_loader(
    dataset: Phase4Dataset, batch_size: int, shuffle: bool, seed: int
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


def _validation_loss(
    model: GoalConditionedGraspModel,
    loader: DataLoader[dict[str, Any]],
    loss_function: nn.Module,
    attention_loss_weight: float,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            colors = batch["color_id"].to(device, non_blocking=True)
            destinations = batch["destination_id"].to(device, non_blocking=True)
            targets = batch["target_normalized"].to(device, non_blocking=True)
            if model.mode == "full":
                predictions, _, attention_logits = model(
                    images, colors, destinations, return_attention=True
                )
                attention_indices = batch["attention_cell_index"].to(
                    device, non_blocking=True
                )
                loss = loss_function(predictions, targets) + attention_loss_weight * nn.functional.cross_entropy(
                    attention_logits.flatten(1), attention_indices
                )
            else:
                predictions = model(images, colors, destinations)
                loss = loss_function(predictions, targets)
            total_loss += float(loss) * len(images)
            total_count += len(images)
    return total_loss / total_count


def train_one_model(
    mode: str,
    config: dict[str, Any],
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    model_dir: Path,
    output_dir: Path,
    image_mean: list[float],
    image_std: list[float],
) -> TrainingResult:
    model = build_model(config, mode, image_mean, image_std).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["lr_scheduler_factor"],
        patience=config["lr_scheduler_patience"],
    )
    loss_function = nn.SmoothL1Loss(beta=config["smooth_l1_beta"])
    weight_path = model_dir / f"{mode}_best_state_dict.pt"
    history_path = output_dir / "histories" / f"{mode}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            colors = batch["color_id"].to(device, non_blocking=True)
            destinations = batch["destination_id"].to(device, non_blocking=True)
            targets = batch["target_normalized"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if model.mode == "full":
                predictions, _, attention_logits = model(
                    images, colors, destinations, return_attention=True
                )
                attention_indices = batch["attention_cell_index"].to(
                    device, non_blocking=True
                )
                loss = loss_function(predictions, targets) + config[
                    "attention_loss_weight"
                ] * nn.functional.cross_entropy(attention_logits.flatten(1), attention_indices)
            else:
                predictions = model(images, colors, destinations)
                loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(images)
            train_count += len(images)
        train_loss = train_loss_sum / train_count
        validation_loss = _validation_loss(
            model,
            validation_loader,
            loss_function,
            config["attention_loss_weight"],
            device,
        )
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"{mode} epoch={epoch:03d} train={train_loss:.6f} "
            f"validation={validation_loss:.6f} lr={learning_rate:.2e}",
            flush=True,
        )
        if validation_loss < best_loss - config["early_stopping_min_delta"]:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), weight_path)
        else:
            stale_epochs += 1
        if stale_epochs >= config["early_stopping_patience"]:
            break

    elapsed = time.perf_counter() - started
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return TrainingResult(
        mode=mode,
        parameter_count=count_parameters(model),
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_loss=best_loss,
        training_seconds=elapsed,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
        weight_path=str(weight_path.resolve()),
        history_path=str(history_path.resolve()),
    )


def predict(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    normalizer: TargetNormalizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    predicted: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            colors = batch["color_id"].to(device, non_blocking=True)
            destinations = batch["destination_id"].to(device, non_blocking=True)
            normalized = model(images, colors, destinations)
            predicted.append(normalizer.denormalize_tensor(normalized).cpu().numpy())
            targets.append(batch["target_world"].numpy())
            sample_ids.extend(batch["sample_id"])
    return np.concatenate(predicted), np.concatenate(targets), sample_ids


def calculate_metrics(
    predicted: np.ndarray, targets: np.ndarray, records_by_id: dict[str, dict[str, Any]], sample_ids: list[str]
) -> dict[str, Any]:
    errors_mm = np.linalg.norm(predicted - targets, axis=1) * 1000.0
    by_color: dict[str, list[float]] = defaultdict(list)
    by_destination: dict[str, list[float]] = defaultdict(list)
    for sample_id, error in zip(sample_ids, errors_mm):
        record = records_by_id[sample_id]
        by_color[record["target_class"]].append(float(error))
        by_destination[record["transport_target_id"]].append(float(error))
    return {
        "count": len(errors_mm),
        "mean_error_mm": float(np.mean(errors_mm)),
        "median_error_mm": float(np.median(errors_mm)),
        "p90_error_mm": float(np.percentile(errors_mm, 90)),
        "within_10mm_fraction": float(np.mean(errors_mm <= 10.0)),
        "within_20mm_fraction": float(np.mean(errors_mm <= 20.0)),
        "by_color_mean_error_mm": {
            key: float(np.mean(values)) for key, values in sorted(by_color.items())
        },
        "by_destination_mean_error_mm": {
            key: float(np.mean(values)) for key, values in sorted(by_destination.items())
        },
        "errors_mm": errors_mm.tolist(),
    }


def conditioning_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
    sample_ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
) -> dict[str, float]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        record = records_by_id[sample_id]
        groups[(record["scene_id"], record["transport_target_id"])].append(index)
    predicted_distances: list[float] = []
    target_distances: list[float] = []
    vector_errors: list[float] = []
    for indices in groups.values():
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                predicted_delta = predicted[left] - predicted[right]
                target_delta = targets[left] - targets[right]
                predicted_distances.append(float(np.linalg.norm(predicted_delta) * 1000.0))
                target_distances.append(float(np.linalg.norm(target_delta) * 1000.0))
                vector_errors.append(float(np.linalg.norm(predicted_delta - target_delta) * 1000.0))
    return {
        "mean_predicted_pairwise_change_mm": float(np.mean(predicted_distances)),
        "mean_ground_truth_pairwise_change_mm": float(np.mean(target_distances)),
        "mean_pairwise_change_vector_error_mm": float(np.mean(vector_errors)),
    }


def write_predictions(
    path: Path,
    predicted: np.ndarray,
    targets: np.ndarray,
    sample_ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction, target, sample_id in zip(predicted, targets, sample_ids):
            record = records_by_id[sample_id]
            payload = {
                "sample_id": sample_id,
                "scene_id": record["scene_id"],
                "instruction": record["instruction"],
                "prediction_xy_world_m": prediction.tolist(),
                "target_xy_world_m": target.tolist(),
                "error_mm": float(np.linalg.norm(prediction - target) * 1000.0),
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _draw_cross(draw: ImageDraw.ImageDraw, point: tuple[float, float], color: str, size: int = 8) -> None:
    x, y = point
    draw.line((x - size, y, x + size, y), fill=color, width=3)
    draw.line((x, y - size, x, y + size), fill=color, width=3)


def create_visualizations(
    config: dict[str, Any],
    dataset_dir: Path,
    output_dir: Path,
    evaluation: dict[str, Any],
    predictions_by_mode: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    test_records_by_id: dict[str, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(config["model_modes"]), figsize=(15, 4))
    for axis, mode in zip(axes, config["model_modes"]):
        history = json.loads((output_dir / "histories" / f"{mode}.json").read_text())
        axis.plot([item["epoch"] for item in history], [item["train_loss"] for item in history], label="train")
        axis.plot(
            [item["epoch"] for item in history],
            [item["validation_loss"] for item in history],
            label="validation",
        )
        axis.set_title(mode)
        axis.set_xlabel("epoch")
        axis.set_ylabel("Smooth L1")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(visualization_dir / "loss_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for mode in config["model_modes"]:
        axis.hist(evaluation[mode]["test"]["errors_mm"], bins=15, alpha=0.5, label=mode)
    axis.axvline(20.0, color="black", linestyle="--", label="20 mm target")
    axis.set_xlabel("test XY error (mm)")
    axis.set_ylabel("samples")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(visualization_dir / "test_error_distribution.png", dpi=160)
    plt.close(figure)

    modes = config["model_modes"]
    means = [evaluation[mode]["test"]["mean_error_mm"] for mode in modes]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(modes, means, color=["#3575d3", "#d38135", "#6f6f6f"])
    axis.axhline(20.0, color="black", linestyle="--", label="20 mm target")
    axis.set_ylabel("mean test XY error (mm)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(visualization_dir / "baseline_comparison.png", dpi=160)
    plt.close(figure)

    predicted, targets, sample_ids = predictions_by_mode["full"]
    examples_dir = visualization_dir / "prediction_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(config["visualization_examples"], len(sample_ids))):
        sample_id = sample_ids[index]
        record = test_records_by_id[sample_id]
        image = Image.open(dataset_dir / record["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        z = record["target_truth_position_m"][2]
        target_pixel = project_world_to_pixel([targets[index, 0], targets[index, 1], z], record["camera"])
        prediction_pixel = project_world_to_pixel(
            [predicted[index, 0], predicted[index, 1], z], record["camera"]
        )
        _draw_cross(draw, target_pixel, "#00ff55")
        _draw_cross(draw, prediction_pixel, "#ff3030")
        draw.text((10, 10), f"{record['instruction']}  green=truth red=prediction", fill="white")
        image.save(examples_dir / f"{sample_id}.png")

    first_scene = test_records_by_id[sample_ids[0]]["scene_id"]
    comparison_indices = [
        index
        for index, sample_id in enumerate(sample_ids)
        if test_records_by_id[sample_id]["scene_id"] == first_scene
        and test_records_by_id[sample_id]["transport_target_id"] == "left_bin"
    ]
    first_record = test_records_by_id[sample_ids[comparison_indices[0]]]
    image = Image.open(dataset_dir / first_record["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"red_cube": "#ff3030", "blue_cube": "#3080ff", "green_cube": "#30e060"}
    for index in comparison_indices:
        record = test_records_by_id[sample_ids[index]]
        color = colors[record["target_class"]]
        z = record["target_truth_position_m"][2]
        target_pixel = project_world_to_pixel([targets[index, 0], targets[index, 1], z], record["camera"])
        prediction_pixel = project_world_to_pixel(
            [predicted[index, 0], predicted[index, 1], z], record["camera"]
        )
        draw.ellipse(
            (target_pixel[0] - 6, target_pixel[1] - 6, target_pixel[0] + 6, target_pixel[1] + 6),
            outline=color,
            width=3,
        )
        _draw_cross(draw, prediction_pixel, color)
        draw.text((prediction_pixel[0] + 8, prediction_pixel[1] - 10), record["target_class"], fill=color)
    draw.text((10, 10), "circles=truth crosses=predictions; same RGB, different color goals", fill="white")
    image.save(visualization_dir / "same_image_color_conditioning.png")


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = (PROJECT_ROOT / config["dataset_dir"]).resolve()
    records = load_jsonl(dataset_dir / "labels.jsonl")
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    train_targets = np.asarray(
        [record["grasp_position_xy_world_m"] for record in split_records["train"]],
        dtype=np.float32,
    )
    normalizer = TargetNormalizer.fit(train_targets)
    image_size_wh = tuple(config["image_size_wh"])
    image_mean, image_std = compute_train_image_stats(
        split_records["train"], dataset_dir, image_size_wh
    )
    image_cache: dict[str, torch.Tensor] = {}
    datasets = {
        split: Phase4Dataset(
            split_records[split],
            dataset_dir,
            image_size_wh,
            image_mean,
            image_std,
            normalizer,
            image_cache,
        )
        for split in split_records
    }
    return {
        "dataset_dir": dataset_dir,
        "records": records,
        "split_records": split_records,
        "normalizer": normalizer,
        "image_mean": image_mean,
        "image_std": image_std,
        "datasets": datasets,
        "version": dataset_version(dataset_dir),
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4 training requires the CUDA training environment")
    set_reproducible(config["seed"])
    device = torch.device("cuda")
    prepared = prepare_data(config)
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    model_dir = (PROJECT_ROOT / config["model_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(
        prepared["datasets"]["train"], config["batch_size"], True, config["seed"]
    )
    validation_loader = make_loader(
        prepared["datasets"]["validation"], config["batch_size"], False, config["seed"]
    )
    results: list[TrainingResult] = []
    validation_metrics: dict[str, Any] = {}
    validation_records = prepared["split_records"]["validation"]
    validation_by_id = {record["sample_id"]: record for record in validation_records}
    for model_index, mode in enumerate(config["model_modes"]):
        set_reproducible(config["seed"] + model_index)
        result = train_one_model(
            mode,
            config,
            train_loader,
            validation_loader,
            device,
            model_dir,
            output_dir,
            prepared["image_mean"],
            prepared["image_std"],
        )
        results.append(result)
        model = build_model(
            config, mode, prepared["image_mean"], prepared["image_std"]
        ).to(device)
        model.load_state_dict(torch.load(result.weight_path, map_location=device, weights_only=True))
        predicted, targets, sample_ids = predict(
            model, validation_loader, prepared["normalizer"], device
        )
        validation_metrics[mode] = calculate_metrics(
            predicted, targets, validation_by_id, sample_ids
        )
        validation_metrics[mode]["conditioning"] = conditioning_metrics(
            predicted, targets, sample_ids, validation_by_id
        )
        del model
        torch.cuda.empty_cache()

    normalization = {
        "target": prepared["normalizer"].to_dict(),
        "image_mean": prepared["image_mean"],
        "image_std": prepared["image_std"],
        "image_size_wh": config["image_size_wh"],
        "computed_from": "train split only",
    }
    (model_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )
    training_summary = {
        "stage": "train_and_validation_only",
        "test_data_accessed": False,
        "seed": config["seed"],
        "device": torch.cuda.get_device_name(0),
        "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__,
        "dataset_version_sha256": prepared["version"],
        "split_counts": {
            split: len(records) for split, records in prepared["split_records"].items()
        },
        "normalization": normalization,
        "config": config,
        "training_results": [asdict(result) for result in results],
        "validation_metrics": validation_metrics,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return training_summary


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4 evaluation requires the CUDA training environment")
    set_reproducible(config["seed"])
    device = torch.device("cuda")
    prepared = prepare_data(config)
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    model_dir = (PROJECT_ROOT / config["model_dir"]).resolve()
    training_summary = json.loads(
        (output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    if training_summary["dataset_version_sha256"] != prepared["version"]:
        raise RuntimeError("Dataset version changed after model selection")
    evaluation: dict[str, Any] = {}
    predictions_by_mode: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    records_by_split = {
        split: {record["sample_id"]: record for record in prepared["split_records"][split]}
        for split in ("validation", "test")
    }
    for mode in config["model_modes"]:
        model = build_model(
            config, mode, prepared["image_mean"], prepared["image_std"]
        ).to(device)
        weight_path = model_dir / f"{mode}_best_state_dict.pt"
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        evaluation[mode] = {}
        for split in ("validation", "test"):
            loader = make_loader(
                prepared["datasets"][split], config["batch_size"], False, config["seed"]
            )
            predicted, targets, sample_ids = predict(
                model, loader, prepared["normalizer"], device
            )
            metrics = calculate_metrics(
                predicted, targets, records_by_split[split], sample_ids
            )
            metrics["conditioning"] = conditioning_metrics(
                predicted, targets, sample_ids, records_by_split[split]
            )
            evaluation[mode][split] = metrics
            write_predictions(
                output_dir / "predictions" / f"{mode}_{split}.jsonl",
                predicted,
                targets,
                sample_ids,
                records_by_split[split],
            )
            if split == "test":
                predictions_by_mode[mode] = (predicted, targets, sample_ids)
        del model
        torch.cuda.empty_cache()

    create_visualizations(
        config,
        prepared["dataset_dir"],
        output_dir,
        evaluation,
        predictions_by_mode,
        records_by_split["test"],
    )
    comparison_path = output_dir / "baseline_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "split", "mean_mm", "median_mm", "p90_mm", "within_10", "within_20"]
        )
        for mode in config["model_modes"]:
            for split in ("validation", "test"):
                metrics = evaluation[mode][split]
                writer.writerow(
                    [
                        mode,
                        split,
                        metrics["mean_error_mm"],
                        metrics["median_error_mm"],
                        metrics["p90_error_mm"],
                        metrics["within_10mm_fraction"],
                        metrics["within_20mm_fraction"],
                    ]
                )
    summary = {
        "stage": "fixed_model_offline_evaluation",
        "test_used_for_model_selection": False,
        "dataset_version_sha256": prepared["version"],
        "evaluation": evaluation,
        "goal_mean_test_error_mm": 20.0,
        "goal_achieved": evaluation["full"]["test"]["mean_error_mm"] <= 20.0,
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if args.stage == "train":
        result = run_training(config)
        for mode, metrics in result["validation_metrics"].items():
            print(f"{mode} validation mean error: {metrics['mean_error_mm']:.3f} mm")
    else:
        result = run_evaluation(config)
        for mode, metrics in result["evaluation"].items():
            print(f"{mode} test mean error: {metrics['test']['mean_error_mm']:.3f} mm")
        print(f"20 mm goal: {'PASS' if result['goal_achieved'] else 'FAIL'}")


if __name__ == "__main__":
    main()
