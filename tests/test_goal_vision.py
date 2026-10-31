import numpy as np
import torch
from PIL import Image

from pai_goal_conditioned.goal_vision import (
    GoalConditionedGraspModel,
    TargetNormalizer,
    load_rgb_tensor,
    preprocess_rgb_array,
    project_world_to_pixel,
)
from pai_goal_conditioned.runtime import PROJECT_ROOT


def test_target_normalizer_round_trip() -> None:
    targets = np.array([[0.4, -0.2], [0.7, 0.08], [0.55, -0.05]], dtype=np.float32)
    normalizer = TargetNormalizer.fit(targets)
    normalized = normalizer.normalize_array(targets)
    restored = normalizer.denormalize_tensor(torch.from_numpy(normalized)).numpy()
    assert np.allclose(restored, targets)


def test_all_model_modes_return_xy() -> None:
    images = torch.zeros((2, 3, 120, 160))
    colors = torch.tensor([0, 2])
    destinations = torch.tensor([0, 1])
    for mode in ("full", "instruction_only", "image_only"):
        model = GoalConditionedGraspModel(mode=mode)
        result = model(images, colors, destinations)
        assert result.shape == (2, 2)
        assert torch.all(result.abs() <= 1.0)


def test_camera_projection_matches_phase3_reference() -> None:
    camera = {
        "resolution": [640, 480],
        "position_m": [1.25, -1.1, 1.05],
        "lookat_m": [0.53, -0.04, 0.22],
        "up": [0.0, 0.0, 1.0],
        "fov_deg": 43.0,
    }
    pixel = project_world_to_pixel([0.5057913, -0.1643869, 0.2249455], camera)
    assert np.allclose(pixel, [282.1943, 258.6650], atol=0.01)


def test_file_and_genesis_array_preprocessing_match() -> None:
    rgb = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    image_path = PROJECT_ROOT / "outputs" / "test-fixtures" / "preprocess-frame.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(image_path)
    arguments = ((16, 12), [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    from_file = load_rgb_tensor(image_path, *arguments)
    from_array = preprocess_rgb_array(rgb, *arguments)
    assert torch.equal(from_file, from_array)
