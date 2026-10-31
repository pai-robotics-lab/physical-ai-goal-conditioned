import numpy as np

from pai_goal_conditioned.phase3_dataset import (
    assign_scene_splits,
    sample_non_overlapping_positions,
    validate_label_consistency,
    validate_split_isolation,
    visibility_from_mask,
)


def test_random_positions_respect_bounds_and_separation() -> None:
    positions = sample_non_overlapping_positions(
        np.random.default_rng(3),
        count=3,
        x_range=[0.4, 0.7],
        y_range=[-0.22, 0.08],
        z=0.225,
        minimum_separation=0.1,
        max_attempts=200,
    )
    assert len(positions) == 3
    for left_index, left in enumerate(positions):
        assert 0.4 <= left[0] <= 0.7
        assert -0.22 <= left[1] <= 0.08
        for right in positions[left_index + 1 :]:
            assert np.linalg.norm(np.asarray(left[:2]) - np.asarray(right[:2])) >= 0.1


def test_scene_split_keeps_each_scene_in_one_partition() -> None:
    scene_ids = [f"scene_{index:04d}" for index in range(10)]
    assignments = assign_scene_splits(
        scene_ids, {"train": 0.8, "validation": 0.1, "test": 0.1}, seed=10
    )
    assert list(assignments.values()).count("train") == 8
    assert list(assignments.values()).count("validation") == 1
    assert list(assignments.values()).count("test") == 1


def test_visibility_extracts_bbox_and_centroid() -> None:
    mask = np.zeros((5, 6), dtype=np.int32)
    mask[1:4, 2:5] = 7
    visibility = visibility_from_mask(mask, 7)
    assert visibility.pixel_count == 9
    assert visibility.bbox_xyxy == [2, 1, 4, 3]
    assert visibility.centroid_xy == [3.0, 2.0]


def test_label_and_split_validators() -> None:
    samples = [
        {
            "sample_id": "a",
            "scene_id": "scene_0000",
            "image_path": "images/scene_0000.png",
            "split": "train",
            "grasp_position_xy_world_m": [0.5, -0.1],
            "target_truth_position_m": [0.5, -0.1, 0.225],
        },
        {
            "sample_id": "b",
            "scene_id": "scene_0000",
            "image_path": "images/scene_0000.png",
            "split": "train",
            "grasp_position_xy_world_m": [0.6, 0.0],
            "target_truth_position_m": [0.6, 0.0, 0.225],
        },
    ]
    assert validate_label_consistency(samples) == []
    assert validate_split_isolation(samples) == []
