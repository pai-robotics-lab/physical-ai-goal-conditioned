import numpy as np

from pai_goal_conditioned.phase7_train import (
    extended_metrics,
    nested_train_scene_subsets,
    seed_ranges_overlap,
)


def test_nested_subsets_are_deterministic_and_nested() -> None:
    scenes = [f"scene_{index:04d}" for index in range(20)]
    first = nested_train_scene_subsets(scenes, [5, 10, 20], 7)
    second = nested_train_scene_subsets(scenes, [5, 10, 20], 7)
    assert first == second
    assert set(first[5]) < set(first[10]) < set(first[20])


def test_seed_range_overlap_detects_excluded_evaluation_seeds() -> None:
    assert not seed_ranges_overlap((70000, 82499), (52000, 52009))
    assert seed_ranges_overlap((52000, 52020), (52000, 52009))


def test_extended_metrics_include_15mm_and_directional_errors() -> None:
    targets = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    predicted = np.asarray([[0.003, 0.004], [0.012, 0.016]], dtype=np.float32)
    sample_ids = ["a", "b"]
    records = {
        "a": {"target_class": "red_cube"},
        "b": {"target_class": "red_cube"},
    }
    for color in ("blue_cube", "green_cube"):
        sample_id = color
        sample_ids.append(sample_id)
        targets = np.vstack([targets, [0.0, 0.0]])
        predicted = np.vstack([predicted, [0.0, 0.0]])
        records[sample_id] = {"target_class": color}
    metrics = extended_metrics(predicted, targets, sample_ids, records)
    assert metrics["within_10mm_fraction"] == 0.75
    assert metrics["within_15mm_fraction"] == 0.75
    assert metrics["within_20mm_fraction"] == 1.0
    assert metrics["axis_error_mm"]["x_orthogonal"]["mean_absolute"] > 0.0
