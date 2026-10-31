from pathlib import Path

import numpy as np

from pai_goal_conditioned.phase2_pick_place import (
    DEFAULT_CONFIG,
    Stage,
    classify_failure,
    horizontal_error,
    load_config,
    supported_height_error,
)


def test_phase2_config_has_ten_trials_and_50_mm_tolerance() -> None:
    config = load_config(Path(DEFAULT_CONFIG))
    assert config["evaluation"]["trial_count"] == 10
    assert config["evaluation"]["placement_xy_tolerance_m"] == 0.05


def test_position_metrics_separate_horizontal_and_support_errors() -> None:
    actual = np.array([0.53, 0.19, 0.23])
    target = np.array([0.55, 0.16, 0.225])
    assert np.isclose(horizontal_error(actual, target), np.hypot(0.02, 0.03))
    assert np.isclose(supported_height_error(actual, target), 0.005)


def test_failure_classification_returns_first_failed_stage() -> None:
    assert classify_failure(False, False, False, False, False) == Stage.CLOSE_GRIPPER.value
    assert classify_failure(True, True, False, False, False) == Stage.TRANSPORT.value
    assert classify_failure(True, True, True, True, True) is None
