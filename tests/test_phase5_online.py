import inspect
import json

import numpy as np
import torch

from pai_goal_conditioned.goal_vision import GoalConditionedGraspModel
from pai_goal_conditioned.phase5_online import (
    GraspPredictor,
    make_control_plan,
    workspace_contains,
)
from pai_goal_conditioned.runtime import PROJECT_ROOT


def test_workspace_rejects_out_of_range_prediction() -> None:
    assert workspace_contains([0.55, -0.1], [0.38, 0.72], [-0.24, 0.10])
    assert not workspace_contains([0.80, -0.1], [0.38, 0.72], [-0.24, 0.10])
    assert not workspace_contains([0.55, 0.20], [0.38, 0.72], [-0.24, 0.10])


def test_control_plan_uses_prediction_without_truth_correction() -> None:
    predicted = [0.512, -0.073]
    destination = [0.42, 0.22]
    plan = make_control_plan(
        predicted,
        destination,
        supported_cube_center_z_m=0.225,
        grasp_config={
            "hand_to_cube_center_z_offset_m": 0.110,
            "pregrasp_to_cube_center_z_offset_m": 0.255,
        },
        motion_config={
            "lift_hand_position_m": [0.55, -0.12, 0.53],
            "retreat_hand_position_m": [0.55, 0.16, 0.50],
        },
    )
    assert np.allclose(plan.grasp_hand_position_m, [0.512, -0.073, 0.335])
    assert plan.predicted_grasp_xy_m == predicted
    assert plan.destination_xy_m == destination


def test_predictor_boundary_has_no_truth_or_scene_input() -> None:
    parameters = inspect.signature(GraspPredictor.predict).parameters
    assert set(parameters) == {"self", "rgb", "target_class", "destination_id"}


def test_predictor_loads_plain_state_dict() -> None:
    model_config = {
        "image_feature_dim": 128,
        "color_embedding_dim": 8,
        "destination_embedding_dim": 4,
        "dropout": 0.1,
    }
    normalization = {
        "target": {"minimum_xy_m": [0.4, -0.2], "maximum_xy_m": [0.7, 0.08]},
        "image_mean": [0.1, 0.2, 0.3],
        "image_std": [0.4, 0.5, 0.6],
        "image_size_wh": [160, 120],
    }
    model = GoalConditionedGraspModel(
        mode="full",
        image_mean=normalization["image_mean"],
        image_std=normalization["image_std"],
    )
    fixture_dir = PROJECT_ROOT / "outputs" / "test-fixtures" / "predictor"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    config_path = fixture_dir / "config.json"
    normalization_path = fixture_dir / "normalization.json"
    state_dict_path = fixture_dir / "weights.pt"
    config_path.write_text(json.dumps(model_config), encoding="utf-8")
    normalization_path.write_text(json.dumps(normalization), encoding="utf-8")
    torch.save(model.state_dict(), state_dict_path)
    predictor = GraspPredictor.from_artifacts(
        config_path, normalization_path, state_dict_path
    )
    prediction = predictor.predict(
        np.zeros((480, 640, 3), dtype=np.uint8), "red_cube", "left_bin"
    )
    assert len(prediction.predicted_xy_m) == 2
