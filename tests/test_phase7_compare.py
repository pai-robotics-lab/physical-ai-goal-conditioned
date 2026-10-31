from pai_goal_conditioned.phase7_compare import summarize_physical


def _result(dx: float, dy: float, success: bool) -> dict[str, object]:
    return {
        "predicted_xy_m": [0.5 + dx, -0.1 + dy],
        "target_truth_xy_m": [0.5, -0.1],
        "correct_object_grasped": success,
        "lift_success": success,
        "transport_success": success,
        "placement_success": success,
        "instruction_achievement_success": success,
        "target_class": "red_cube",
        "final_placement_error_mm": 3.0 if success else 300.0,
        "inference_seconds": 0.01,
        "physical_operation_seconds": 10.0,
        "total_seconds": 10.5,
    }


def test_summarize_physical_reports_stage_counts_and_axes() -> None:
    summary = summarize_physical([_result(0.003, 0.004, True), _result(0.012, 0.0, False)])
    assert summary["instruction_success_count"] == 1
    assert summary["instruction_success_rate"] == 0.5
    assert summary["correct_object_count"] == 1
    assert summary["x_mean_absolute_error_mm"] > summary["y_mean_absolute_error_mm"]
