from pai_goal_conditioned.phase6_grasp_diagnostics import (
    build_probe_specs,
    classify_cause,
    contiguous_success_interval,
)

FAILURES = [
    {
        "trial": 1,
        "predicted_xy_m": [0.647, -0.193],
        "target_truth_xy_m": [0.647, -0.210],
    },
    {
        "trial": 5,
        "predicted_xy_m": [0.667, 0.037],
        "target_truth_xy_m": [0.698, 0.066],
    },
]


def test_pilot_plan_contains_only_ai_and_truth_comparisons() -> None:
    specs = build_probe_specs(FAILURES, 1, [-5, 0, 5], "pilot")
    assert len(specs) == 4
    assert {spec.command_source for spec in specs} == {"ai_replay", "diagnostic_truth"}


def test_full_plan_keeps_truth_and_offset_use_explicit() -> None:
    specs = build_probe_specs(FAILURES, 1, [-5, 0, 5], "all")
    assert len(specs) == 14
    assert sum(spec.command_source == "ai_replay" for spec in specs) == 2
    assert sum(spec.command_source == "diagnostic_truth" for spec in specs) == 2
    assert sum(spec.command_source == "phase5_error_component" for spec in specs) == 4
    assert sum(spec.command_source == "diagnostic_offset_sweep" for spec in specs) == 6


def test_contiguous_success_interval_is_observed_not_interpolated() -> None:
    points = [(-20, False), (-10, True), (0, True), (10, True), (20, False)]
    assert contiguous_success_interval(points) == [-10, 10]
    assert contiguous_success_interval([(-5, True), (0, False), (5, True)]) is None


def test_cause_classification_uses_truth_and_axis_components() -> None:
    assert classify_cause(True, False, True, False).startswith("closing-axis")
    assert classify_cause(True, False, False, True).startswith("orthogonal-axis")
    assert classify_cause(True, False, True, True).startswith("combined")
    assert classify_cause(False, False, False, False).startswith("truth grasp")
