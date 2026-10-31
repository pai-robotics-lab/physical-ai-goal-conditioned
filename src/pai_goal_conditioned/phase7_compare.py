"""Compare paired old/improved phase-7 physical evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import PROJECT_ROOT

DEFAULT_OLD = PROJECT_ROOT / "outputs" / "phase7" / "physical" / "old"
DEFAULT_IMPROVED = PROJECT_ROOT / "outputs" / "phase7" / "physical" / "improved"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "phase7" / "physical_comparison"


def summarize_physical(results: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_delta = np.asarray(
        [
            np.asarray(result["predicted_xy_m"]) - np.asarray(result["target_truth_xy_m"])
            for result in results
        ]
    ) * 1000.0
    errors = np.linalg.norm(prediction_delta, axis=1)
    success = np.asarray([result["instruction_achievement_success"] for result in results])
    successful_placement_errors = [
        result["final_placement_error_mm"]
        for result in results
        if result["instruction_achievement_success"]
    ]
    output: dict[str, Any] = {
        "trial_count": len(results),
        "correct_object_count": sum(result["correct_object_grasped"] for result in results),
        "lift_success_count": sum(result["lift_success"] for result in results),
        "transport_success_count": sum(result["transport_success"] for result in results),
        "placement_success_count": sum(result["placement_success"] for result in results),
        "instruction_success_count": int(np.sum(success)),
        "instruction_success_rate": float(np.mean(success)),
        "mean_error_mm": float(np.mean(errors)),
        "median_error_mm": float(np.median(errors)),
        "p90_error_mm": float(np.percentile(errors, 90)),
        "x_mean_absolute_error_mm": float(np.mean(np.abs(prediction_delta[:, 0]))),
        "y_mean_absolute_error_mm": float(np.mean(np.abs(prediction_delta[:, 1]))),
        "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in results])),
        "mean_physical_operation_seconds": float(
            np.mean([r["physical_operation_seconds"] for r in results])
        ),
        "mean_total_seconds": float(np.mean([r["total_seconds"] for r in results])),
        "successful_placement_error_mean_mm": float(np.mean(successful_placement_errors)),
        "successful_placement_error_max_mm": float(np.max(successful_placement_errors)),
        "by_color": {},
    }
    for color in ("red_cube", "blue_cube", "green_cube"):
        color_results = [result for result in results if result["target_class"] == color]
        color_successes = sum(
            result["instruction_achievement_success"] for result in color_results
        )
        output["by_color"][color] = {
            "success_count": color_successes,
            "trial_count": len(color_results),
            "success_rate": None if not color_results else color_successes / len(color_results),
        }
    for label, mask in (("success", success), ("failure", ~success)):
        if np.any(mask):
            output[f"{label}_mean_error_mm"] = float(np.mean(errors[mask]))
            output[f"{label}_x_mean_absolute_error_mm"] = float(
                np.mean(np.abs(prediction_delta[mask, 0]))
            )
            output[f"{label}_y_mean_absolute_error_mm"] = float(
                np.mean(np.abs(prediction_delta[mask, 1]))
            )
        else:
            output[f"{label}_mean_error_mm"] = None
            output[f"{label}_x_mean_absolute_error_mm"] = None
            output[f"{label}_y_mean_absolute_error_mm"] = None
    return output


def compare(old_dir: Path, improved_dir: Path, output_dir: Path) -> dict[str, Any]:
    old = json.loads((old_dir / "phase5_results.json").read_text(encoding="utf-8"))
    improved = json.loads(
        (improved_dir / "phase5_results.json").read_text(encoding="utf-8")
    )
    if len(old["results"]) != len(improved["results"]):
        raise ValueError("Paired physical evaluations have different trial counts")
    paired_rows: list[dict[str, Any]] = []
    old_fail_improved_success = 0
    old_success_improved_fail = 0
    maximum_initial_position_delta_mm = 0.0
    for old_result, improved_result in zip(old["results"], improved["results"]):
        identity_old = (
            old_result["trial"],
            old_result["seed"],
            old_result["target_class"],
            old_result["destination_id"],
        )
        identity_improved = (
            improved_result["trial"],
            improved_result["seed"],
            improved_result["target_class"],
            improved_result["destination_id"],
        )
        if identity_old != identity_improved:
            raise ValueError(f"Physical pair mismatch: {identity_old} != {identity_improved}")
        for object_class in old_result["initial_object_positions_m"]:
            delta = np.linalg.norm(
                np.asarray(old_result["initial_object_positions_m"][object_class])
                - np.asarray(improved_result["initial_object_positions_m"][object_class])
            ) * 1000.0
            maximum_initial_position_delta_mm = max(maximum_initial_position_delta_mm, float(delta))
        old_success = bool(old_result["instruction_achievement_success"])
        improved_success = bool(improved_result["instruction_achievement_success"])
        old_fail_improved_success += int(not old_success and improved_success)
        old_success_improved_fail += int(old_success and not improved_success)
        paired_rows.append(
            {
                "trial": old_result["trial"],
                "seed": old_result["seed"],
                "target_class": old_result["target_class"],
                "destination_id": old_result["destination_id"],
                "old_prediction_error_mm": old_result["prediction_error_mm"],
                "improved_prediction_error_mm": improved_result["prediction_error_mm"],
                "old_success": old_success,
                "improved_success": improved_success,
                "old_failed_stage": old_result["failed_stage"],
                "improved_failed_stage": improved_result["failed_stage"],
            }
        )
    payload = {
        "phase": 7,
        "comparison": "paired old phase4 model vs train_2000 phase7 model",
        "same_initial_conditions": maximum_initial_position_delta_mm <= 1e-6,
        "maximum_initial_position_delta_mm": maximum_initial_position_delta_mm,
        "old": summarize_physical(old["results"]),
        "improved": summarize_physical(improved["results"]),
        "old_failure_to_improved_success_count": old_fail_improved_success,
        "old_success_to_improved_failure_count": old_success_improved_fail,
        "paired_results": paired_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "physical_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "physical_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    _save_plot(payload, output_dir)
    return payload


def _save_plot(payload: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["old phase4", "improved 2000"]
    values = [
        payload["old"]["instruction_success_rate"] * 100.0,
        payload["improved"]["instruction_success_rate"] * 100.0,
    ]
    figure, axis = plt.subplots(figsize=(6, 5))
    bars = axis.bar(labels, values, color=["#756bb1", "#31a354"])
    axis.bar_label(bars, fmt="%.1f%%")
    axis.axhline(90.0, color="#d7301f", linestyle="--", label="90% target")
    axis.set_ylim(0, 105)
    axis.set_ylabel("instruction success rate")
    axis.set_title("Phase 7 paired physical operation comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "physical_success_comparison.png", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--improved-dir", type=Path, default=DEFAULT_IMPROVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare(args.old_dir.resolve(), args.improved_dir.resolve(), args.output_dir.resolve())
    print(
        f"Phase 7 physical: old={payload['old']['instruction_success_count']}/"
        f"{payload['old']['trial_count']} improved="
        f"{payload['improved']['instruction_success_count']}/"
        f"{payload['improved']['trial_count']}"
    )


if __name__ == "__main__":
    main()
