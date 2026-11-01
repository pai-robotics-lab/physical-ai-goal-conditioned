"""Cross-check final-report numbers against the phase-7 machine-readable results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance


def verify() -> dict[str, Any]:
    phase4 = _json("outputs/phase4/evaluation_summary.json")
    offline = _json("outputs/phase7/offline/evaluation_summary.json")
    physical = _json("outputs/phase7/physical_comparison/physical_comparison.json")
    manifest = _json("datasets/phase7/manifest.json")
    with (ROOT / "outputs/phase7/offline/offline_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        csv_rows = {row["model"]: row for row in csv.DictReader(stream)}

    expected_offline = {
        "train_80": (12.478469848632812, 11.798794746398926, 22.157306671142578),
        "train_500": (3.402360439300537, 3.013808250427246, 6.095652103424072),
        "train_2000": (2.1374223232269287, 1.975610375404358, 3.666198492050171),
    }
    for model, expected in expected_offline.items():
        metrics = offline["evaluation"][model]
        actual = (metrics["mean_error_mm"], metrics["median_error_mm"], metrics["p90_error_mm"])
        assert all(_close(left, right) for left, right in zip(actual, expected))
        csv_actual = tuple(float(csv_rows[model][key]) for key in ("mean_mm", "median_mm", "p90_mm"))
        assert all(_close(left, right) for left, right in zip(actual, csv_actual))

    assert offline["test_used_for_model_selection"] is False
    assert offline["test_scene_count"] == 250
    assert offline["test_sample_count"] == 1500
    assert manifest["scene_count"] == 2500
    assert manifest["sample_count"] == 15000
    assert manifest["split_scene_counts"] == {"train": 2000, "validation": 250, "test": 250}
    assert manifest["all_labels_valid"] is True
    assert manifest["split_leakage_count"] == 0

    assert physical["same_initial_conditions"] is True
    assert physical["maximum_initial_position_delta_mm"] == 0.0
    assert physical["old"]["instruction_success_count"] == 15
    assert physical["improved"]["instruction_success_count"] == 24
    assert physical["old_failure_to_improved_success_count"] == 9
    assert physical["old_success_to_improved_failure_count"] == 0

    phase4_expected = {
        "full": 15.250306129455566,
        "instruction_only": 126.52632141113281,
        "image_only": 127.82539367675781,
    }
    for model, expected in phase4_expected.items():
        actual = phase4["evaluation"][model]["test"]["mean_error_mm"]
        assert _close(actual, expected)

    published_text = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in ("README.md", "docs/final-report.md")
    )
    required_values = (
        "12.478",
        "3.402",
        "2.137",
        "15/24",
        "24/24",
        "62.5%",
        "100%",
        "49,366",
        "15.250",
        "126.526",
        "127.825",
    )
    assert all(value in published_text for value in required_values)
    required_explanations = (
        "視覚AIを用いたPandaロボットアームの物体把持・搬送",
        "把持位置XY",
        "搬送先ID",
        "従来型制御",
        "Data Scaling",
    )
    assert all(text in published_text for text in required_explanations)
    assert "Goal-Conditioned Visual Robotic Manipulation" not in published_text

    return {
        "offline_models_checked": list(expected_offline),
        "offline_csv_matches_json": True,
        "dataset_scene_sample_counts": [manifest["scene_count"], manifest["sample_count"]],
        "split_leakage_count": manifest["split_leakage_count"],
        "paired_physical_success": {"old": "15/24", "improved": "24/24"},
        "phase4_ablation_mean_error_mm": {
            model: round(value, 3) for model, value in phase4_expected.items()
        },
        "published_values_present": True,
        "published_role_explanations_present": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
