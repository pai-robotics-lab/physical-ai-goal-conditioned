"""Train fixed phase-4 models on nested phase-7 scene subsets and evaluate once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .goal_vision import GoalConditionedGraspModel, TargetNormalizer, count_parameters
from .phase4_train import (
    Phase4Dataset,
    build_model,
    compute_train_image_stats,
    load_jsonl,
    make_loader,
    predict,
    set_reproducible,
    train_one_model,
    write_predictions,
)
from .runtime import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase7_training.json"


def nested_train_scene_subsets(
    scene_ids: list[str], counts: list[int], seed: int
) -> dict[int, list[str]]:
    """Create deterministic nested subsets, preserving the data-only comparison."""
    if not counts or min(counts) <= 0 or max(counts) > len(scene_ids):
        raise ValueError("Train scene counts must be positive and fit the train pool")
    shuffled = list(scene_ids)
    np.random.default_rng(seed).shuffle(shuffled)
    return {count: sorted(shuffled[:count]) for count in sorted(set(counts))}


def seed_ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def extended_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
    sample_ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    delta_mm = (predicted - targets) * 1000.0
    errors_mm = np.linalg.norm(delta_mm, axis=1)
    result: dict[str, Any] = {
        "count": len(errors_mm),
        "mean_error_mm": float(np.mean(errors_mm)),
        "median_error_mm": float(np.median(errors_mm)),
        "p90_error_mm": float(np.percentile(errors_mm, 90)),
        "within_10mm_fraction": float(np.mean(errors_mm <= 10.0 + 1e-6)),
        "within_15mm_fraction": float(np.mean(errors_mm <= 15.0 + 1e-6)),
        "within_20mm_fraction": float(np.mean(errors_mm <= 20.0 + 1e-6)),
        "axis_error_mm": {},
        "by_color_mean_error_mm": {},
    }
    for axis_index, axis_name in enumerate(("x_orthogonal", "y_closing")):
        signed = delta_mm[:, axis_index]
        absolute = np.abs(signed)
        result["axis_error_mm"][axis_name] = {
            "signed_mean": float(np.mean(signed)),
            "mean_absolute": float(np.mean(absolute)),
            "median_absolute": float(np.median(absolute)),
            "p90_absolute": float(np.percentile(absolute, 90)),
        }
    for color in ("red_cube", "blue_cube", "green_cube"):
        indices = [
            index
            for index, sample_id in enumerate(sample_ids)
            if records_by_id[sample_id]["target_class"] == color
        ]
        result["by_color_mean_error_mm"][color] = float(np.mean(errors_mm[indices]))
    return result


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _subset_digest(scene_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(scene_ids).encode()).hexdigest()


def _make_dataset(
    records: list[dict[str, Any]],
    dataset_dir: Path,
    config: dict[str, Any],
    image_mean: list[float],
    image_std: list[float],
    normalizer: TargetNormalizer,
    cache: dict[str, torch.Tensor],
) -> Phase4Dataset:
    return Phase4Dataset(
        records,
        dataset_dir,
        tuple(config["image_size_wh"]),
        image_mean,
        image_std,
        normalizer,
        cache,
    )


def _normalization_for_records(
    records: list[dict[str, Any]], dataset_dir: Path, config: dict[str, Any]
) -> tuple[TargetNormalizer, list[float], list[float]]:
    targets = np.asarray(
        [record["grasp_position_xy_world_m"] for record in records], dtype=np.float32
    )
    normalizer = TargetNormalizer.fit(targets)
    mean, std = compute_train_image_stats(records, dataset_dir, tuple(config["image_size_wh"]))
    return normalizer, mean, std


def run_training(config_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7 training requires the CUDA training environment")
    config = _load_config(config_path)
    set_reproducible(config["seed"])
    device = torch.device("cuda")
    dataset_dir = PROJECT_ROOT / config["dataset_dir"]
    output_dir = PROJECT_ROOT / config["output_dir"]
    model_root = PROJECT_ROOT / config["model_dir"]
    records = load_jsonl(dataset_dir / "labels.jsonl")
    train_pool = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    train_scene_ids = sorted({record["scene_id"] for record in train_pool})
    subsets = nested_train_scene_subsets(
        train_scene_ids, config["train_scene_counts"], config["subset_seed"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_scene_subsets.json").write_text(
        json.dumps({str(key): value for key, value in subsets.items()}, indent=2),
        encoding="utf-8",
    )
    results: dict[str, Any] = {}
    for model_index, scene_count in enumerate(config["train_scene_counts"]):
        set_reproducible(config["seed"] + model_index)
        selected = set(subsets[scene_count])
        train_records = [record for record in train_pool if record["scene_id"] in selected]
        normalizer, image_mean, image_std = _normalization_for_records(
            train_records, dataset_dir, config
        )
        cache: dict[str, torch.Tensor] = {}
        train_dataset = _make_dataset(
            train_records, dataset_dir, config, image_mean, image_std, normalizer, cache
        )
        validation_dataset = _make_dataset(
            validation_records, dataset_dir, config, image_mean, image_std, normalizer, cache
        )
        train_loader = make_loader(
            train_dataset, config["batch_size"], True, config["seed"] + model_index
        )
        validation_loader = make_loader(
            validation_dataset, config["batch_size"], False, config["seed"] + model_index
        )
        tag = f"train_{scene_count}"
        model_dir = model_root / tag
        run_output = output_dir / tag
        training_result = train_one_model(
            "full",
            config,
            train_loader,
            validation_loader,
            device,
            model_dir,
            run_output,
            image_mean,
            image_std,
        )
        model = build_model(config, "full", image_mean, image_std).to(device)
        model.load_state_dict(
            torch.load(training_result.weight_path, map_location=device, weights_only=True)
        )
        predicted, targets, sample_ids = predict(model, validation_loader, normalizer, device)
        validation_by_id = {record["sample_id"]: record for record in validation_records}
        metrics = extended_metrics(predicted, targets, sample_ids, validation_by_id)
        normalization = {
            "target": normalizer.to_dict(),
            "image_mean": image_mean,
            "image_std": image_std,
            "image_size_wh": config["image_size_wh"],
            "computed_from": f"phase7 nested train subset with {scene_count} scenes only",
        }
        (model_dir / "normalization.json").write_text(
            json.dumps(normalization, indent=2), encoding="utf-8"
        )
        results[tag] = {
            "train_scene_count": scene_count,
            "train_sample_count": len(train_records),
            "validation_scene_count": len({r["scene_id"] for r in validation_records}),
            "validation_sample_count": len(validation_records),
            "train_scene_subset_sha256": _subset_digest(subsets[scene_count]),
            "training": asdict(training_result),
            "validation_metrics": metrics,
            "normalization": normalization,
        }
        del model, train_dataset, validation_dataset, train_loader, validation_loader, cache
        torch.cuda.empty_cache()
    summary = {
        "stage": "train_and_validation_only",
        "test_data_accessed": False,
        "parameter_count": 49366,
        "architecture_changed": False,
        "training_method_changed": False,
        "seed": config["seed"],
        "subset_seed": config["subset_seed"],
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "train_pool_scene_count": len(train_scene_ids),
        "results": results,
        "config": config,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _load_model_for_eval(
    config: dict[str, Any],
    normalization: dict[str, Any],
    state_dict_path: Path,
    device: torch.device,
) -> tuple[GoalConditionedGraspModel, TargetNormalizer]:
    model = GoalConditionedGraspModel(
        mode="full",
        image_feature_dim=config["image_feature_dim"],
        color_embedding_dim=config["color_embedding_dim"],
        destination_embedding_dim=config["destination_embedding_dim"],
        dropout=config["dropout"],
        image_mean=normalization["image_mean"],
        image_std=normalization["image_std"],
    ).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return model, TargetNormalizer(**normalization["target"])


def run_evaluation(config_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7 evaluation requires the CUDA training environment")
    config = _load_config(config_path)
    set_reproducible(config["seed"])
    device = torch.device("cuda")
    dataset_dir = PROJECT_ROOT / config["dataset_dir"]
    output_dir = PROJECT_ROOT / config["output_dir"]
    model_root = PROJECT_ROOT / config["model_dir"]
    training_summary = _load_config(output_dir / "training_summary.json")
    records = load_jsonl(dataset_dir / "labels.jsonl")
    test_records = [record for record in records if record["split"] == "test"]
    test_by_id = {record["sample_id"]: record for record in test_records}
    model_specs: list[tuple[str, dict[str, Any], Path]] = []
    phase4_config = _load_config(PROJECT_ROOT / config["phase4_config"])
    old_normalization = _load_config(PROJECT_ROOT / config["phase4_normalization"])
    model_specs.append(
        (
            "phase4_old_80",
            old_normalization,
            PROJECT_ROOT / config["phase4_model_state_dict"],
        )
    )
    for scene_count in config["train_scene_counts"]:
        tag = f"train_{scene_count}"
        model_specs.append(
            (
                tag,
                _load_config(model_root / tag / "normalization.json"),
                model_root / tag / "full_best_state_dict.pt",
            )
        )
    evaluation: dict[str, Any] = {}
    for tag, normalization, state_path in model_specs:
        architecture_config = phase4_config if tag == "phase4_old_80" else config
        model, normalizer = _load_model_for_eval(
            architecture_config, normalization, state_path, device
        )
        cache: dict[str, torch.Tensor] = {}
        test_dataset = Phase4Dataset(
            test_records,
            dataset_dir,
            tuple(normalization["image_size_wh"]),
            normalization["image_mean"],
            normalization["image_std"],
            normalizer,
            cache,
        )
        test_loader = make_loader(test_dataset, config["batch_size"], False, config["seed"])
        predicted, targets, sample_ids = predict(model, test_loader, normalizer, device)
        metrics = extended_metrics(predicted, targets, sample_ids, test_by_id)
        metrics["parameter_count"] = count_parameters(model)
        evaluation[tag] = metrics
        write_predictions(
            output_dir / "predictions" / f"{tag}_test.jsonl",
            predicted,
            targets,
            sample_ids,
            test_by_id,
        )
        del model, test_dataset, test_loader, cache
        torch.cuda.empty_cache()
    summary = {
        "stage": "fixed_models_common_unused_test_evaluation",
        "test_used_for_model_selection": False,
        "test_scene_count": len({record["scene_id"] for record in test_records}),
        "test_sample_count": len(test_records),
        "training_summary_path": str((output_dir / "training_summary.json").resolve()),
        "training_results": training_summary["results"],
        "evaluation": evaluation,
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_comparison(summary, output_dir)
    return summary


def _write_comparison(summary: dict[str, Any], output_dir: Path) -> None:
    fields = [
        "model",
        "mean_mm",
        "median_mm",
        "p90_mm",
        "within_10",
        "within_15",
        "within_20",
        "x_mean_abs_mm",
        "y_mean_abs_mm",
    ]
    with (output_dir / "offline_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for tag, metrics in summary["evaluation"].items():
            writer.writerow(
                {
                    "model": tag,
                    "mean_mm": metrics["mean_error_mm"],
                    "median_mm": metrics["median_error_mm"],
                    "p90_mm": metrics["p90_error_mm"],
                    "within_10": metrics["within_10mm_fraction"],
                    "within_15": metrics["within_15mm_fraction"],
                    "within_20": metrics["within_20mm_fraction"],
                    "x_mean_abs_mm": metrics["axis_error_mm"]["x_orthogonal"][
                        "mean_absolute"
                    ],
                    "y_mean_abs_mm": metrics["axis_error_mm"]["y_closing"][
                        "mean_absolute"
                    ],
                }
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [f"train_{count}" for count in (80, 500, 2000)]
    sizes = [80, 500, 2000]
    means = [summary["evaluation"][tag]["mean_error_mm"] for tag in tags]
    medians = [summary["evaluation"][tag]["median_error_mm"] for tag in tags]
    p90s = [summary["evaluation"][tag]["p90_error_mm"] for tag in tags]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(sizes, means, marker="o", label="mean")
    axis.plot(sizes, medians, marker="o", label="median")
    axis.plot(sizes, p90s, marker="o", label="P90")
    axis.axhline(10.0, color="#d7301f", linestyle="--", label="10 mm goal")
    axis.set_xscale("log")
    axis.set_xticks(sizes, labels=[str(value) for value in sizes])
    axis.set_xlabel("training scenes")
    axis.set_ylabel("XY error (mm)")
    axis.set_title("Phase 7 data scale vs common-test position error")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(visual_dir / "data_scale_vs_error.png", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "train":
        summary = run_training(args.config.resolve())
        print(f"Phase 7 trained {len(summary['results'])} data-scale models")
    else:
        summary = run_evaluation(args.config.resolve())
        best = min(
            summary["evaluation"].items(), key=lambda item: item[1]["mean_error_mm"]
        )
        print(f"Phase 7 best common-test model: {best[0]} mean={best[1]['mean_error_mm']:.3f}mm")


if __name__ == "__main__":
    main()
