"""Generate goal-conditioned RGB supervision without executing pick and place."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .runtime import PROJECT_ROOT, configure_genesis_cache

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase3_dataset.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "phase3"


@dataclass(frozen=True)
class Visibility:
    pixel_count: int
    bbox_xyxy: list[int] | None
    centroid_xy: list[float] | None


def to_array(value: object) -> np.ndarray:
    """Convert a Genesis/Torch value to NumPy without changing its dtype."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()  # type: ignore[union-attr]
    return np.asarray(value)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"simulation", "scene", "objects", "camera", "grasp", "destinations", "dataset"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if len(config["objects"]) != 3:
        raise ValueError("Phase 3 requires exactly three colored cubes")
    return config


def sample_non_overlapping_positions(
    rng: np.random.Generator,
    count: int,
    x_range: list[float],
    y_range: list[float],
    z: float,
    minimum_separation: float,
    max_attempts: int,
) -> list[list[float]]:
    """Sample reachable planar positions with a minimum pairwise separation."""
    positions: list[np.ndarray] = []
    for _ in range(max_attempts):
        candidate = np.array(
            [rng.uniform(*x_range), rng.uniform(*y_range), z], dtype=np.float64
        )
        if all(np.linalg.norm(candidate[:2] - position[:2]) >= minimum_separation for position in positions):
            positions.append(candidate)
            if len(positions) == count:
                return [position.tolist() for position in positions]
    raise RuntimeError(f"Could not sample {count} non-overlapping positions in {max_attempts} attempts")


def assign_scene_splits(
    scene_ids: list[str], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign entire scenes to deterministic train/validation/test splits."""
    if not scene_ids:
        return {}
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0")
    shuffled = list(scene_ids)
    np.random.default_rng(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = round(total * ratios["train"])
    validation_count = round(total * ratios["validation"])
    if total >= 3:
        train_count = min(max(train_count, 1), total - 2)
        validation_count = min(max(validation_count, 1), total - train_count - 1)
    test_count = total - train_count - validation_count
    counts = {"train": train_count, "validation": validation_count, "test": test_count}
    assignments: dict[str, str] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        for scene_id in shuffled[cursor : cursor + counts[split]]:
            assignments[scene_id] = split
        cursor += counts[split]
    return assignments


def visibility_from_mask(mask: np.ndarray, entity_index: int) -> Visibility:
    """Return entity visibility statistics from an entity-level segmentation mask."""
    ys, xs = np.nonzero(mask == entity_index)
    if len(xs) == 0:
        return Visibility(pixel_count=0, bbox_xyxy=None, centroid_xy=None)
    return Visibility(
        pixel_count=len(xs),
        bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        centroid_xy=[float(xs.mean()), float(ys.mean())],
    )


def validate_split_isolation(samples: list[dict[str, Any]]) -> list[str]:
    """Find images or scenes that leak across dataset splits."""
    errors: list[str] = []
    by_scene: dict[str, set[str]] = {}
    by_image: dict[str, set[str]] = {}
    for sample in samples:
        by_scene.setdefault(sample["scene_id"], set()).add(sample["split"])
        by_image.setdefault(sample["image_path"], set()).add(sample["split"])
    for scene_id, splits in by_scene.items():
        if len(splits) != 1:
            errors.append(f"scene {scene_id} appears in {sorted(splits)}")
    for image_path, splits in by_image.items():
        if len(splits) != 1:
            errors.append(f"image {image_path} appears in {sorted(splits)}")
    return errors


def validate_label_consistency(samples: list[dict[str, Any]], tolerance: float = 1e-9) -> list[str]:
    """Check that each grasp XY exactly identifies the instructed target object."""
    errors: list[str] = []
    for sample in samples:
        grasp = np.asarray(sample["grasp_position_xy_world_m"], dtype=np.float64)
        truth = np.asarray(sample["target_truth_position_m"][:2], dtype=np.float64)
        if float(np.linalg.norm(grasp - truth)) > tolerance:
            errors.append(sample["sample_id"])
    return errors


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_overlay(
    rgb: np.ndarray,
    scene_id: str,
    object_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    colors = {"red": "#ff3030", "blue": "#3080ff", "green": "#20d060"}
    for item in object_records:
        visibility = item["visibility"]
        if visibility["bbox_xyxy"] is None or visibility["centroid_xy"] is None:
            continue
        color = colors[item["color_name"]]
        draw.rectangle(visibility["bbox_xyxy"], outline=color, width=3)
        cx, cy = visibility["centroid_xy"]
        draw.line((cx - 8, cy, cx + 8, cy), fill="white", width=3)
        draw.line((cx, cy - 8, cx, cy + 8), fill="white", width=3)
        xy = item["truth_position_m"][:2]
        draw.text((cx + 10, cy - 18), f"{item['color_name']} grasp ({xy[0]:.3f},{xy[1]:.3f})", fill=color)
    draw.text((10, 10), f"{scene_id}: crosses mark top-down grasp centers", fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def generate_dataset(config: dict[str, Any], output_dir: Path, scene_count: int) -> dict[str, Any]:
    """Reuse one Genesis scene to generate stable multi-object RGB scenes and labels."""
    configure_genesis_cache("cpu")
    import genesis as gs

    started = time.perf_counter()
    sim_config = config["simulation"]
    scene_config = config["scene"]
    dataset_config = config["dataset"]
    camera_config = config["camera"]
    base_seed = int(dataset_config["base_seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    overlays_dir = output_dir / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=gs.cpu, logging_level="warning", seed=base_seed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_config["dt_s"], substeps=sim_config["substeps"]),
        rigid_options=gs.options.RigidOptions(box_box_detection=True, enable_multi_contact=True),
        vis_options=gs.options.VisOptions(segmentation_level="link"),
        show_viewer=False,
    )
    material = gs.materials.Rigid(friction=scene_config["rigid_friction"])
    scene.add_entity(
        morph=gs.morphs.Box(
            size=tuple(scene_config["table_size_m"]),
            pos=tuple(scene_config["table_position_m"]),
            fixed=True,
        ),
        material=material,
        surface=gs.surfaces.Default(color=(0.55, 0.42, 0.28, 1.0)),
    )
    cubes = []
    cube_z = scene_config["table_position_m"][2] + (
        scene_config["table_size_m"][2] + scene_config["cube_size_m"][2]
    ) / 2
    for object_config in config["objects"]:
        cubes.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    size=tuple(scene_config["cube_size_m"]),
                    pos=(0.55, 0.0, cube_z),
                ),
                material=gs.materials.Rigid(
                    friction=scene_config["rigid_friction"],
                    rho=scene_config["cube_density_kg_m3"],
                ),
                surface=gs.surfaces.Default(color=tuple(object_config["rgba"])),
            )
        )
    panda = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    camera = scene.add_camera(
        res=tuple(camera_config["resolution"]),
        pos=tuple(camera_config["position_m"]),
        lookat=tuple(camera_config["lookat_m"]),
        up=tuple(camera_config["up"]),
        fov=camera_config["fov_deg"],
        GUI=False,
    )
    scene.build()
    initial_qpos = np.asarray(config["robot"]["initial_qpos_rad_m"], dtype=np.float64)
    panda.set_dofs_kp(np.asarray(config["robot"]["kp"], dtype=np.float64))
    panda.set_dofs_kv(np.asarray(config["robot"]["kv"], dtype=np.float64))
    panda.set_qpos(initial_qpos, zero_velocity=True)
    panda.control_dofs_position(initial_qpos)

    scene_ids = [f"scene_{index:04d}" for index in range(scene_count)]
    split_seed = base_seed + int(dataset_config["split_seed_offset"])
    split_by_scene = assign_scene_splits(scene_ids, dataset_config["split_ratios"], split_seed)
    scene_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []

    for scene_index, scene_id in enumerate(scene_ids):
        accepted = False
        for retry in range(scene_config["scene_retries"]):
            seed = base_seed + scene_index * scene_config["scene_retries"] + retry
            rng = np.random.default_rng(seed)
            try:
                positions = sample_non_overlapping_positions(
                    rng=rng,
                    count=len(cubes),
                    x_range=scene_config["placement_x_range_m"],
                    y_range=scene_config["placement_y_range_m"],
                    z=cube_z,
                    minimum_separation=scene_config["minimum_center_separation_m"],
                    max_attempts=scene_config["placement_attempts"],
                )
                for cube, position in zip(cubes, positions):
                    yaw = rng.uniform(-math.pi, math.pi)
                    cube.set_pos(position, zero_velocity=True)
                    cube.set_quat(
                        (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                        zero_velocity=True,
                    )
                for _ in range(sim_config["settle_steps"]):
                    panda.control_dofs_position(initial_qpos)
                    scene.step()

                actual_positions = [to_array(cube.get_pos()).astype(float) for cube in cubes]
                speeds = [float(np.linalg.norm(to_array(cube.get_vel()))) for cube in cubes]
                rgb_value, _, segmentation_value, _ = camera.render(
                    rgb=True, segmentation=True, force_render=True
                )
                rgb = to_array(rgb_value).astype(np.uint8)
                segmentation = to_array(segmentation_value)
                object_records: list[dict[str, Any]] = []
                invalid_reasons: list[str] = []
                for object_config, cube, actual_position, speed in zip(
                    config["objects"], cubes, actual_positions, speeds
                ):
                    # Genesis 1.2.1 reserves mask value 0 for the background, so the
                    # rasterized link label is the zero-based solver link index plus one.
                    segmentation_index = int(cube.base_link_idx) + 1
                    visibility = visibility_from_mask(segmentation, segmentation_index)
                    reasons: list[str] = []
                    if visibility.pixel_count < dataset_config["minimum_visible_pixels"]:
                        reasons.append("insufficient_visible_pixels")
                    if abs(actual_position[2] - cube_z) > sim_config["support_height_tolerance_m"]:
                        reasons.append("unsupported_height")
                    if speed > sim_config["max_linear_speed_m_s"]:
                        reasons.append("not_stable")
                    invalid_reasons.extend(f"{object_config['class']}:{reason}" for reason in reasons)
                    object_records.append(
                        {
                            "class": object_config["class"],
                            "color_name": object_config["color_name"],
                            "segmentation_link_index": int(cube.base_link_idx),
                            "segmentation_mask_value": segmentation_index,
                            "truth_position_m": actual_position.tolist(),
                            "linear_speed_m_s": speed,
                            "visibility": asdict(visibility),
                            "label_valid": not reasons,
                            "invalid_reasons": reasons,
                        }
                    )
                if invalid_reasons:
                    failed_attempts.append(
                        {"scene_id": scene_id, "seed": seed, "reasons": invalid_reasons}
                    )
                    continue

                image_path = images_dir / f"{scene_id}.png"
                Image.fromarray(rgb).save(image_path)
                if scene_index < dataset_config["overlay_scene_count"]:
                    _save_overlay(
                        rgb,
                        scene_id,
                        object_records,
                        overlays_dir / f"{scene_id}_grasp_overlay.png",
                    )
                camera_record = {
                    "resolution": camera_config["resolution"],
                    "position_m": camera_config["position_m"],
                    "lookat_m": camera_config["lookat_m"],
                    "up": camera_config["up"],
                    "fov_deg": camera_config["fov_deg"],
                }
                split = split_by_scene[scene_id]
                scene_records.append(
                    {
                        "scene_id": scene_id,
                        "seed": seed,
                        "split": split,
                        "image_path": image_path.relative_to(output_dir).as_posix(),
                        "camera": camera_record,
                        "objects": object_records,
                        "label_valid": True,
                    }
                )
                for object_record in object_records:
                    truth_position = object_record["truth_position_m"]
                    hand_z = truth_position[2] + config["grasp"]["hand_to_cube_center_z_offset_m"]
                    for destination in config["destinations"]:
                        sample_id = (
                            f"{scene_id}_{object_record['color_name']}_{destination['id']}"
                        )
                        sample_records.append(
                            {
                                "sample_id": sample_id,
                                "scene_id": scene_id,
                                "seed": seed,
                                "split": split,
                                "image_path": image_path.relative_to(output_dir).as_posix(),
                                "instruction": (
                                    f"{object_record['color_name']} cube to "
                                    f"{destination['instruction_name']}"
                                ),
                                "target_class": object_record["class"],
                                "grasp_position_xy_world_m": truth_position[:2],
                                "transport_target_id": destination["id"],
                                "transport_target_position_m": destination["world_position_m"],
                                "camera": camera_record,
                                "target_truth_position_m": truth_position,
                                "label_validity": {
                                    "valid": object_record["label_valid"],
                                    "reasons": object_record["invalid_reasons"],
                                    "visible_pixel_count": object_record["visibility"]["pixel_count"],
                                },
                                "physical_grasp_mapping": {
                                    "approach": config["grasp"]["approach"],
                                    "hand_target_position_m": [
                                        truth_position[0],
                                        truth_position[1],
                                        hand_z,
                                    ],
                                    "hand_quaternion_wxyz": config["grasp"][
                                        "hand_quaternion_wxyz"
                                    ],
                                    "hand_to_cube_center_z_offset_m": config["grasp"][
                                        "hand_to_cube_center_z_offset_m"
                                    ],
                                    "validated_by": config["grasp"]["source_phase"],
                                },
                            }
                        )
                accepted = True
                print(
                    f"{scene_id}: PASS seed={seed} samples={len(config['objects']) * len(config['destinations'])}",
                    flush=True,
                )
                break
            except Exception as exc:  # noqa: BLE001 - generation failures must be recorded.
                failed_attempts.append(
                    {"scene_id": scene_id, "seed": seed, "reasons": [f"{type(exc).__name__}: {exc}"]}
                )
        if not accepted:
            raise RuntimeError(f"Failed to generate valid {scene_id} after retries")

    label_errors = validate_label_consistency(sample_records)
    leakage_errors = validate_split_isolation(sample_records)
    if label_errors or leakage_errors:
        raise RuntimeError(
            f"Dataset validation failed: label_errors={label_errors}, leakage_errors={leakage_errors}"
        )

    _write_jsonl(output_dir / "scenes.jsonl", scene_records)
    _write_jsonl(output_dir / "labels.jsonl", sample_records)
    csv_fields = [
        "sample_id",
        "scene_id",
        "seed",
        "split",
        "image_path",
        "instruction",
        "target_class",
        "grasp_x_world_m",
        "grasp_y_world_m",
        "transport_target_id",
        "target_truth_x_m",
        "target_truth_y_m",
        "target_truth_z_m",
        "label_valid",
        "visible_pixel_count",
    ]
    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for sample in sample_records:
            truth = sample["target_truth_position_m"]
            grasp = sample["grasp_position_xy_world_m"]
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "scene_id": sample["scene_id"],
                    "seed": sample["seed"],
                    "split": sample["split"],
                    "image_path": sample["image_path"],
                    "instruction": sample["instruction"],
                    "target_class": sample["target_class"],
                    "grasp_x_world_m": grasp[0],
                    "grasp_y_world_m": grasp[1],
                    "transport_target_id": sample["transport_target_id"],
                    "target_truth_x_m": truth[0],
                    "target_truth_y_m": truth[1],
                    "target_truth_z_m": truth[2],
                    "label_valid": sample["label_validity"]["valid"],
                    "visible_pixel_count": sample["label_validity"]["visible_pixel_count"],
                }
            )
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_scene_ids = [scene_id for scene_id in scene_ids if split_by_scene[scene_id] == split]
        split_counts[split] = len(split_scene_ids)
        (output_dir / "splits").mkdir(parents=True, exist_ok=True)
        (output_dir / "splits" / f"{split}_scenes.txt").write_text(
            "\n".join(split_scene_ids) + "\n", encoding="utf-8"
        )
    (output_dir / "phase3_config_snapshot.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    visible_counts = [
        item["visibility"]["pixel_count"]
        for scene_record in scene_records
        for item in scene_record["objects"]
    ]
    manifest = {
        "phase": 3,
        "genesis_version": gs.__version__,
        "backend": "cpu",
        "scene_count": len(scene_records),
        "sample_count": len(sample_records),
        "samples_per_scene": len(config["objects"]) * len(config["destinations"]),
        "split_scene_counts": split_counts,
        "failed_attempt_count": len(failed_attempts),
        "failed_attempts": failed_attempts,
        "all_labels_valid": not label_errors,
        "split_leakage_count": len(leakage_errors),
        "minimum_visible_pixels_observed": min(visible_counts),
        "maximum_visible_pixels_observed": max(visible_counts),
        "generation_seconds": elapsed,
        "scenes_per_second": len(scene_records) / elapsed,
        "samples_per_second": len(sample_records) / elapsed,
        "disk_usage_bytes": 0,
        "grasp_label_definition": (
            "cube center world XY with phase-2 top-down hand pose at cube center z + 0.110 m"
        ),
        "rgb_input_only": True,
        "segmentation_usage": "visibility QA and overlays only; not stored as model input",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["disk_usage_bytes"] = _directory_size_bytes(output_dir)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    scene_count = args.scenes or int(config["dataset"]["full_scene_count"])
    manifest = generate_dataset(config, args.output_dir.resolve(), scene_count)
    print(
        f"Phase 3 dataset: {manifest['scene_count']} scenes, "
        f"{manifest['sample_count']} samples, {manifest['generation_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
