"""Diagnose phase-5 grasp failures with truth-only comparison and offset probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .panda_ik import to_numpy
from .phase3_dataset import sample_non_overlapping_positions
from .phase5_online import GraspPredictor, make_control_plan
from .runtime import PROJECT_ROOT, configure_genesis_cache

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase6_grasp_diagnostics.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase6"


@dataclass
class ProbeSpec:
    probe_id: str
    source_trial: int
    command_source: str
    axis: str | None = None
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0


@dataclass
class ProbeResult:
    probe_id: str
    source_trial: int
    source_seed: int
    target_class: str
    destination_id: str
    command_source: str
    axis: str | None
    requested_offset_x_mm: float
    requested_offset_y_mm: float
    actual_offset_x_mm: float
    actual_offset_y_mm: float
    actual_parallel_offset_mm: float
    actual_orthogonal_offset_mm: float
    commanded_xy_m: list[float]
    truth_xy_m: list[float]
    phase5_predicted_xy_m: list[float]
    replay_predicted_xy_m: list[float] | None
    replay_prediction_delta_mm: float | None
    phase5_initial_position_delta_max_mm: float
    success: bool
    lifted_object_class: str | None
    target_lift_delta_m: float
    finger_positions_after_close_m: list[float]
    finger_positions_after_lift_m: list[float]
    gripper_width_after_close_m: float
    gripper_width_after_lift_m: float
    hand_position_after_close_m: list[float]
    hand_position_after_lift_m: list[float]
    target_position_after_close_m: list[float]
    target_position_after_lift_m: list[float]
    initial_object_positions_m: dict[str, list[float]]
    elapsed_seconds: float
    lift_image_path: str


def build_probe_specs(
    phase5_failures: list[dict[str, Any]],
    sweep_source_trial: int,
    sweep_offsets_mm: list[float],
    mode: str,
) -> list[ProbeSpec]:
    """Build a bounded diagnostic plan; each offset point is one physical probe."""
    specs: list[ProbeSpec] = []
    for result in phase5_failures:
        trial = int(result["trial"])
        specs.extend(
            [
                ProbeSpec(f"trial_{trial:02d}_ai_replay", trial, "ai_replay"),
                ProbeSpec(f"trial_{trial:02d}_truth", trial, "diagnostic_truth"),
            ]
        )
    if mode == "pilot":
        return specs

    for result in phase5_failures:
        trial = int(result["trial"])
        predicted = np.asarray(result["predicted_xy_m"], dtype=np.float64)
        truth = np.asarray(result["target_truth_xy_m"], dtype=np.float64)
        delta_mm = (predicted - truth) * 1000.0
        specs.extend(
            [
                ProbeSpec(
                    f"trial_{trial:02d}_ai_x_component",
                    trial,
                    "phase5_error_component",
                    axis="x",
                    offset_x_mm=float(delta_mm[0]),
                ),
                ProbeSpec(
                    f"trial_{trial:02d}_ai_y_component",
                    trial,
                    "phase5_error_component",
                    axis="y",
                    offset_y_mm=float(delta_mm[1]),
                ),
            ]
        )
    for axis in ("x", "y"):
        for offset_mm in sweep_offsets_mm:
            specs.append(
                ProbeSpec(
                    probe_id=f"trial_{sweep_source_trial:02d}_sweep_{axis}_{offset_mm:+g}mm",
                    source_trial=sweep_source_trial,
                    command_source="diagnostic_offset_sweep",
                    axis=axis,
                    offset_x_mm=float(offset_mm) if axis == "x" else 0.0,
                    offset_y_mm=float(offset_mm) if axis == "y" else 0.0,
                )
            )
    return specs


def contiguous_success_interval(points: list[tuple[float, bool]]) -> list[float] | None:
    """Return the sampled contiguous success interval containing zero, not a hard limit."""
    ordered = sorted(points)
    zero_indices = [index for index, (offset, _) in enumerate(ordered) if offset == 0.0]
    if not zero_indices or not ordered[zero_indices[0]][1]:
        return None
    low = high = zero_indices[0]
    while low > 0 and ordered[low - 1][1]:
        low -= 1
    while high + 1 < len(ordered) and ordered[high + 1][1]:
        high += 1
    return [ordered[low][0], ordered[high][0]]


def classify_cause(
    truth_success: bool, replay_success: bool, x_component_success: bool, y_component_success: bool
) -> str:
    if not truth_success:
        return "truth grasp also failed: reachability/contact/reset condition"
    if replay_success:
        return "phase-5 failure not reproduced"
    if x_component_success and not y_component_success:
        return "closing-axis (world Y) offset dominated"
    if y_component_success and not x_component_success:
        return "orthogonal-axis (world X) offset dominated"
    if x_component_success and y_component_success:
        return "combined two-axis offset exceeded contact tolerance"
    return "both axis components independently exceeded contact tolerance"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def run_diagnostics(config_path: Path, output_dir: Path, mode: str) -> dict[str, Any]:
    configure_genesis_cache("cpu")
    import genesis as gs

    started_all = time.perf_counter()
    config = _load_json(config_path)
    phase5_payload = _load_json(PROJECT_ROOT / config["phase5_results"])
    phase2 = _load_json(PROJECT_ROOT / config["phase2_config"])
    phase3 = _load_json(PROJECT_ROOT / config["phase3_config"])
    failures_by_trial = {
        int(result["trial"]): result
        for result in phase5_payload["results"]
        if int(result["trial"]) in config["failure_trials"]
    }
    if set(failures_by_trial) != set(config["failure_trials"]):
        raise ValueError("Configured failure trials do not match phase-5 results")
    failures = [failures_by_trial[trial] for trial in config["failure_trials"]]
    specs = build_probe_specs(
        failures,
        int(config["sweep_source_trial"]),
        [float(value) for value in config["sweep_offsets_mm"]],
        mode,
    )
    predictor = GraspPredictor.from_artifacts(
        PROJECT_ROOT / config["phase4_config"],
        PROJECT_ROOT / config["normalization"],
        PROJECT_ROOT / config["model_state_dict"],
    )

    scene_config = phase3["scene"]
    sim_config = phase3["simulation"]
    robot_config = phase2["robot"]
    motion_config = phase2["motion"]
    grasp_config = phase3["grasp"]
    output_dir.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.cpu, logging_level="warning", seed=int(failures[0]["seed"]))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_config["dt_s"], substeps=sim_config["substeps"]),
        rigid_options=gs.options.RigidOptions(
            iterations=phase2["simulation"]["constraint_solver_iterations"],
            noslip_iterations=phase2["simulation"]["noslip_iterations"],
            box_box_detection=True,
            enable_multi_contact=True,
        ),
        show_viewer=False,
    )
    rigid_material = gs.materials.Rigid(friction=scene_config["rigid_friction"])
    scene.add_entity(
        morph=gs.morphs.Box(
            size=tuple(scene_config["table_size_m"]),
            pos=tuple(scene_config["table_position_m"]),
            fixed=True,
        ),
        material=rigid_material,
        surface=gs.surfaces.Default(color=(0.55, 0.42, 0.28, 1.0)),
    )
    cube_z = scene_config["table_position_m"][2] + (
        scene_config["table_size_m"][2] + scene_config["cube_size_m"][2]
    ) / 2
    cubes: dict[str, object] = {}
    for object_config in phase3["objects"]:
        cubes[object_config["class"]] = scene.add_entity(
            morph=gs.morphs.Box(
                size=tuple(scene_config["cube_size_m"]), pos=(0.55, 0.0, cube_z)
            ),
            material=gs.materials.Rigid(
                friction=scene_config["rigid_friction"],
                rho=scene_config["cube_density_kg_m3"],
            ),
            surface=gs.surfaces.Default(color=tuple(object_config["rgba"])),
        )
    panda = scene.add_entity(
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=rigid_material,
    )
    camera_config = phase3["camera"]
    camera = scene.add_camera(
        res=tuple(camera_config["resolution"]),
        pos=tuple(camera_config["position_m"]),
        lookat=tuple(camera_config["lookat_m"]),
        up=tuple(camera_config["up"]),
        fov=camera_config["fov_deg"],
        GUI=False,
    )
    scene.build()

    arm_dofs = np.arange(7, dtype=np.int32)
    finger_dofs = np.arange(7, 9, dtype=np.int32)
    initial_qpos = np.asarray(robot_config["initial_qpos_rad_m"], dtype=np.float64)
    panda.set_dofs_kp(np.asarray(robot_config["kp"], dtype=np.float64))
    panda.set_dofs_kv(np.asarray(robot_config["kv"], dtype=np.float64))
    panda.set_dofs_force_range(
        np.asarray(robot_config["force_lower_n"], dtype=np.float64),
        np.asarray(robot_config["force_upper_n"], dtype=np.float64),
    )
    end_effector = panda.get_link("hand")
    target_quaternion = np.asarray(robot_config["downward_quaternion_wxyz"])
    close_force = np.full(2, robot_config["gripper_close_force_n"], dtype=np.float64)
    open_width = np.full(2, robot_config["gripper_open_m"], dtype=np.float64)

    def hold(steps: int, closed: bool) -> None:
        for _ in range(steps):
            if closed:
                panda.control_dofs_force(close_force, dofs_idx_local=finger_dofs)
            else:
                panda.control_dofs_position(open_width, dofs_idx_local=finger_dofs)
            scene.step()

    def move(target: list[float], steps: int, closed: bool) -> None:
        qpos = to_numpy(
            panda.inverse_kinematics(
                link=end_effector,
                pos=np.asarray(target, dtype=np.float64),
                quat=target_quaternion,
                dofs_idx_local=arm_dofs,
            )
        )[:7]
        start_qpos = to_numpy(panda.get_dofs_position(arm_dofs))
        for step in range(1, steps + 1):
            alpha = step / steps
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            panda.control_dofs_position(
                start_qpos + smooth_alpha * (qpos - start_qpos), dofs_idx_local=arm_dofs
            )
            if closed:
                panda.control_dofs_force(close_force, dofs_idx_local=finger_dofs)
            else:
                panda.control_dofs_position(open_width, dofs_idx_local=finger_dofs)
            scene.step()

    results: list[ProbeResult] = []
    for index, spec in enumerate(specs, start=1):
        probe_started = time.perf_counter()
        source = failures_by_trial[spec.source_trial]
        seed = int(source["seed"])
        rng = np.random.default_rng(seed)
        random.seed(seed)
        positions = sample_non_overlapping_positions(
            rng,
            len(cubes),
            scene_config["placement_x_range_m"],
            scene_config["placement_y_range_m"],
            cube_z,
            scene_config["minimum_center_separation_m"],
            scene_config["placement_attempts"],
        )
        panda.set_qpos(initial_qpos, zero_velocity=True)
        panda.control_dofs_position(initial_qpos)
        for (_, cube), position in zip(cubes.items(), positions):
            yaw = rng.uniform(-math.pi, math.pi)
            cube.set_pos(position, zero_velocity=True)  # type: ignore[union-attr]
            cube.set_quat(  # type: ignore[union-attr]
                (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), zero_velocity=True
            )
        hold(sim_config["settle_steps"], closed=False)
        initial_positions = {
            name: to_numpy(cube.get_pos()).tolist()  # type: ignore[union-attr]
            for name, cube in cubes.items()
        }
        phase5_positions = source["initial_object_positions_m"]
        initial_delta_max_mm = max(
            float(
                np.linalg.norm(
                    np.asarray(initial_positions[name]) - np.asarray(phase5_positions[name])
                )
                * 1000.0
            )
            for name in cubes
        )
        target_class = source["target_class"]
        truth_xyz = np.asarray(initial_positions[target_class], dtype=np.float64)
        replay_prediction: list[float] | None = None
        replay_delta_mm: float | None = None
        if spec.command_source == "ai_replay":
            rgb, _, _, _ = camera.render(rgb=True, force_render=True)
            replay_prediction = predictor.predict(
                np.asarray(rgb, dtype=np.uint8), target_class, source["destination_id"]
            ).predicted_xy_m
            commanded_xy = np.asarray(replay_prediction, dtype=np.float64)
            replay_delta_mm = float(
                np.linalg.norm(
                    commanded_xy - np.asarray(source["predicted_xy_m"], dtype=np.float64)
                )
                * 1000.0
            )
        else:
            commanded_xy = truth_xyz[:2] + np.asarray(
                [spec.offset_x_mm, spec.offset_y_mm], dtype=np.float64
            ) / 1000.0
        actual_offset_mm = (commanded_xy - truth_xyz[:2]) * 1000.0
        plan = make_control_plan(
            commanded_xy.tolist(),
            [0.55, 0.20],
            cube_z,
            grasp_config,
            motion_config,
        )
        move(plan.pregrasp_hand_position_m, motion_config["move_steps"], False)
        move(plan.grasp_hand_position_m, motion_config["descend_steps"], False)
        hold(motion_config["close_steps"], True)
        fingers_after_close = to_numpy(panda.get_dofs_position(finger_dofs))
        hand_after_close = to_numpy(end_effector.get_pos())
        target_after_close = to_numpy(cubes[target_class].get_pos())  # type: ignore[union-attr]
        move(plan.lift_hand_position_m, motion_config["lift_steps"], True)
        fingers_after_lift = to_numpy(panda.get_dofs_position(finger_dofs))
        hand_after_lift = to_numpy(end_effector.get_pos())
        target_after_lift = to_numpy(cubes[target_class].get_pos())  # type: ignore[union-attr]
        lifted_positions = {
            name: to_numpy(cube.get_pos())  # type: ignore[union-attr]
            for name, cube in cubes.items()
        }
        candidates = [
            name
            for name, position in lifted_positions.items()
            if position[2]
            >= np.asarray(initial_positions[name])[2] + config["lift_height_threshold_m"]
        ]
        lifted_object = candidates[0] if len(candidates) == 1 else None
        success = lifted_object == target_class
        lift_path = output_dir / "images" / f"{spec.probe_id}_lift.png"
        rgb, _, _, _ = camera.render(rgb=True, force_render=True)
        _save_rgb(np.asarray(rgb, dtype=np.uint8), lift_path)
        result = ProbeResult(
            probe_id=spec.probe_id,
            source_trial=spec.source_trial,
            source_seed=seed,
            target_class=target_class,
            destination_id=source["destination_id"],
            command_source=spec.command_source,
            axis=spec.axis,
            requested_offset_x_mm=spec.offset_x_mm,
            requested_offset_y_mm=spec.offset_y_mm,
            actual_offset_x_mm=float(actual_offset_mm[0]),
            actual_offset_y_mm=float(actual_offset_mm[1]),
            actual_parallel_offset_mm=float(actual_offset_mm[1]),
            actual_orthogonal_offset_mm=float(actual_offset_mm[0]),
            commanded_xy_m=commanded_xy.tolist(),
            truth_xy_m=truth_xyz[:2].tolist(),
            phase5_predicted_xy_m=source["predicted_xy_m"],
            replay_predicted_xy_m=replay_prediction,
            replay_prediction_delta_mm=replay_delta_mm,
            phase5_initial_position_delta_max_mm=initial_delta_max_mm,
            success=success,
            lifted_object_class=lifted_object,
            target_lift_delta_m=float(target_after_lift[2] - truth_xyz[2]),
            finger_positions_after_close_m=fingers_after_close.tolist(),
            finger_positions_after_lift_m=fingers_after_lift.tolist(),
            gripper_width_after_close_m=float(np.sum(fingers_after_close)),
            gripper_width_after_lift_m=float(np.sum(fingers_after_lift)),
            hand_position_after_close_m=hand_after_close.tolist(),
            hand_position_after_lift_m=hand_after_lift.tolist(),
            target_position_after_close_m=target_after_close.tolist(),
            target_position_after_lift_m=target_after_lift.tolist(),
            initial_object_positions_m=initial_positions,
            elapsed_seconds=time.perf_counter() - probe_started,
            lift_image_path=lift_path.relative_to(output_dir).as_posix(),
        )
        results.append(result)
        print(
            f"probe {index:02d}/{len(specs)} {spec.probe_id}: "
            f"{'PASS' if success else 'FAIL'} offset=({actual_offset_mm[0]:.1f},"
            f" {actual_offset_mm[1]:.1f})mm width={result.gripper_width_after_lift_m:.4f}m",
            flush=True,
        )

    payload = _summarize(config, phase5_payload, results, mode, time.perf_counter() - started_all)
    _write_results(payload, output_dir)
    return payload


def _summarize(
    config: dict[str, Any],
    phase5_payload: dict[str, Any],
    results: list[ProbeResult],
    mode: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    by_id = {result.probe_id: result for result in results}
    comparisons: list[dict[str, Any]] = []
    for trial in config["failure_trials"]:
        replay = by_id[f"trial_{trial:02d}_ai_replay"]
        truth = by_id[f"trial_{trial:02d}_truth"]
        comparison: dict[str, Any] = {
            "source_trial": trial,
            "phase5_failed": True,
            "ai_replay_success": replay.success,
            "failure_reproduced": not replay.success,
            "truth_grasp_success": truth.success,
            "ai_replay_offset_xy_mm": [
                replay.actual_offset_x_mm,
                replay.actual_offset_y_mm,
            ],
            "replay_prediction_delta_mm": replay.replay_prediction_delta_mm,
        }
        if mode == "all":
            x_success = by_id[f"trial_{trial:02d}_ai_x_component"].success
            y_success = by_id[f"trial_{trial:02d}_ai_y_component"].success
            comparison.update(
                {
                    "x_component_success": x_success,
                    "y_component_success": y_success,
                    "cause_classification": classify_cause(
                        truth.success, replay.success, x_success, y_success
                    ),
                }
            )
        comparisons.append(comparison)

    axis_summary: dict[str, Any] = {}
    if mode == "all":
        for axis in ("x", "y"):
            points = [
                (
                    result.requested_offset_x_mm
                    if axis == "x"
                    else result.requested_offset_y_mm,
                    result.success,
                )
                for result in results
                if result.command_source == "diagnostic_offset_sweep" and result.axis == axis
            ]
            axis_summary[axis] = {
                "sample_count": len(points),
                "points": [
                    {"offset_mm": offset, "success": success} for offset, success in sorted(points)
                ],
                "observed_contiguous_success_interval_mm": contiguous_success_interval(points),
                "interpretation": "single probe per point; observed interval is not a hard limit",
            }
    return {
        "phase": 6,
        "description": "diagnostic grasp-offset tolerance evaluation",
        "mode": mode,
        "probe_count": len(results),
        "success_count": sum(result.success for result in results),
        "elapsed_seconds": elapsed_seconds,
        "mean_probe_seconds": float(np.mean([result.elapsed_seconds for result in results])),
        "finger_closing_axis_world": config["finger_closing_axis_world"],
        "orthogonal_axis_world": config["orthogonal_axis_world"],
        "truth_use": "diagnostic comparison and deliberate offset probes only",
        "phase5_results_unchanged": True,
        "phase5_result_summary": {
            "success_count": phase5_payload["success_count"],
            "trial_count": phase5_payload["trial_count"],
        },
        "failure_comparisons": comparisons,
        "axis_sweep": axis_summary,
        "config": config,
        "results": [asdict(result) for result in results],
    }


def _write_results(payload: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "phase6_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = payload["results"]
    csv_fields = [
        "probe_id",
        "source_trial",
        "source_seed",
        "target_class",
        "command_source",
        "axis",
        "actual_offset_x_mm",
        "actual_offset_y_mm",
        "actual_parallel_offset_mm",
        "actual_orthogonal_offset_mm",
        "success",
        "lifted_object_class",
        "target_lift_delta_m",
        "gripper_width_after_close_m",
        "gripper_width_after_lift_m",
        "elapsed_seconds",
    ]
    with (output_dir / "phase6_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in csv_fields} for row in rows)
    if payload["axis_sweep"]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for axis_name, plot_axis in zip(("x", "y"), axes):
            points = payload["axis_sweep"][axis_name]["points"]
            colors = ["#2ca25f" if point["success"] else "#de2d26" for point in points]
            plot_axis.scatter(
                [point["offset_mm"] for point in points],
                [1 if point["success"] else 0 for point in points],
                c=colors,
                s=70,
            )
            plot_axis.set_title(
                f"world {axis_name.upper()} "
                f"({'closing' if axis_name == 'y' else 'orthogonal'} axis)"
            )
            plot_axis.set_xlabel("command offset (mm)")
            plot_axis.set_yticks([0, 1], labels=["FAIL", "PASS"])
            plot_axis.grid(alpha=0.3)
        figure.suptitle("Phase 6 observed grasp outcomes (one probe per point)")
        figure.tight_layout()
        plot_path = output_dir / "visualizations" / "offset_success.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("pilot", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_diagnostics(args.config.resolve(), args.output_dir.resolve(), args.mode)
    print(
        f"Phase 6 {payload['mode']}: {payload['success_count']}/{payload['probe_count']} "
        f"grasp probes succeeded",
        flush=True,
    )


if __name__ == "__main__":
    main()
