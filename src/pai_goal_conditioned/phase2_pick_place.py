"""Run fixed-coordinate Franka Panda pick-and-place trials in Genesis."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .panda_ik import to_numpy
from .runtime import PROJECT_ROOT, configure_genesis_cache

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase2_pick_place.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2"


class Stage(str, Enum):
    INITIAL = "initial_pose"
    MOVE_ABOVE = "move_above_object"
    DESCEND_TO_GRASP = "descend_to_grasp"
    CLOSE_GRIPPER = "close_gripper"
    LIFT = "lift"
    TRANSPORT = "transport"
    DESCEND_TO_PLACE = "descend_to_place"
    OPEN_GRIPPER = "open_gripper"
    RETREAT = "retreat"
    SETTLE = "settle"


@dataclass
class StageRecord:
    stage: str
    steps: int
    cube_position_m: list[float]
    elapsed_seconds: float


@dataclass
class TrialResult:
    trial: int
    seed: int
    success: bool
    grasp_success: bool
    lift_success: bool
    transport_success: bool
    placement_success: bool
    stable_after_release: bool
    final_cube_position_m: list[float] | None
    final_position_error_m: float | None
    final_linear_speed_m_s: float | None
    final_angular_speed_rad_s: float | None
    failed_stage: str | None
    failure_reason: str | None
    elapsed_seconds: float
    reset_conditions: dict[str, Any]
    stage_records: list[dict[str, Any]]


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the versioned phase-2 JSON configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"simulation", "scene", "robot", "motion", "evaluation"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


def horizontal_error(position: np.ndarray, target: np.ndarray) -> float:
    """Return planar XY error in metres."""
    return float(np.linalg.norm(position[:2] - target[:2]))


def supported_height_error(position: np.ndarray, target: np.ndarray) -> float:
    """Return absolute cube-centre height error from its supported target height."""
    return abs(float(position[2] - target[2]))


def classify_failure(
    grasp_success: bool,
    lift_success: bool,
    transport_success: bool,
    placement_success: bool,
    stable_after_release: bool,
) -> str | None:
    """Map evaluation flags to the first failed state-machine stage."""
    if not grasp_success:
        return Stage.CLOSE_GRIPPER.value
    if not lift_success:
        return Stage.LIFT.value
    if not transport_success:
        return Stage.TRANSPORT.value
    if not placement_success:
        return Stage.OPEN_GRIPPER.value
    if not stable_after_release:
        return Stage.SETTLE.value
    return None


def _save_image(camera: object, path: Path) -> None:
    from PIL import Image

    rgb, _, _, _ = camera.render(rgb=True)  # type: ignore[union-attr]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_numpy(rgb).astype(np.uint8)).save(path)


def run_single_trial(config: dict[str, Any], trial: int, seed: int, output_dir: Path) -> TrialResult:
    """Build one fresh Genesis scene and execute one physical pick-and-place trial."""
    configure_genesis_cache("cpu")
    import genesis as gs

    sim = config["simulation"]
    scene_config = config["scene"]
    robot_config = config["robot"]
    motion = config["motion"]
    evaluation = config["evaluation"]
    started = time.perf_counter()
    failed_stage: str | None = None
    failure_reason: str | None = None
    stage_records: list[StageRecord] = []
    grasp_success = lift_success = transport_success = False
    placement_success = stable_after_release = False
    final_position: np.ndarray | None = None
    final_error: float | None = None
    linear_speed: float | None = None
    angular_speed: float | None = None

    try:
        gs.init(backend=gs.cpu, logging_level="warning", seed=seed)
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=sim["dt_s"],
                substeps=sim["substeps"],
            ),
            rigid_options=gs.options.RigidOptions(
                iterations=sim["constraint_solver_iterations"],
                noslip_iterations=sim["noslip_iterations"],
                enable_multi_contact=True,
                box_box_detection=True,
            ),
            show_viewer=False,
            show_FPS=False,
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
        cube = scene.add_entity(
            morph=gs.morphs.Box(
                size=tuple(scene_config["cube_size_m"]),
                pos=tuple(scene_config["cube_initial_position_m"]),
            ),
            material=gs.materials.Rigid(
                friction=scene_config["rigid_friction"],
                rho=scene_config["cube_density_kg_m3"],
            ),
            surface=gs.surfaces.Default(color=(0.15, 0.45, 0.9, 1.0)),
        )
        panda = scene.add_entity(
            morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
            material=rigid_material,
        )
        camera = None
        if trial == 1:
            camera = scene.add_camera(
                res=(640, 480),
                pos=(1.35, -1.15, 0.95),
                lookat=(0.50, 0.02, 0.30),
                fov=42,
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
        panda.set_qpos(initial_qpos, zero_velocity=True)
        end_effector = panda.get_link("hand")
        target_quaternion = np.asarray(robot_config["downward_quaternion_wxyz"])
        close_force = np.full(2, robot_config["gripper_close_force_n"], dtype=np.float64)
        open_width = np.full(2, robot_config["gripper_open_m"], dtype=np.float64)

        def record(stage: Stage, steps: int, stage_started: float) -> None:
            stage_records.append(
                StageRecord(
                    stage=stage.value,
                    steps=steps,
                    cube_position_m=to_numpy(cube.get_pos()).tolist(),
                    elapsed_seconds=time.perf_counter() - stage_started,
                )
            )

        def hold(stage: Stage, steps: int, closed: bool) -> None:
            stage_started = time.perf_counter()
            for _ in range(steps):
                if closed:
                    panda.control_dofs_force(close_force, dofs_idx_local=finger_dofs)
                else:
                    panda.control_dofs_position(open_width, dofs_idx_local=finger_dofs)
                scene.step()
            record(stage, steps, stage_started)

        def move(stage: Stage, target_key: str, steps: int, closed: bool) -> None:
            stage_started = time.perf_counter()
            target_position = np.asarray(motion[target_key], dtype=np.float64)
            qpos = to_numpy(
                panda.inverse_kinematics(
                    link=end_effector,
                    pos=target_position,
                    quat=target_quaternion,
                    dofs_idx_local=arm_dofs,
                )
            )[:7]
            start_qpos = to_numpy(panda.get_dofs_position(arm_dofs))
            for step in range(1, steps + 1):
                alpha = step / steps
                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                arm_target = start_qpos + smooth_alpha * (qpos - start_qpos)
                panda.control_dofs_position(arm_target, dofs_idx_local=arm_dofs)
                if closed:
                    panda.control_dofs_force(close_force, dofs_idx_local=finger_dofs)
                else:
                    panda.control_dofs_position(open_width, dofs_idx_local=finger_dofs)
                scene.step()
            record(stage, steps, stage_started)

        hold(Stage.INITIAL, motion["initial_steps"], closed=False)
        move(Stage.MOVE_ABOVE, "pregrasp_hand_position_m", motion["move_steps"], closed=False)
        move(
            Stage.DESCEND_TO_GRASP,
            "grasp_hand_position_m",
            motion["descend_steps"],
            closed=False,
        )
        hold(Stage.CLOSE_GRIPPER, motion["close_steps"], closed=True)
        grasp_position = to_numpy(cube.get_pos())
        cube_start = np.asarray(scene_config["cube_initial_position_m"], dtype=np.float64)
        grasp_success = horizontal_error(grasp_position, cube_start) <= 0.035

        move(Stage.LIFT, "lift_hand_position_m", motion["lift_steps"], closed=True)
        lift_position = to_numpy(cube.get_pos())
        lift_success = bool(
            grasp_success
            and lift_position[2] >= cube_start[2] + evaluation["lift_height_threshold_m"]
        )
        if camera is not None:
            _save_image(camera, output_dir / "images" / f"trial_{trial:02d}_lift.png")

        move(
            Stage.TRANSPORT,
            "transport_hand_position_m",
            motion["transport_steps"],
            closed=True,
        )
        transport_position = to_numpy(cube.get_pos())
        cube_target = np.asarray(scene_config["cube_target_position_m"], dtype=np.float64)
        transport_success = bool(
            lift_success
            and horizontal_error(transport_position, cube_target)
            <= evaluation["transport_xy_tolerance_m"]
            and transport_position[2]
            >= cube_target[2] + evaluation["lift_height_threshold_m"]
        )

        move(
            Stage.DESCEND_TO_PLACE,
            "place_hand_position_m",
            motion["place_steps"],
            closed=True,
        )
        hold(Stage.OPEN_GRIPPER, motion["open_steps"], closed=False)
        move(Stage.RETREAT, "retreat_hand_position_m", motion["retreat_steps"], closed=False)
        hold(Stage.SETTLE, sim["settle_steps"], closed=False)

        final_position = to_numpy(cube.get_pos())
        final_error = horizontal_error(final_position, cube_target)
        linear_speed = float(np.linalg.norm(to_numpy(cube.get_vel())))
        angular_speed = float(np.linalg.norm(to_numpy(cube.get_ang())))
        placement_success = bool(
            final_error <= evaluation["placement_xy_tolerance_m"]
            and supported_height_error(final_position, cube_target)
            <= evaluation["support_height_tolerance_m"]
        )
        stable_after_release = bool(
            linear_speed <= evaluation["stable_linear_speed_m_s"]
            and angular_speed <= evaluation["stable_angular_speed_rad_s"]
        )
        failed_stage = classify_failure(
            grasp_success,
            lift_success,
            transport_success,
            placement_success,
            stable_after_release,
        )
        if failed_stage is not None:
            failure_reason = f"evaluation condition failed at {failed_stage}"
        if camera is not None:
            _save_image(camera, output_dir / "images" / f"trial_{trial:02d}_final.png")
    except Exception as exc:  # noqa: BLE001 - a failed physical trial must be persisted.
        failed_stage = failed_stage or "scene_or_control_exception"
        failure_reason = f"{type(exc).__name__}: {exc}"

    success = bool(
        grasp_success
        and lift_success
        and transport_success
        and placement_success
        and stable_after_release
        and failure_reason is None
    )
    return TrialResult(
        trial=trial,
        seed=seed,
        success=success,
        grasp_success=grasp_success,
        lift_success=lift_success,
        transport_success=transport_success,
        placement_success=placement_success,
        stable_after_release=stable_after_release,
        final_cube_position_m=None if final_position is None else final_position.tolist(),
        final_position_error_m=final_error,
        final_linear_speed_m_s=linear_speed,
        final_angular_speed_rad_s=angular_speed,
        failed_stage=failed_stage,
        failure_reason=failure_reason,
        elapsed_seconds=time.perf_counter() - started,
        reset_conditions={
            "cube_initial_position_m": scene_config["cube_initial_position_m"],
            "cube_target_position_m": scene_config["cube_target_position_m"],
            "robot_initial_qpos_rad_m": robot_config["initial_qpos_rad_m"],
            "dt_s": sim["dt_s"],
            "substeps": sim["substeps"],
            "rigid_friction": scene_config["rigid_friction"],
            "cube_density_kg_m3": scene_config["cube_density_kg_m3"],
        },
        stage_records=[asdict(item) for item in stage_records],
    )


def write_trial_result(result: TrialResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def aggregate_results(
    results: list[TrialResult], config: dict[str, Any], config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Persist the machine-readable ten-trial summary and per-trial CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = sum(result.success for result in results)
    payload = {
        "phase": 2,
        "description": "fixed-coordinate physical-contact pick and place",
        "genesis_version": "1.2.1",
        "backend": "cpu",
        "independent_initialization": "one Python process and fresh Genesis scene per trial",
        "config_path": str(config_path.resolve()),
        "success_count": success_count,
        "trial_count": len(results),
        "pass_criterion": "at least 8 full successes in 10 trials",
        "phase_pass": len(results) == 10 and success_count >= 8,
        "config": config,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "phase2_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_fields = [
        "trial",
        "seed",
        "success",
        "grasp_success",
        "lift_success",
        "transport_success",
        "placement_success",
        "stable_after_release",
        "final_cube_position_m",
        "final_position_error_m",
        "final_linear_speed_m_s",
        "final_angular_speed_rad_s",
        "failed_stage",
        "failure_reason",
        "elapsed_seconds",
    ]
    with (output_dir / "phase2_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({field: row[field] for field in csv_fields})
    shutil.copyfile(config_path, output_dir / "phase2_config_snapshot.json")
    return payload


def run_independent_trials(
    config: dict[str, Any], config_path: Path, output_dir: Path, trial_count: int
) -> dict[str, Any]:
    """Launch each trial in a new process so Genesis is independently initialized."""
    base_seed = int(config["evaluation"]["base_seed"])
    results: list[TrialResult] = []
    trial_dir = output_dir / "trials"
    for trial in range(1, trial_count + 1):
        result_path = trial_dir / f"trial_{trial:02d}.json"
        command = [
            sys.executable,
            "-m",
            "pai_goal_conditioned.phase2_pick_place",
            "--single-trial",
            "--trial-index",
            str(trial),
            "--seed",
            str(base_seed + trial - 1),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--result-path",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result_path.exists():
            result = TrialResult(**json.loads(result_path.read_text(encoding="utf-8")))
        else:
            result = TrialResult(
                trial=trial,
                seed=base_seed + trial - 1,
                success=False,
                grasp_success=False,
                lift_success=False,
                transport_success=False,
                placement_success=False,
                stable_after_release=False,
                final_cube_position_m=None,
                final_position_error_m=None,
                final_linear_speed_m_s=None,
                final_angular_speed_rad_s=None,
                failed_stage="process_failure",
                failure_reason=(completed.stderr or completed.stdout)[-2000:],
                elapsed_seconds=0.0,
                reset_conditions={},
                stage_records=[],
            )
        results.append(result)
        print(
            f"trial {trial:02d}: {'PASS' if result.success else 'FAIL'} "
            f"error={result.final_position_error_m!r} stage={result.failed_stage!r}",
            flush=True,
        )
    return aggregate_results(results, config, config_path, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--single-trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trial-index", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    if args.single_trial:
        if args.seed is None or args.result_path is None:
            raise ValueError("--single-trial requires --seed and --result-path")
        result = run_single_trial(config, args.trial_index, args.seed, output_dir)
        write_trial_result(result, args.result_path)
        print(json.dumps(asdict(result), ensure_ascii=False))
        return

    trial_count = args.trials or int(config["evaluation"]["trial_count"])
    payload = run_independent_trials(config, config_path, output_dir, trial_count)
    print(f"Phase 2: {payload['success_count']}/{payload['trial_count']} full successes")
    print(f"Result: {'PASS' if payload['phase_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
