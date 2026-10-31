"""Record one unchanged phase-5 controller trial for the final demonstration."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .panda_ik import to_numpy
from .phase3_dataset import sample_non_overlapping_positions
from .phase5_online import GraspPredictor, make_control_plan
from .runtime import PROJECT_ROOT, configure_genesis_cache


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def record_trial(
    config_path: Path,
    seed: int,
    target_class: str,
    destination_id: str,
    model_label: str,
    output_video: Path,
    output_json: Path,
    frame_stride: int = 6,
    fps: int = 20,
) -> dict[str, Any]:
    configure_genesis_cache("cpu")
    import genesis as gs

    config = _load(config_path)
    phase2 = _load(PROJECT_ROOT / config["phase2_config"])
    phase3 = _load(PROJECT_ROOT / config["phase3_config"])
    scene_config = phase3["scene"]
    sim_config = phase3["simulation"]
    robot_config = phase2["robot"]
    motion_config = phase2["motion"]
    grasp_config = phase3["grasp"]
    predictor = GraspPredictor.from_artifacts(
        PROJECT_ROOT / config["phase4_config"],
        PROJECT_ROOT / config["normalization"],
        PROJECT_ROOT / config["model_state_dict"],
    )
    gs.init(backend=gs.cpu, logging_level="warning", seed=seed)
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
    panda.set_qpos(initial_qpos, zero_velocity=True)
    panda.control_dofs_position(initial_qpos)
    end_effector = panda.get_link("hand")
    target_quaternion = np.asarray(robot_config["downward_quaternion_wxyz"])
    close_force = np.full(2, robot_config["gripper_close_force_n"], dtype=np.float64)
    open_width = np.full(2, robot_config["gripper_open_m"], dtype=np.float64)

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
    for (_, cube), position in zip(cubes.items(), positions):
        yaw = rng.uniform(-math.pi, math.pi)
        cube.set_pos(position, zero_velocity=True)  # type: ignore[union-attr]
        cube.set_quat(  # type: ignore[union-attr]
            (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), zero_velocity=True
        )

    def unrecorded_hold(steps: int) -> None:
        for _ in range(steps):
            panda.control_dofs_position(open_width, dofs_idx_local=finger_dofs)
            scene.step()

    unrecorded_hold(sim_config["settle_steps"])
    initial_positions = {
        name: to_numpy(cube.get_pos()).tolist()  # type: ignore[union-attr]
        for name, cube in cubes.items()
    }
    rgb, _, _, _ = camera.render(rgb=True, force_render=True)
    inference = predictor.predict(np.asarray(rgb, dtype=np.uint8), target_class, destination_id)
    truth_xy = np.asarray(initial_positions[target_class][:2])
    prediction_error_mm = float(
        np.linalg.norm(np.asarray(inference.predicted_xy_m) - truth_xy) * 1000.0
    )
    destination = next(
        item for item in phase3["destinations"] if item["id"] == destination_id
    )["world_position_m"][:2]
    plan = make_control_plan(
        inference.predicted_xy_m,
        destination,
        cube_z,
        grasp_config,
        motion_config,
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)
    width, height = camera_config["resolution"]
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")
    step_count = 0
    stage_frames: dict[str, int] = {}

    def step_and_record(stage: str) -> None:
        nonlocal step_count
        scene.step()
        step_count += 1
        if step_count % frame_stride == 0:
            frame, _, _, _ = camera.render(rgb=True, force_render=True)
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
            stage_frames[stage] = stage_frames.get(stage, 0) + 1

    def hold(stage: str, steps: int, closed: bool) -> None:
        for _ in range(steps):
            if closed:
                panda.control_dofs_force(close_force, dofs_idx_local=finger_dofs)
            else:
                panda.control_dofs_position(open_width, dofs_idx_local=finger_dofs)
            step_and_record(stage)

    def move(stage: str, target: list[float], steps: int, closed: bool) -> None:
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
            step_and_record(stage)

    try:
        hold("initial", 60, False)
        move("move_above", plan.pregrasp_hand_position_m, motion_config["move_steps"], False)
        move("descend", plan.grasp_hand_position_m, motion_config["descend_steps"], False)
        hold("close", motion_config["close_steps"], True)
        move("lift", plan.lift_hand_position_m, motion_config["lift_steps"], True)
        move("transport", plan.transport_hand_position_m, motion_config["transport_steps"], True)
        move("place", plan.place_hand_position_m, motion_config["place_steps"], True)
        hold("open", motion_config["open_steps"], False)
        move("retreat", plan.retreat_hand_position_m, motion_config["retreat_steps"], False)
        hold("settle", phase2["simulation"]["settle_steps"], False)
    finally:
        writer.release()

    final_position = to_numpy(cubes[target_class].get_pos())  # type: ignore[union-attr]
    destination_array = np.asarray(destination)
    final_error_mm = float(np.linalg.norm(final_position[:2] - destination_array) * 1000.0)
    success = bool(
        final_error_mm <= config["placement_xy_tolerance_m"] * 1000.0
        and abs(final_position[2] - cube_z) <= config["support_height_tolerance_m"]
    )
    result = {
        "model_label": model_label,
        "seed": seed,
        "target_class": target_class,
        "destination_id": destination_id,
        "predicted_xy_m": inference.predicted_xy_m,
        "truth_xy_m": truth_xy.tolist(),
        "prediction_error_mm": prediction_error_mm,
        "final_position_m": final_position.tolist(),
        "final_placement_error_mm": final_error_mm,
        "placement_success": success,
        "frame_count": sum(stage_frames.values()),
        "fps": fps,
        "duration_seconds": sum(stage_frames.values()) / fps,
        "frame_stride": frame_stride,
        "stage_frames": stage_frames,
        "video_path": str(output_video.resolve()),
        "control_change": False,
        "truth_used_for_control": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--target-class", default="red_cube")
    parser.add_argument("--destination-id", default="left_bin")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = record_trial(
        args.config.resolve(),
        args.seed,
        args.target_class,
        args.destination_id,
        args.model_label,
        args.output_video.resolve(),
        args.output_json.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
