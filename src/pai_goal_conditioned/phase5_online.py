"""Run open-loop Goal-conditioned Panda manipulation using CPU inference."""

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
import torch
from PIL import Image, ImageDraw

from .goal_vision import (
    COLOR_TO_ID,
    DESTINATION_TO_ID,
    GoalConditionedGraspModel,
    TargetNormalizer,
    preprocess_rgb_array,
    project_world_to_pixel,
)
from .panda_ik import to_numpy
from .phase3_dataset import sample_non_overlapping_positions
from .runtime import PROJECT_ROOT, configure_genesis_cache

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase5_online.json"


@dataclass(frozen=True)
class InferenceResult:
    predicted_xy_m: list[float]
    inference_seconds: float


@dataclass(frozen=True)
class ControlPlan:
    predicted_grasp_xy_m: list[float]
    destination_xy_m: list[float]
    pregrasp_hand_position_m: list[float]
    grasp_hand_position_m: list[float]
    lift_hand_position_m: list[float]
    transport_hand_position_m: list[float]
    place_hand_position_m: list[float]
    retreat_hand_position_m: list[float]


@dataclass
class OnlineTrialResult:
    trial: int
    seed: int
    target_class: str
    destination_id: str
    success: bool
    predicted_xy_m: list[float] | None
    target_truth_xy_m: list[float]
    prediction_error_mm: float | None
    workspace_accepted: bool
    correct_object_grasped: bool
    lift_success: bool
    transport_success: bool
    placement_success: bool
    stable_after_release: bool
    instruction_achievement_success: bool
    lifted_object_class: str | None
    final_target_position_m: list[float] | None
    final_placement_error_mm: float | None
    failed_stage: str | None
    failure_reason: str | None
    inference_seconds: float
    physical_operation_seconds: float
    total_seconds: float
    initial_object_positions_m: dict[str, list[float]]
    image_path: str
    evaluation_overlay_path: str | None
    lift_image_path: str | None
    final_image_path: str | None


class GraspPredictor:
    """CPU-only inference boundary; accepts no simulator truth or scene identifiers."""

    def __init__(
        self,
        model: GoalConditionedGraspModel,
        target_normalizer: TargetNormalizer,
        image_size_wh: tuple[int, int],
        image_mean: list[float],
        image_std: list[float],
    ) -> None:
        self.model = model.cpu().eval()
        self.target_normalizer = target_normalizer
        self.image_size_wh = image_size_wh
        self.image_mean = image_mean
        self.image_std = image_std

    @classmethod
    def from_artifacts(
        cls, model_config_path: Path, normalization_path: Path, state_dict_path: Path
    ) -> GraspPredictor:
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        model = GoalConditionedGraspModel(
            mode="full",
            image_feature_dim=model_config["image_feature_dim"],
            color_embedding_dim=model_config["color_embedding_dim"],
            destination_embedding_dim=model_config["destination_embedding_dim"],
            dropout=model_config["dropout"],
            image_mean=normalization["image_mean"],
            image_std=normalization["image_std"],
        )
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        target_normalizer = TargetNormalizer(**normalization["target"])
        return cls(
            model=model,
            target_normalizer=target_normalizer,
            image_size_wh=tuple(normalization["image_size_wh"]),
            image_mean=normalization["image_mean"],
            image_std=normalization["image_std"],
        )

    def predict(self, rgb: np.ndarray, target_class: str, destination_id: str) -> InferenceResult:
        started = time.perf_counter()
        image = preprocess_rgb_array(
            rgb,
            self.image_size_wh,
            self.image_mean,
            self.image_std,
        ).unsqueeze(0)
        color_id = torch.tensor([COLOR_TO_ID[target_class]], dtype=torch.long)
        destination = torch.tensor([DESTINATION_TO_ID[destination_id]], dtype=torch.long)
        with torch.inference_mode():
            normalized = self.model(image, color_id, destination)
            predicted = self.target_normalizer.denormalize_tensor(normalized)[0]
        return InferenceResult(
            predicted_xy_m=predicted.cpu().numpy().astype(float).tolist(),
            inference_seconds=time.perf_counter() - started,
        )


def workspace_contains(
    predicted_xy_m: list[float], x_range_m: list[float], y_range_m: list[float]
) -> bool:
    return bool(
        x_range_m[0] <= predicted_xy_m[0] <= x_range_m[1]
        and y_range_m[0] <= predicted_xy_m[1] <= y_range_m[1]
    )


def make_control_plan(
    predicted_xy_m: list[float],
    destination_xy_m: list[float],
    supported_cube_center_z_m: float,
    grasp_config: dict[str, Any],
    motion_config: dict[str, Any],
) -> ControlPlan:
    """Create motion targets only from AI XY, known destination, and scene geometry."""
    grasp_z = supported_cube_center_z_m + grasp_config["hand_to_cube_center_z_offset_m"]
    pregrasp_z = (
        supported_cube_center_z_m + grasp_config["pregrasp_to_cube_center_z_offset_m"]
    )
    lift_z = motion_config["lift_hand_position_m"][2]
    retreat_z = motion_config["retreat_hand_position_m"][2]
    return ControlPlan(
        predicted_grasp_xy_m=list(predicted_xy_m),
        destination_xy_m=list(destination_xy_m),
        pregrasp_hand_position_m=[*predicted_xy_m, pregrasp_z],
        grasp_hand_position_m=[*predicted_xy_m, grasp_z],
        lift_hand_position_m=[*predicted_xy_m, lift_z],
        transport_hand_position_m=[*destination_xy_m, lift_z],
        place_hand_position_m=[*destination_xy_m, grasp_z],
        retreat_hand_position_m=[*destination_xy_m, retreat_z],
    )


def classify_failure(
    workspace_accepted: bool,
    correct_object_grasped: bool,
    lift_success: bool,
    transport_success: bool,
    placement_success: bool,
    stable_after_release: bool,
) -> str | None:
    if not workspace_accepted:
        return "workspace_rejection"
    if not correct_object_grasped:
        return "grasp"
    if not lift_success:
        return "lift"
    if not transport_success:
        return "transport"
    if not placement_success:
        return "placement"
    if not stable_after_release:
        return "settle"
    return None


def _load_configs(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["phase2"] = json.loads(
        (PROJECT_ROOT / config["phase2_config"]).read_text(encoding="utf-8")
    )
    config["phase3"] = json.loads(
        (PROJECT_ROOT / config["phase3_config"]).read_text(encoding="utf-8")
    )
    return config


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def _save_evaluation_overlay(
    rgb: np.ndarray,
    predicted_xy_m: list[float],
    truth_xyz_m: list[float],
    camera_config: dict[str, Any],
    path: Path,
) -> None:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    predicted_pixel = project_world_to_pixel(
        [predicted_xy_m[0], predicted_xy_m[1], truth_xyz_m[2]], camera_config
    )
    truth_pixel = project_world_to_pixel(truth_xyz_m, camera_config)
    for point, color, radius in (
        (truth_pixel, "#20ff60", 8),
        (predicted_pixel, "#ff3030", 6),
    ):
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.text((10, 10), "green=truth (evaluation only), red=AI prediction", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _capture_camera(camera: object) -> np.ndarray:
    rgb, _, _, _ = camera.render(rgb=True, force_render=True)  # type: ignore[union-attr]
    return np.asarray(rgb, dtype=np.uint8)


def run_trials(config_path: Path, trial_count: int, base_seed: int, output_dir: Path) -> dict[str, Any]:
    configure_genesis_cache("cpu")
    import genesis as gs

    config = _load_configs(config_path)
    phase2 = config["phase2"]
    phase3 = config["phase3"]
    scene_config = phase3["scene"]
    sim_config = phase3["simulation"]
    camera_config = phase3["camera"]
    robot_config = phase2["robot"]
    motion_config = phase2["motion"]
    grasp_config = phase3["grasp"]
    predictor = GraspPredictor.from_artifacts(
        PROJECT_ROOT / config["phase4_config"],
        PROJECT_ROOT / config["normalization"],
        PROJECT_ROOT / config["model_state_dict"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    gs.init(backend=gs.cpu, logging_level="warning", seed=base_seed)
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
    destinations = {item["id"]: item for item in phase3["destinations"]}
    schedule = config["trial_schedule"]
    results: list[OnlineTrialResult] = []
    instruction_sensitivity: dict[str, Any] | None = None

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
            arm_target = start_qpos + smooth_alpha * (qpos - start_qpos)
            panda.control_dofs_position(arm_target, dofs_idx_local=arm_dofs)
            if closed:
                panda.control_dofs_force(close_force, dofs_idx_local=finger_dofs)
            else:
                panda.control_dofs_position(open_width, dofs_idx_local=finger_dofs)
            scene.step()

    for trial_index in range(trial_count):
        trial_started = time.perf_counter()
        trial_number = trial_index + 1
        seed = base_seed + trial_index
        target_class, destination_id = schedule[trial_index % len(schedule)]
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
        for (object_class, cube), position in zip(cubes.items(), positions):
            yaw = rng.uniform(-math.pi, math.pi)
            cube.set_pos(position, zero_velocity=True)  # type: ignore[union-attr]
            cube.set_quat(  # type: ignore[union-attr]
                (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), zero_velocity=True
            )
        hold(sim_config["settle_steps"], closed=False)

        # Evaluation truth is captured into a separate record and is never passed to inference/control.
        evaluation_initial_positions = {
            object_class: to_numpy(cube.get_pos()).tolist()  # type: ignore[union-attr]
            for object_class, cube in cubes.items()
        }
        rgb = _capture_camera(camera)
        image_path = output_dir / "images" / f"trial_{trial_number:02d}_input.png"
        _save_rgb(rgb, image_path)
        inference = predictor.predict(rgb, target_class, destination_id)
        if instruction_sensitivity is None:
            color_predictions = {
                color: predictor.predict(rgb, color, "left_bin").predicted_xy_m
                for color in COLOR_TO_ID
            }
            pairwise_distances_mm: dict[str, float] = {}
            color_names = list(color_predictions)
            for left_index, left in enumerate(color_names):
                for right in color_names[left_index + 1 :]:
                    pairwise_distances_mm[f"{left}__{right}"] = float(
                        np.linalg.norm(
                            np.asarray(color_predictions[left])
                            - np.asarray(color_predictions[right])
                        )
                        * 1000.0
                    )
            instruction_sensitivity = {
                "source_image": image_path.relative_to(output_dir).as_posix(),
                "fixed_destination_id": "left_bin",
                "predictions_by_target_class_m": color_predictions,
                "pairwise_prediction_distances_mm": pairwise_distances_mm,
                "all_color_predictions_distinct": all(
                    distance > 0.001 for distance in pairwise_distances_mm.values()
                ),
            }
        target_truth = evaluation_initial_positions[target_class]
        prediction_error_mm = float(
            np.linalg.norm(
                np.asarray(inference.predicted_xy_m) - np.asarray(target_truth[:2])
            )
            * 1000.0
        )
        accepted = workspace_contains(
            inference.predicted_xy_m,
            config["workspace_x_range_m"],
            config["workspace_y_range_m"],
        )
        overlay_path = output_dir / "overlays" / f"trial_{trial_number:02d}_prediction.png"
        _save_evaluation_overlay(
            rgb,
            inference.predicted_xy_m,
            target_truth,
            camera_config,
            overlay_path,
        )
        physical_started = time.perf_counter()
        correct_object_grasped = False
        lift_success = False
        transport_success = False
        placement_success = False
        stable_after_release = False
        lifted_object_class: str | None = None
        final_target_position: np.ndarray | None = None
        final_placement_error_mm: float | None = None
        lift_path: Path | None = None
        final_path: Path | None = None
        failure_reason: str | None = None
        try:
            if not accepted:
                failure_reason = "AI prediction lies outside configured workspace"
            else:
                destination_xy = destinations[destination_id]["world_position_m"][:2]
                plan = make_control_plan(
                    inference.predicted_xy_m,
                    destination_xy,
                    cube_z,
                    grasp_config,
                    motion_config,
                )
                move(plan.pregrasp_hand_position_m, motion_config["move_steps"], False)
                move(plan.grasp_hand_position_m, motion_config["descend_steps"], False)
                hold(motion_config["close_steps"], True)
                move(plan.lift_hand_position_m, motion_config["lift_steps"], True)
                lifted_positions = {
                    object_class: to_numpy(cube.get_pos())  # type: ignore[union-attr]
                    for object_class, cube in cubes.items()
                }
                lifted_candidates = [
                    object_class
                    for object_class, position in lifted_positions.items()
                    if position[2]
                    >= evaluation_initial_positions[object_class][2]
                    + config["lift_height_threshold_m"]
                ]
                lifted_object_class = lifted_candidates[0] if len(lifted_candidates) == 1 else None
                correct_object_grasped = lifted_object_class == target_class
                lift_success = correct_object_grasped
                lift_path = output_dir / "actions" / f"trial_{trial_number:02d}_lift.png"
                _save_rgb(_capture_camera(camera), lift_path)

                move(plan.transport_hand_position_m, motion_config["transport_steps"], True)
                transport_position = to_numpy(cubes[target_class].get_pos())  # type: ignore[union-attr]
                transport_success = bool(
                    lift_success
                    and np.linalg.norm(transport_position[:2] - np.asarray(destination_xy))
                    <= config["transport_xy_tolerance_m"]
                    and transport_position[2] >= cube_z + config["lift_height_threshold_m"]
                )
                move(plan.place_hand_position_m, motion_config["place_steps"], True)
                hold(motion_config["open_steps"], False)
                move(plan.retreat_hand_position_m, motion_config["retreat_steps"], False)
                hold(phase2["simulation"]["settle_steps"], False)
                final_target_position = to_numpy(cubes[target_class].get_pos())  # type: ignore[union-attr]
                final_placement_error_mm = float(
                    np.linalg.norm(final_target_position[:2] - np.asarray(destination_xy)) * 1000.0
                )
                target_speed = float(
                    np.linalg.norm(to_numpy(cubes[target_class].get_vel()))  # type: ignore[union-attr]
                )
                target_angular_speed = float(
                    np.linalg.norm(to_numpy(cubes[target_class].get_ang()))  # type: ignore[union-attr]
                )
                placement_success = bool(
                    final_placement_error_mm <= config["placement_xy_tolerance_m"] * 1000.0
                    and abs(final_target_position[2] - cube_z)
                    <= config["support_height_tolerance_m"]
                )
                stable_after_release = bool(
                    target_speed <= config["stable_linear_speed_m_s"]
                    and target_angular_speed <= config["stable_angular_speed_rad_s"]
                )
                final_path = output_dir / "actions" / f"trial_{trial_number:02d}_final.png"
                _save_rgb(_capture_camera(camera), final_path)
        except Exception as exc:  # noqa: BLE001 - persist trial failures without retry.
            failure_reason = f"{type(exc).__name__}: {exc}"

        failed_stage = classify_failure(
            accepted,
            correct_object_grasped,
            lift_success,
            transport_success,
            placement_success,
            stable_after_release,
        )
        if failed_stage and failure_reason is None:
            failure_reason = f"evaluation condition failed at {failed_stage}"
        instruction_success = bool(
            accepted
            and correct_object_grasped
            and lift_success
            and transport_success
            and placement_success
            and stable_after_release
            and failure_reason is None
        )
        physical_seconds = time.perf_counter() - physical_started
        result = OnlineTrialResult(
            trial=trial_number,
            seed=seed,
            target_class=target_class,
            destination_id=destination_id,
            success=instruction_success,
            predicted_xy_m=inference.predicted_xy_m,
            target_truth_xy_m=target_truth[:2],
            prediction_error_mm=prediction_error_mm,
            workspace_accepted=accepted,
            correct_object_grasped=correct_object_grasped,
            lift_success=lift_success,
            transport_success=transport_success,
            placement_success=placement_success,
            stable_after_release=stable_after_release,
            instruction_achievement_success=instruction_success,
            lifted_object_class=lifted_object_class,
            final_target_position_m=(
                None if final_target_position is None else final_target_position.tolist()
            ),
            final_placement_error_mm=final_placement_error_mm,
            failed_stage=failed_stage,
            failure_reason=failure_reason,
            inference_seconds=inference.inference_seconds,
            physical_operation_seconds=physical_seconds,
            total_seconds=time.perf_counter() - trial_started,
            initial_object_positions_m=evaluation_initial_positions,
            image_path=image_path.relative_to(output_dir).as_posix(),
            evaluation_overlay_path=overlay_path.relative_to(output_dir).as_posix(),
            lift_image_path=None if lift_path is None else lift_path.relative_to(output_dir).as_posix(),
            final_image_path=(
                None if final_path is None else final_path.relative_to(output_dir).as_posix()
            ),
        )
        results.append(result)
        print(
            f"trial {trial_number:02d} {target_class}->{destination_id}: "
            f"{'PASS' if result.success else 'FAIL'} prediction_error={prediction_error_mm:.2f}mm "
            f"stage={failed_stage}",
            flush=True,
        )

    payload = _write_results(
        results,
        config,
        output_dir,
        time.perf_counter() - started_all,
        instruction_sensitivity,
    )
    _create_summary_plot(results, output_dir)
    return payload


def _write_results(
    results: list[OnlineTrialResult],
    config: dict[str, Any],
    output_dir: Path,
    elapsed_seconds: float,
    instruction_sensitivity: dict[str, Any] | None,
) -> dict[str, Any]:
    errors = [result.prediction_error_mm for result in results if result.prediction_error_mm is not None]
    success_count = sum(result.success for result in results)
    payload = {
        "phase": 5,
        "description": "open-loop Goal-conditioned AI physical manipulation",
        "backend": "Genesis 1.2.1 CPU and PyTorch CPU inference",
        "trial_count": len(results),
        "success_count": success_count,
        "success_rate": success_count / len(results),
        "prediction_error_mean_mm": float(np.mean(errors)),
        "prediction_error_median_mm": float(np.median(errors)),
        "prediction_error_max_mm": float(np.max(errors)),
        "mean_inference_seconds": float(np.mean([item.inference_seconds for item in results])),
        "mean_physical_operation_seconds": float(
            np.mean([item.physical_operation_seconds for item in results])
        ),
        "mean_total_seconds": float(np.mean([item.total_seconds for item in results])),
        "total_elapsed_seconds": elapsed_seconds,
        "phase2_reference": "10/10 with fixed truth-coordinate single-cube setup; not identical conditions",
        "closed_loop_retry": False,
        "truth_usage": "evaluation and visualization only",
        "instruction_sensitivity_check": instruction_sensitivity,
        "config": {key: value for key, value in config.items() if key not in {"phase2", "phase3"}},
        "results": [asdict(result) for result in results],
    }
    (output_dir / "phase5_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "trial",
        "seed",
        "target_class",
        "destination_id",
        "success",
        "predicted_xy_m",
        "target_truth_xy_m",
        "prediction_error_mm",
        "workspace_accepted",
        "correct_object_grasped",
        "lift_success",
        "transport_success",
        "placement_success",
        "stable_after_release",
        "instruction_achievement_success",
        "lifted_object_class",
        "final_placement_error_mm",
        "failed_stage",
        "failure_reason",
        "inference_seconds",
        "physical_operation_seconds",
        "total_seconds",
    ]
    with (output_dir / "phase5_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({field: row[field] for field in fields})
    return payload


def _create_summary_plot(results: list[OnlineTrialResult], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    errors = [result.prediction_error_mm for result in results]
    success = [1 if result.success else 0 for result in results]
    colors = ["#38a852" if value else "#d43f3a" for value in success]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(errors, success, c=colors, s=70)
    axis.set_xlabel("AI grasp XY prediction error (mm)")
    axis.set_ylabel("instruction success")
    axis.set_yticks([0, 1], ["FAIL", "PASS"])
    axis.grid(alpha=0.3)
    figure.tight_layout()
    plot_path = output_dir / "visualizations" / "prediction_error_vs_success.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    trial_count = args.trials or base_config["trial_count"]
    base_seed = args.seed or base_config["full_base_seed"]
    output_dir = args.output_dir or (PROJECT_ROOT / base_config["output_dir"])
    payload = run_trials(args.config.resolve(), trial_count, base_seed, output_dir.resolve())
    print(
        f"Phase 5: {payload['success_count']}/{payload['trial_count']} instruction successes, "
        f"mean prediction error={payload['prediction_error_mean_mm']:.3f} mm"
    )


if __name__ == "__main__":
    main()
