"""Validate Franka Panda IK through simulated joint-position control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .runtime import PROJECT_ROOT, configure_genesis_cache

TARGETS = (
    ("front_low", (0.45, 0.00, 0.30)),
    ("left_low", (0.48, 0.14, 0.34)),
    ("right_low", (0.48, -0.14, 0.34)),
    ("left_high", (0.40, 0.12, 0.46)),
    ("right_high", (0.40, -0.12, 0.46)),
)
TARGET_QUATERNION = np.array((0.0, 1.0, 0.0, 0.0), dtype=np.float64)


@dataclass
class TargetResult:
    name: str
    success: bool
    target_position_m: list[float]
    actual_position_m: list[float] | None
    target_quaternion_wxyz: list[float]
    actual_quaternion_wxyz: list[float] | None
    position_error_m: float | None
    orientation_error_deg: float | None
    ik_position_residual_m: float | None
    ik_orientation_residual_rad: float | None
    ik_seconds: float
    control_seconds: float
    steps: int
    failure: str | None


def to_numpy(value: object) -> np.ndarray:
    """Convert Genesis/Torch output to a CPU NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()  # type: ignore[union-attr]
    return np.asarray(value, dtype=np.float64)


def quaternion_error_deg(target: np.ndarray, actual: np.ndarray) -> float:
    """Return the shortest angular distance between wxyz quaternions."""
    target = target / np.linalg.norm(target)
    actual = actual / np.linalg.norm(actual)
    cosine = float(np.clip(abs(np.dot(target, actual)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def save_image(camera: object, path: Path) -> None:
    from PIL import Image

    rgb, _, _, _ = camera.render(rgb=True)  # type: ignore[union-attr]
    image = to_numpy(rgb).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate five Franka Panda IK targets")
    parser.add_argument("--steps-per-target", type=int, default=300)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "phase1")
    args = parser.parse_args()

    configure_genesis_cache("cpu")

    import genesis as gs

    total_started = time.perf_counter()
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    camera = None
    if args.save_images:
        camera = scene.add_camera(
            res=(640, 480),
            pos=(1.45, -1.35, 1.05),
            lookat=(0.42, 0.0, 0.32),
            fov=40,
            GUI=False,
        )
    scene.build()

    arm_dofs = np.arange(7, dtype=np.int32)
    initial_qpos = np.array((0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.8, 0.04, 0.04))
    panda.set_dofs_kp(np.array((4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100)))
    panda.set_dofs_kv(np.array((450, 450, 350, 350, 200, 200, 200, 10, 10)))
    panda.set_dofs_force_range(
        np.array((-87, -87, -87, -87, -12, -12, -12, -100, -100)),
        np.array((87, 87, 87, 87, 12, 12, 12, 100, 100)),
    )
    panda.set_qpos(initial_qpos, zero_velocity=True)
    panda.control_dofs_position(initial_qpos)
    for _ in range(100):
        scene.step()

    end_effector = panda.get_link("hand")
    results: list[TargetResult] = []

    for name, target_tuple in TARGETS:
        target = np.asarray(target_tuple, dtype=np.float64)
        ik_started = time.perf_counter()
        try:
            qpos, ik_error = panda.inverse_kinematics(
                link=end_effector,
                pos=target,
                quat=TARGET_QUATERNION,
                dofs_idx_local=arm_dofs,
                return_error=True,
            )
            ik_seconds = time.perf_counter() - ik_started
            qpos_array = to_numpy(qpos)
            ik_error_array = to_numpy(ik_error)

            control_started = time.perf_counter()
            panda.control_dofs_position(qpos_array[:7], dofs_idx_local=arm_dofs)
            for _ in range(args.steps_per_target):
                scene.step()
            control_seconds = time.perf_counter() - control_started

            actual_position = to_numpy(end_effector.get_pos())
            actual_quaternion = to_numpy(end_effector.get_quat())
            position_error = float(np.linalg.norm(actual_position - target))
            orientation_error = quaternion_error_deg(TARGET_QUATERNION, actual_quaternion)
            success = position_error <= 0.01 and orientation_error <= 3.0
            failure = None if success else "final pose exceeded tolerance"

            if camera is not None:
                save_image(camera, args.output_dir / "images" / f"{name}.png")

            results.append(
                TargetResult(
                    name=name,
                    success=success,
                    target_position_m=target.tolist(),
                    actual_position_m=actual_position.tolist(),
                    target_quaternion_wxyz=TARGET_QUATERNION.tolist(),
                    actual_quaternion_wxyz=actual_quaternion.tolist(),
                    position_error_m=position_error,
                    orientation_error_deg=orientation_error,
                    ik_position_residual_m=float(np.linalg.norm(ik_error_array[:3])),
                    ik_orientation_residual_rad=float(np.linalg.norm(ik_error_array[3:])),
                    ik_seconds=ik_seconds,
                    control_seconds=control_seconds,
                    steps=args.steps_per_target,
                    failure=failure,
                )
            )
        except Exception as exc:  # noqa: BLE001 - each target failure must be persisted.
            results.append(
                TargetResult(
                    name=name,
                    success=False,
                    target_position_m=target.tolist(),
                    actual_position_m=None,
                    target_quaternion_wxyz=TARGET_QUATERNION.tolist(),
                    actual_quaternion_wxyz=None,
                    position_error_m=None,
                    orientation_error_deg=None,
                    ik_position_residual_m=None,
                    ik_orientation_residual_rad=None,
                    ik_seconds=time.perf_counter() - ik_started,
                    control_seconds=0.0,
                    steps=0,
                    failure=f"{type(exc).__name__}: {exc}",
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "genesis": gs.__version__,
        "backend": "cpu",
        "steps_per_target": args.steps_per_target,
        "success_count": sum(item.success for item in results),
        "target_count": len(results),
        "total_elapsed_seconds": time.perf_counter() - total_started,
        "results": [asdict(item) for item in results],
    }
    json_path = args.output_dir / "panda_ik_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "panda_ik_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)

    for item in results:
        print(
            f"{item.name}: {'PASS' if item.success else 'FAIL'} "
            f"position={item.position_error_m!r} m orientation={item.orientation_error_deg!r} deg"
        )
    print(f"Panda IK validation: {payload['success_count']}/{payload['target_count']} PASS")
    print(f"Results: {json_path}")

    if payload["success_count"] != payload["target_count"]:
        raise RuntimeError("One or more Panda IK targets failed")


if __name__ == "__main__":
    main()

