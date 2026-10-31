"""Run a deterministic CPU Genesis smoke test."""

from __future__ import annotations

import json
import time

from .runtime import PROJECT_ROOT, configure_genesis_cache


def main() -> None:
    configure_genesis_cache("cpu")

    import genesis as gs

    started = time.perf_counter()
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    sphere = scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0), radius=0.08))
    scene.build()

    initial_z = float(sphere.get_pos()[-1])
    for _ in range(30):
        scene.step()
    final_z = float(sphere.get_pos()[-1])
    elapsed = time.perf_counter() - started

    if final_z >= initial_z:
        raise RuntimeError(f"Sphere did not fall: {initial_z=}, {final_z=}")

    result = {
        "backend": "cpu",
        "genesis": gs.__version__,
        "steps": 30,
        "dt_seconds": 0.01,
        "initial_z_m": initial_z,
        "final_z_m": final_z,
        "elapsed_seconds": elapsed,
    }
    output = PROJECT_ROOT / "outputs" / "phase0" / "cpu_smoke_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Genesis CPU smoke test PASS: z={initial_z:.3f}->{final_z:.3f}, {elapsed:.3f}s")


if __name__ == "__main__":
    main()

