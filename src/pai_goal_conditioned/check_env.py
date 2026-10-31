"""Check the isolated simulation or training environment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .runtime import PROJECT_ROOT, configure_genesis_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the project environment")
    parser.add_argument("--mode", choices=("sim", "train"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch

    result: dict[str, object] = {
        "mode": args.mode,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    if args.mode == "sim":
        configure_genesis_cache("cpu")
        import genesis as gs

        result["genesis"] = gs.__version__
        if gs.__version__ != "1.2.1":
            raise RuntimeError(f"Expected Genesis 1.2.1, got {gs.__version__}")
    else:
        import torchvision

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the training environment")
        device = torch.device("cuda")
        lhs = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        rhs = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
        product = lhs @ rhs
        images = torch.randn(8, 3, 32, 32, device=device)
        conv = torch.nn.Conv2d(3, 8, kernel_size=3).to(device)
        features = conv(images)
        torch.cuda.synchronize()
        result["cuda_matmul_sum"] = float(product.sum().cpu())
        result["torchvision"] = torchvision.__version__
        result["cuda_conv_output_shape"] = list(features.shape)

    output = args.output or PROJECT_ROOT / "outputs" / "phase0" / f"{args.mode}_environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Environment check PASS: {output}")


if __name__ == "__main__":
    main()

