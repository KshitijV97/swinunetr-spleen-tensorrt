"""Export the best checkpoint to ONNX and check it matches PyTorch.

Examples
--------
    python -m src.export_onnx                 # uses checkpoints/best.pt
    python -m src.export_onnx --smoke-test    # uses checkpoints/smoke/best.pt

What is ONNX and why a fixed shape?
-----------------------------------
ONNX is a framework-neutral file format for a neural network's computation
graph plus its weights. TensorRT (and ONNX Runtime, OpenVINO, ...) read it.

We export with a single, fixed input shape ``(sw_batch_size, 1, *roi_size)``,
e.g. ``(4, 1, 96, 96, 96)``, because:
  * sliding-window inference only ever feeds the network that exact shape;
  * Swin's window partitioning reshapes tensors based on the input size, and
    fixed shapes keep those reshapes as constants, which exports cleanly;
  * TensorRT can pick the fastest kernels for one known shape.

Verification
------------
After export we run the same random input through PyTorch and through ONNX
Runtime (CPU) and report the max absolute difference of the logits, plus the
fraction of voxels where the predicted class (argmax) agrees. The export
fails loudly unless every logit satisfies
``|pytorch - onnx| <= atol + rtol * |pytorch|`` (the ``numpy.allclose`` rule).
Small differences are expected: the two runtimes add numbers in different
orders, and float32 addition is not associative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.model import load_checkpoint
from src.utils import load_config, resolve_path, save_json


@torch.no_grad()
def export(model: torch.nn.Module, onnx_path: Path, input_shape, opset: int) -> Path:
    """Write ``model`` to ``onnx_path`` with one fixed input shape."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    dummy = torch.rand(*input_shape, device=device)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=opset,
        # The TorchScript-based exporter handles SwinUNETR's window reshapes
        # reliably; the newer torch.export ("dynamo") path is not needed here.
        dynamo=False,
        do_constant_folding=True,
        # No dynamic_axes on purpose: every dimension is fixed (see module docstring).
    )
    return onnx_path


def verify(model: torch.nn.Module, onnx_path: Path, input_shape, atol: float, rtol: float,
           seed: int = 0) -> dict:
    """Compare PyTorch and ONNX Runtime outputs on the same random input."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(onnx_path))  # structural validity of the file

    torch.manual_seed(seed)
    x = torch.rand(*input_shape)  # inputs are in [0, 1] after intensity scaling
    device = next(model.parameters()).device
    # On NVIDIA GPUs cuDNN may use TF32 (10-bit mantissa) for "fp32" convolutions,
    # which alone can exceed the tolerance. Turn it off for a true fp32 reference.
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            ref = model(x.to(device)).float().cpu().numpy()
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = session.run(["logits"], {"image": x.numpy()})[0]

    diff = np.abs(ref - out)
    within = diff <= atol + rtol * np.abs(ref)
    report = {
        "input_shape": list(input_shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "fraction_within_tolerance": float(within.mean()),
        "argmax_agreement": float((ref.argmax(1) == out.argmax(1)).mean()),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(within.all()),
    }
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Export SwinUNETR to ONNX and verify it.")
    p.add_argument("--config", default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--checkpoint", help="default: <train.checkpoint_dir>/best.pt")
    p.add_argument("--output", help="default: export.onnx_path from the config")
    args = p.parse_args(argv)

    cfg = load_config(args.config, smoke_test=args.smoke_test)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_path(cfg["train"]["checkpoint_dir"]) / "best.pt"
    onnx_path = Path(args.output) if args.output else resolve_path(cfg["export"]["onnx_path"])

    # Export in fp32: the ONNX file stores fp32 weights and TensorRT decides
    # later which layers run in fp16. Use the GPU when there is one - tracing
    # the full-size model at batch 4 on CPU needs a lot of RAM on Colab.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(ckpt_path, device=device)
    train_cfg = ckpt["config"]  # the shapes the model was actually trained with
    input_shape = (cfg["inference"]["sw_batch_size"], train_cfg["model"]["in_channels"], *train_cfg["data"]["roi_size"])

    print(f"Exporting {ckpt_path} -> {onnx_path} with input shape {input_shape}")
    export(model, onnx_path, input_shape, cfg["export"]["opset"])
    report = verify(model, onnx_path, input_shape, cfg["export"]["atol"], cfg["export"]["rtol"],
                    seed=cfg["seed"])
    report.update(checkpoint=str(ckpt_path), onnx_path=str(onnx_path),
                  onnx_size_mb=onnx_path.stat().st_size / 2**20, opset=cfg["export"]["opset"])
    save_json(report, resolve_path(cfg["results_dir"]) / "onnx_export.json")

    print(f"max |PyTorch - ONNX Runtime| = {report['max_abs_diff']:.2e}, "
          f"{report['fraction_within_tolerance'] * 100:.4f}% of logits within atol={report['atol']:.0e} "
          f"+ rtol={report['rtol']:.0e}, argmax agreement {report['argmax_agreement'] * 100:.4f}%")
    if not report["passed"]:
        raise SystemExit("ONNX output differs from PyTorch by more than the tolerance")
    print("ONNX export verified.")


if __name__ == "__main__":
    main()
