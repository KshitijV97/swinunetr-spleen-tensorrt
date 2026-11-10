"""Benchmark PyTorch FP32, PyTorch AMP and TensorRT FP16 on the validation set.

Examples
--------
    python -m src.benchmark                 # on a CUDA GPU: all three backends
    python -m src.benchmark --smoke-test    # on a Mac: PyTorch FP32 only

Backends
--------
* ``pytorch_fp32``  - the trained model as-is, float32 everywhere.
* ``pytorch_amp``   - same weights under ``torch.autocast(float16)``. CUDA only.
* ``tensorrt_fp16`` - the ONNX export compiled by TensorRT with FP16 enabled.
                      CUDA + ``tensorrt`` package only; the engine is built on
                      first use, cached at ``benchmark.engine_path``, and
                      rebuilt when the ONNX file, TensorRT or GPU changes.
Unavailable backends are skipped and recorded as such, never faked.

What is timed (see docs/reference/benchmarking.md for the full methodology)
-----------------------------------------------------------------
Latency = wall-clock time for sliding-window inference over one whole,
already-preprocessed validation volume that is already on the GPU, i.e. the
network work only. Loading/resampling the NIfTI file and computing Dice are
excluded. CUDA work is asynchronous, so we call ``torch.cuda.synchronize()``
before starting and stopping the clock. The first ``warmup_volumes`` passes
are not timed (CUDA context creation, cuDNN autotuning, memory pool growth).

Outputs
-------
``<results_dir>/benchmark.json`` (everything, including per-volume numbers and
environment info) and ``<results_dir>/benchmark.md`` (a table for the README).
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.data import SYNTHETIC_TASK, build_dataloaders
from src.evaluation import DiceScorer, predict_volume
from src.model import load_checkpoint
from src.utils import environment_info, get_device, load_config, resolve_path, save_json, set_seed


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work so the timer measures real execution time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_backend(
    name: str,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    volumes: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    device: torch.device,
    autocast=nullcontext,
) -> dict[str, Any]:
    """Time sliding-window inference and score Dice for one backend."""
    roi = cfg["data"]["roi_size"]
    sw, overlap = cfg["inference"]["sw_batch_size"], cfg["inference"]["overlap"]

    def infer(image):
        with torch.no_grad(), autocast():
            return predict_volume(predictor, image, roi, sw, overlap)

    for image, _ in volumes[: cfg["benchmark"]["warmup_volumes"]]:
        infer(image)
    _sync(device)

    scorer = DiceScorer(cfg["model"]["out_channels"])
    latencies, dices = [], []
    for repeat in range(cfg["benchmark"]["repeats"]):
        for image, label in volumes:
            _sync(device)
            t0 = time.perf_counter()
            logits = infer(image)
            _sync(device)
            latencies.append((time.perf_counter() - t0) * 1000)
            if repeat == 0:  # Dice is deterministic; score each volume once
                dices.append(scorer(logits.float(), label))

    lat = np.array(latencies)
    result = {
        "backend": name,
        "status": "ok",
        "n_volumes": len(volumes),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_std": float(lat.std()),
        "dice_mean": scorer.aggregate(),
        "dice_per_volume": dices,
        "latency_ms_all": latencies,
    }
    if device.type == "cuda":
        result["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
    print(f"{name:14s} mean {result['latency_ms_mean']:9.1f} ms  p95 {result['latency_ms_p95']:9.1f} ms  "
          f"Dice {result['dice_mean']:.4f}")
    return result


def to_markdown(results: list[dict[str, Any]], env: dict[str, Any]) -> str:
    """Render the results as a Markdown table (pasted into the README)."""
    baseline = next((r for r in results if r["backend"] == "pytorch_fp32" and r["status"] == "ok"), None)
    lines = [
        f"Hardware: {env.get('gpu', env['device'] + ' / ' + env['platform'])} | torch {env['torch']}"
        + (f" | TensorRT {env['tensorrt']}" if "tensorrt" in env else ""),
        "",
        "| Backend | Mean latency / volume (ms) | p95 latency (ms) | Speed-up vs FP32 | Mean Dice |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        if r["status"] != "ok":
            lines.append(f"| {r['backend']} | skipped: {r['reason']} | | | |")
            continue
        speedup = baseline["latency_ms_mean"] / r["latency_ms_mean"] if baseline else float("nan")
        lines.append(
            f"| {r['backend']} | {r['latency_ms_mean']:.1f} | {r['latency_ms_p95']:.1f} | "
            f"{speedup:.2f}x | {r['dice_mean']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT inference.")
    p.add_argument("--config", default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="use generated fake volumes")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint", help="default: <train.checkpoint_dir>/best.pt")
    p.add_argument("--rebuild-engine", action="store_true", help="rebuild the TensorRT engine")
    args = p.parse_args(argv)

    cfg = load_config(args.config, smoke_test=args.smoke_test)
    if args.synthetic:
        cfg["data"]["task"] = SYNTHETIC_TASK
    set_seed(cfg["seed"])
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_path(cfg["train"]["checkpoint_dir"]) / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device=device)
    # Use the crop size the model was trained (and exported) with.
    cfg["data"]["roi_size"] = ckpt["config"]["data"]["roi_size"]

    # Preprocess every validation volume once and keep it on the device, so
    # data loading never leaks into the timings.
    _, val_loader = build_dataloaders(cfg)
    volumes = [(b["image"].to(device), b["label"].to(device)) for b in val_loader]
    print(f"Benchmarking on {len(volumes)} validation volumes, device={device}, checkpoint={ckpt_path}")

    env = environment_info()
    env["device"] = device.type
    results: list[dict[str, Any]] = []
    results.append(run_backend("pytorch_fp32", model, volumes, cfg, device))

    if device.type == "cuda":
        amp = lambda: torch.autocast("cuda", dtype=torch.float16)  # noqa: E731
        results.append(run_backend("pytorch_amp", model, volumes, cfg, device, autocast=amp))
    else:
        results.append({"backend": "pytorch_amp", "status": "skipped", "reason": "needs CUDA"})

    engine_info: dict[str, Any] = {}
    if device.type == "cuda":
        # Imported here, never at module level: tensorrt only exists on CUDA machines.
        from src.trt_utils import TRTPredictor, ensure_engine, tensorrt_available

        if tensorrt_available():
            onnx_path = resolve_path(cfg["export"]["onnx_path"])
            engine_path = resolve_path(cfg["benchmark"]["engine_path"])
            if not onnx_path.exists():
                raise SystemExit(f"{onnx_path} not found - run `python -m src.export_onnx` first.")
            engine_info = ensure_engine(onnx_path, engine_path, cfg["benchmark"]["workspace_gb"],
                                        force=args.rebuild_engine)
            import tensorrt as trt

            env["tensorrt"] = trt.__version__
            del model  # free GPU memory for a fair TensorRT measurement
            torch.cuda.empty_cache()
            results.append(run_backend("tensorrt_fp16", TRTPredictor(engine_path), volumes, cfg, device))
        else:
            results.append({"backend": "tensorrt_fp16", "status": "skipped", "reason": "tensorrt not installed"})
    else:
        results.append({"backend": "tensorrt_fp16", "status": "skipped", "reason": "needs CUDA"})

    out_dir = resolve_path(cfg["results_dir"])
    report = {
        "smoke_test": cfg["smoke_test"],
        "checkpoint": str(ckpt_path),
        "checkpoint_val_dice": ckpt.get("val_dice"),
        "roi_size": cfg["data"]["roi_size"],
        "sw_batch_size": cfg["inference"]["sw_batch_size"],
        "overlap": cfg["inference"]["overlap"],
        "warmup_volumes": cfg["benchmark"]["warmup_volumes"],
        "repeats": cfg["benchmark"]["repeats"],
        "tensorrt_engine": engine_info,
        "environment": env,
        "results": results,
    }
    save_json(report, out_dir / "benchmark.json")
    md = to_markdown(results, env)
    (out_dir / "benchmark.md").write_text(md)
    print("\n" + md)
    print(f"Wrote {out_dir / 'benchmark.json'} and {out_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()
