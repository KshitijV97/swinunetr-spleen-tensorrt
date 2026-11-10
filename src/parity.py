"""Numerical parity gate: is a deployed backend still "the same model"?

Examples
--------
    python -m src.parity --smoke-test                          # ONNX Runtime + TensorRT (if CUDA)
    python -m src.parity --backend onnxruntime                 # PyTorch FP32 vs ONNX Runtime
    python -m src.parity --backend tensorrt_fp16 --volumes 4   # + whole-volume Dice check (CUDA)
    python -m src.parity --smoke-test --synthetic --volumes 1  # CPU-only end-to-end check

The reference is always the PyTorch checkpoint run in true FP32 (TF32 turned
off). Each candidate backend is compared against it and judged by the
tolerance *tier* of its precision, read from ``parity.tiers`` in the config.

Why tiers instead of one tolerance?
-----------------------------------
Exact bitwise equality is impossible: float addition is not associative, so
two runtimes that sum in a different order disagree in the last bits.
How much they may disagree depends on precision:

* ``fp32`` (ONNX Runtime): logits should match to ~1e-4. Anything bigger means
  the *graph* is different (a wrong op, a dropped layer), so this tier is strict
  and every logit must satisfy ``|ref - out| <= atol + rtol * |ref|``.
* ``fp16`` / ``bf16`` / ``int8``: each logit legitimately moves by 1e-2 or
  more, so requiring every element to pass ``allclose`` would fail every
  healthy engine. These tiers judge mostly on what the model is *for*: the
  predicted class (argmax) and, on real volumes, the Dice score.

The margin check
----------------
The *margin* of a voxel is the gap between its top two reference logits. A
voxel's predicted class can only flip if the numeric error is bigger than its
margin. Flips on low-margin voxels (the model was unsure anyway, usually at the
organ boundary) are expected in reduced precision. A flip on a voxel the model
was *confident* about (margin above ``margin_threshold``) means something is
genuinely broken, so the gate allows none.

This is also why margin-based training losses matter for deployment: a model
trained to keep larger logit margins tolerates more FP16/INT8 noise.

Two levels of evidence
----------------------
1. *Golden inputs*: ``golden_batches`` fixed-seed random batches of exactly the
   exported shape. Cheap, needs no dataset, catches graph-level breakage.
2. *Volumes* (``--volumes N``): whole validation volumes through sliding-window
   inference with each backend. Slower, but measures the real task (Dice vs the
   ground-truth label) and real activation ranges, which random noise does not.

Outputs ``<results_dir>/parity.json`` and ``parity.md``; exits with status 1
if any backend fails a check, so CI can block the change.
"""

from __future__ import annotations

import argparse
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from src.model import load_checkpoint
from src.utils import environment_info, file_sha256, load_config, resolve_path, save_json

Predictor = Callable[[torch.Tensor], torch.Tensor]

# Which tolerance tier judges which backend.
BACKEND_TIER = {
    "onnxruntime": "fp32",
    "tensorrt_fp16": "fp16",
    # torch.autocast: float16 on CUDA, bfloat16 on CPU (the only CPU option).
    "pytorch_autocast": None,  # resolved at run time from the device
}
ALL_BACKENDS = list(BACKEND_TIER)
# pytorch_autocast is opt-in: on CPU it means bfloat16, which many CPUs
# (including Apple Silicon) emulate ~15x slower than float32. It exists to
# exercise a reduced-precision tier without a GPU, not as a deployment target.
DEFAULT_BACKENDS = ["onnxruntime", "tensorrt_fp16"]

# Margin buckets for the flip-rate breakdown in the report.
MARGIN_BUCKETS = [0.0, 0.1, 0.5, 1.0, 2.0, math.inf]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def top2_margin(logits: np.ndarray) -> np.ndarray:
    """Gap between the largest and second-largest logit along the class axis (1)."""
    if logits.shape[1] < 2:
        raise ValueError("margin needs at least two classes")
    part = np.partition(logits, -2, axis=1)
    return part[:, -1] - part[:, -2]


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two boolean masks; 1.0 when both are empty (they agree)."""
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / total)


def compare(ref: np.ndarray, out: np.ndarray, atol: float, rtol: float,
            margin_threshold: float) -> dict[str, Any]:
    """Every number the gate can judge, for one reference/candidate logits pair.

    ``ref`` and ``out`` have shape ``(N, C, *spatial)``. Class 0 is background,
    so "foreground" means argmax > 0.
    """
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs candidate {out.shape}")
    ref = ref.astype(np.float64)
    out = out.astype(np.float64)
    if not np.isfinite(out).all():
        n_bad = int((~np.isfinite(out)).sum())
        return {"finite": False, "non_finite_count": n_bad}

    diff = np.abs(ref - out)
    within = diff <= atol + rtol * np.abs(ref)
    r, o = ref.ravel(), out.ravel()
    denom = np.linalg.norm(r) * np.linalg.norm(o)
    cosine = float(r @ o / denom) if denom > 0 else float(np.array_equal(r, o))

    ref_cls, out_cls = ref.argmax(1), out.argmax(1)
    flipped = ref_cls != out_cls
    margin = top2_margin(ref)
    confident = margin > margin_threshold

    buckets = []
    for lo, hi in zip(MARGIN_BUCKETS[:-1], MARGIN_BUCKETS[1:]):
        sel = (margin >= lo) & (margin < hi)
        n = int(sel.sum())
        buckets.append({"margin_from": lo, "margin_to": hi, "voxels": n,
                        "flips": int((flipped & sel).sum())})

    return {
        "finite": True,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p99_abs_diff": float(np.percentile(diff, 99)),
        "fraction_within_tolerance": float(within.mean()),
        "cosine_similarity": cosine,
        "flip_rate": float(flipped.mean()),
        "confident_voxels": int(confident.sum()),
        "confident_flips": int((flipped & confident).sum()),
        "foreground_dice_between_backends": mask_dice(ref_cls > 0, out_cls > 0),
        "flips_by_margin": buckets,
    }


# -----------------------------------------------------------------------------
# Gate logic
# -----------------------------------------------------------------------------
# metric name -> (tier key holding the threshold, direction)
#   "min": the metric must be >= threshold;  "max": it must be <= threshold.
CHECKS = [
    ("fraction_within_tolerance", "min_fraction_within", "min"),
    ("cosine_similarity", "min_cosine", "min"),
    ("flip_rate", "max_flip_rate", "max"),
    ("confident_flips", "max_confident_flips", "max"),
    ("foreground_dice_between_backends", "min_foreground_dice", "min"),
]
VOLUME_CHECKS = [
    ("dice_drop", "max_dice_drop", "max"),
]


def judge(metrics: dict[str, Any], tier: dict[str, Any], checks=CHECKS) -> list[dict[str, Any]]:
    """Turn metrics into a list of named pass/fail checks.

    A check is only run if the tier defines its threshold, so tiers can opt
    out (e.g. there is no Dice-drop check on random golden inputs).
    """
    if not metrics.get("finite", True):
        return [{"check": "finite_outputs", "value": metrics["non_finite_count"],
                 "threshold": 0, "passed": False}]
    results = []
    for metric, key, direction in checks:
        if key not in tier or metric not in metrics:
            continue
        value, threshold = metrics[metric], tier[key]
        ok = value >= threshold if direction == "min" else value <= threshold
        results.append({"check": metric, "value": value, "threshold": threshold,
                        "direction": direction, "passed": bool(ok)})
    return results


# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------
@contextmanager
def fp32_reference() -> Iterator[None]:
    """Run PyTorch in true FP32: no TF32 in convolutions or matmuls.

    On Ampere and newer GPUs, TF32 (10-bit mantissa) may be used for "fp32"
    maths, which would make the reference itself noisy.
    """
    saved = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


class ORTPredictor:
    """ONNX Runtime session that behaves like a PyTorch model.

    The ONNX file has a fixed batch size B. Sliding-window inference can pass
    a smaller final batch, so we zero-pad to B and slice the result (exactly
    what ``TRTPredictor`` does for TensorRT).
    """

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = tuple(inp.shape)
        self.batch = self.input_shape[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_np = x.detach().float().cpu().numpy()
        if tuple(x_np.shape[1:]) != self.input_shape[1:]:
            raise ValueError(f"ONNX model expects (*, {self.input_shape[1:]}), got {x_np.shape}")
        outs = []
        for i in range(0, x_np.shape[0], self.batch):
            chunk = x_np[i:i + self.batch]
            n = chunk.shape[0]
            if n < self.batch:
                pad = np.zeros((self.batch - n, *chunk.shape[1:]), dtype=chunk.dtype)
                chunk = np.concatenate([chunk, pad])
            outs.append(self.session.run([self.output_name], {self.input_name: chunk})[0][:n])
        return torch.from_numpy(np.concatenate(outs)).to(device)


def autocast_predictor(model: torch.nn.Module, device: torch.device) -> tuple[Predictor, str]:
    """PyTorch under autocast: float16 on CUDA, bfloat16 elsewhere."""
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    tier = "fp16" if dtype == torch.float16 else "bf16"

    def predict(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device.type, dtype=dtype):
            return model(x).float()

    return predict, tier


def make_backend(name: str, model: torch.nn.Module, device: torch.device,
                 cfg: dict[str, Any], rebuild_engine: bool = False):
    """Return ``(predictor, tier, device_for_inputs, info)`` or a skip reason (str)."""
    onnx_path = resolve_path(cfg["export"]["onnx_path"])
    if name == "onnxruntime":
        if not onnx_path.exists():
            return f"{onnx_path} not found - run `python -m src.export_onnx` first"
        return ORTPredictor(onnx_path), "fp32", torch.device("cpu"), {"onnx_path": str(onnx_path)}
    if name == "pytorch_autocast":
        predict, tier = autocast_predictor(model, device)
        return predict, tier, device, {"autocast_dtype": tier}
    if name == "tensorrt_fp16":
        if device.type != "cuda":
            return "needs CUDA"
        from src.trt_utils import TRTPredictor, ensure_engine, tensorrt_available

        if not tensorrt_available():
            return "tensorrt not installed"
        if not onnx_path.exists():
            return f"{onnx_path} not found - run `python -m src.export_onnx` first"
        engine_path = resolve_path(cfg["benchmark"]["engine_path"])
        # ensure_engine rebuilds if the cached engine came from a different
        # ONNX file, TensorRT version or GPU, so the gate never judges a stale engine.
        info = ensure_engine(onnx_path, engine_path, cfg["benchmark"]["workspace_gb"],
                             force=rebuild_engine)
        return TRTPredictor(engine_path), "fp16", device, info
    raise ValueError(f"unknown backend {name!r}; choose from {ALL_BACKENDS}")


# -----------------------------------------------------------------------------
# Running the comparisons
# -----------------------------------------------------------------------------
def golden_inputs(input_shape, n_batches: int, seed: int) -> list[torch.Tensor]:
    """Fixed-seed random batches in [0, 1] (the range after intensity scaling)."""
    gen = torch.Generator().manual_seed(seed)
    return [torch.rand(*input_shape, generator=gen) for _ in range(n_batches)]


@torch.no_grad()
def run_golden(model: torch.nn.Module, predictor: Predictor, inputs: list[torch.Tensor],
               ref_device: torch.device, cand_device: torch.device, tier: dict[str, Any],
               margin_threshold: float) -> dict[str, Any]:
    """Compare reference and candidate on the golden batches, pooled together."""
    refs, outs = [], []
    for x in inputs:
        with fp32_reference():
            refs.append(model(x.to(ref_device)).float().cpu().numpy())
        outs.append(predictor(x.to(cand_device)).float().cpu().numpy())
    metrics = compare(np.concatenate(refs), np.concatenate(outs),
                      tier["atol"], tier["rtol"], margin_threshold)
    return {"metrics": metrics, "checks": judge(metrics, tier)}


@torch.no_grad()
def run_volumes(model: torch.nn.Module, predictor: Predictor, volumes, ref_device: torch.device,
                cand_device: torch.device, tier: dict[str, Any], cfg: dict[str, Any],
                margin_threshold: float) -> dict[str, Any]:
    """Whole-volume comparison through sliding-window inference, plus Dice vs label."""
    from src.evaluation import DiceScorer, predict_volume

    roi = cfg["data"]["roi_size"]
    sw, overlap = cfg["inference"]["sw_batch_size"], cfg["inference"]["overlap"]
    n_classes = cfg["model"]["out_channels"]
    ref_scorer, cand_scorer = DiceScorer(n_classes), DiceScorer(n_classes)
    per_volume, refs, outs = [], [], []
    for image, label in volumes:
        with fp32_reference():
            ref = predict_volume(model, image.to(ref_device), roi, sw, overlap).float().cpu()
        out = predict_volume(predictor, image.to(cand_device), roi, sw, overlap).float().cpu()
        label = label.cpu()
        d_ref, d_out = ref_scorer(ref, label), cand_scorer(out, label)
        per_volume.append({"dice_reference": d_ref, "dice_candidate": d_out})
        refs.append(ref.numpy())
        outs.append(out.numpy())

    # Volumes differ in size, so pool metrics over flattened voxels per class.
    flat = lambda arrs: np.concatenate([a.reshape(a.shape[0], a.shape[1], -1) for a in arrs], axis=2)  # noqa: E731
    metrics = compare(flat(refs), flat(outs), tier["atol"], tier["rtol"], margin_threshold)
    dice_ref, dice_out = ref_scorer.aggregate(), cand_scorer.aggregate()
    metrics.update(dice_reference=dice_ref, dice_candidate=dice_out,
                   dice_drop=dice_ref - dice_out, per_volume=per_volume)
    return {"metrics": metrics, "checks": judge(metrics, tier, CHECKS + VOLUME_CHECKS)}


def to_markdown(report: dict[str, Any]) -> str:
    """Render the gate outcome as a table for a CI summary or PR comment."""
    onnx = report.get("onnx_sha256")
    lines = [f"Parity gate: **{'PASS' if report['passed'] else 'FAIL'}** "
             f"(checkpoint `{report['checkpoint_sha256'][:12]}`"
             f"{f', ONNX `{onnx[:12]}`' if onnx else ''})", "",
             "| Backend | Tier | Level | Check | Value | Threshold | Result |",
             "|---|---|---|---|---:|---:|---|"]
    for b in report["backends"]:
        if b["status"] != "ok":
            lines.append(f"| {b['backend']} | | | skipped: {b['reason']} | | | |")
            continue
        for level in ("golden", "volumes"):
            if level not in b:
                continue
            for c in b[level]["checks"]:
                sign = ">=" if c.get("direction") == "min" else "<="
                lines.append(f"| {b['backend']} | {b['tier']} | {level} | {c['check']} | "
                             f"{c['value']:.6g} | {sign} {c['threshold']:.6g} | "
                             f"{'pass' if c['passed'] else '**FAIL**'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Check deployed backends match the PyTorch FP32 model.")
    p.add_argument("--config", default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--checkpoint", help="default: <train.checkpoint_dir>/best.pt")
    p.add_argument("--backend", action="append", choices=ALL_BACKENDS,
                   help=f"repeatable; default: {' + '.join(DEFAULT_BACKENDS)}")
    p.add_argument("--volumes", type=int, default=0,
                   help="also compare N whole validation volumes (needs the dataset or --synthetic)")
    p.add_argument("--synthetic", action="store_true", help="use generated fake volumes")
    p.add_argument("--device", default="auto")
    p.add_argument("--rebuild-engine", action="store_true")
    args = p.parse_args(argv)

    from src.utils import get_device

    cfg = load_config(args.config, smoke_test=args.smoke_test)
    pcfg = cfg["parity"]
    device = get_device(args.device)
    if device.type == "mps":  # MPS is not a deployment target; keep the reference exact
        device = torch.device("cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_path(cfg["train"]["checkpoint_dir"]) / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device=device)
    cfg["data"]["roi_size"] = ckpt["config"]["data"]["roi_size"]
    input_shape = (cfg["inference"]["sw_batch_size"], ckpt["config"]["model"]["in_channels"],
                   *cfg["data"]["roi_size"])

    inputs = golden_inputs(input_shape, pcfg["golden_batches"], cfg["seed"])
    volumes = []
    if args.volumes > 0:
        from src.data import SYNTHETIC_TASK, build_dataloaders

        if args.synthetic:
            cfg["data"]["task"] = SYNTHETIC_TASK
        _, val_loader = build_dataloaders(cfg)
        for i, b in enumerate(val_loader):
            if i >= args.volumes:
                break
            volumes.append((b["image"], b["label"]))

    backends = []
    for name in args.backend or DEFAULT_BACKENDS:
        made = make_backend(name, model, device, cfg, args.rebuild_engine)
        if isinstance(made, str):
            print(f"{name:17s} skipped: {made}")
            backends.append({"backend": name, "status": "skipped", "reason": made})
            continue
        predictor, tier_name, cand_device, info = made
        tier = pcfg["tiers"][tier_name]
        entry: dict[str, Any] = {"backend": name, "status": "ok", "tier": tier_name, "info": info,
                                 "golden": run_golden(model, predictor, inputs, device, cand_device,
                                                      tier, pcfg["margin_threshold"])}
        if volumes:
            entry["volumes"] = run_volumes(model, predictor, volumes, device, cand_device, tier,
                                           cfg, pcfg["margin_threshold"])
        checks = [c for lvl in ("golden", "volumes") if lvl in entry for c in entry[lvl]["checks"]]
        entry["passed"] = all(c["passed"] for c in checks)
        g = entry["golden"]["metrics"]
        print(f"{name:17s} tier={tier_name:4s} {'PASS' if entry['passed'] else 'FAIL'}  "
              f"max|diff|={g.get('max_abs_diff', float('nan')):.2e}  "
              f"flip={g.get('flip_rate', float('nan')):.2e}  "
              f"confident flips={g.get('confident_flips', 'n/a')}")
        backends.append(entry)

    ran = [b for b in backends if b["status"] == "ok"]
    onnx_path = resolve_path(cfg["export"]["onnx_path"])
    report = {
        "passed": bool(ran) and all(b["passed"] for b in ran),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": file_sha256(ckpt_path),
        # The ONNX file every non-PyTorch backend was built from (None if not exported yet).
        "onnx_sha256": file_sha256(onnx_path) if onnx_path.exists() else None,
        "input_shape": list(input_shape),
        "golden_batches": pcfg["golden_batches"],
        "volumes": len(volumes),
        "margin_threshold": pcfg["margin_threshold"],
        "tiers": pcfg["tiers"],
        "environment": environment_info(),
        "backends": backends,
    }
    out_dir = resolve_path(cfg["results_dir"])
    save_json(report, out_dir / "parity.json")
    md = to_markdown(report)
    (out_dir / "parity.md").write_text(md)
    print("\n" + md)
    print(f"Wrote {out_dir / 'parity.json'} and {out_dir / 'parity.md'}")
    if not ran:
        raise SystemExit("Parity gate: no backend could run, nothing was verified")
    if not report["passed"]:
        raise SystemExit("Parity gate FAILED: see the table above")


if __name__ == "__main__":
    main()
