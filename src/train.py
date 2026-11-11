"""Train SwinUNETR on MSD Task09 Spleen.

Examples
--------
    # Tiny end-to-end check on a laptop (CPU or Apple GPU); ~13 min on a MacBook Air:
    python -m src.train --smoke-test

    # Same, without downloading the real dataset (uses fake volumes):
    python -m src.train --smoke-test --synthetic

    # Full training (meant for a CUDA GPU such as Colab's T4):
    python -m src.train --config configs/default.yaml

    # Continue an interrupted run from <checkpoint_dir>/last.pt (same flags as before):
    python -m src.train --epochs 20 --resume

What one epoch does
-------------------
1. For every training volume, take ``num_samples`` random 96^3 crops
   (half centred on spleen, half on background) and augment them.
2. Forward pass -> DiceCE loss -> backward pass -> AdamW step.
   On CUDA this runs in mixed precision (fp16 matmuls, fp32 master weights).
3. Every ``val_interval`` epochs, segment each *whole* validation volume with
   sliding-window inference and compute the mean spleen Dice.
4. If that Dice is the best so far, save ``best.pt``.
5. Save ``last.pt``: the full training state (weights, optimizer, LR schedule,
   AMP scaler, random-number generators, history so far). It is overwritten
   every epoch, so an interruption loses at most the epoch in progress.

Resuming
--------
``--resume`` reloads ``last.pt`` and continues from the next epoch. The config
must match the interrupted run (same epochs, model, data, learning rate, ...):
the learning-rate schedule and the train/val split depend on it, so a mismatch
stops with a list of the differing keys. Only settings that do not change the
maths may differ (paths, worker count, cache rate, gradient checkpointing).
A resumed run is statistically equivalent to an uninterrupted one, not
bit-identical: MONAI's per-transform augmentation generators restart.

Outputs: ``<checkpoint_dir>/best.pt``, ``<checkpoint_dir>/last.pt`` and
``<results_dir>/training_history.json`` (loss + Dice per epoch).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.losses import DiceCELoss

from src.data import SYNTHETIC_TASK, build_dataloaders
from src.evaluation import DiceScorer, predict_volume
from src.model import build_model, count_parameters, validate_roi
from src.utils import (
    environment_info,
    get_device,
    load_config,
    resolve_path,
    save_json,
    set_seed,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--smoke-test", action="store_true", help="tiny model, 2 volumes, 2 epochs")
    p.add_argument("--synthetic", action="store_true", help="use generated fake volumes (no download)")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    # Optional overrides of individual config values:
    p.add_argument("--data-dir", help="override data.root_dir")
    p.add_argument("--max-volumes", type=int, help="override data.max_volumes")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--lr", type=float, help="override train.lr")
    p.add_argument("--batch-size", type=int, help="override train.batch_size")
    p.add_argument("--val-interval", type=int, help="override train.val_interval")
    p.add_argument("--feature-size", type=int, help="override model.feature_size")
    p.add_argument("--checkpoint-dir", help="override train.checkpoint_dir")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--use-checkpoint", action="store_true",
                   help="gradient checkpointing: much less GPU memory, ~30%% slower")
    p.add_argument("--resume", nargs="?", const="auto", default=None, metavar="PATH",
                   help="continue an interrupted run from PATH "
                        "(default: <checkpoint_dir>/last.pt); pass the same flags as the original run")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load the YAML config and apply command-line overrides on top."""
    cfg = load_config(args.config, smoke_test=args.smoke_test)
    if args.synthetic:
        cfg["data"]["task"] = SYNTHETIC_TASK
    overrides = {
        ("data", "root_dir"): args.data_dir,
        ("data", "max_volumes"): args.max_volumes,
        ("train", "epochs"): args.epochs,
        ("train", "lr"): args.lr,
        ("train", "batch_size"): args.batch_size,
        ("train", "val_interval"): args.val_interval,
        ("train", "checkpoint_dir"): args.checkpoint_dir,
        ("model", "feature_size"): args.feature_size,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            cfg[section][key] = value
    if args.use_checkpoint:
        cfg["model"]["use_checkpoint"] = True
    if args.no_amp:
        cfg["train"]["amp"] = False
    return cfg


# Config keys that may differ between an interrupted run and its resumption:
# they change where files live or how fast/memory-hungry training is, never the
# numbers being computed. Everything else under these sections must match.
RESUME_COMPARED_SECTIONS = ("seed", "data", "model", "train")
RESUME_ALLOWED_DIFFERENCES = {
    ("data", "root_dir"),
    ("data", "num_workers"),
    ("data", "cache_rate"),
    ("model", "use_checkpoint"),
    ("train", "checkpoint_dir"),
}


def config_differences(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """List ``section.key: saved -> current`` for every resume-relevant mismatch."""
    diffs = []
    for section in RESUME_COMPARED_SECTIONS:
        old, new = saved.get(section), current.get(section)
        if not isinstance(old, dict) or not isinstance(new, dict):
            if old != new:
                diffs.append(f"{section}: {old!r} -> {new!r}")
            continue
        for key in sorted(set(old) | set(new)):
            if (section, key) in RESUME_ALLOWED_DIFFERENCES:
                continue
            if old.get(key) != new.get(key):
                diffs.append(f"{section}.{key}: {old.get(key)!r} -> {new.get(key)!r}")
    return diffs


def rng_state() -> dict[str, Any]:
    """Snapshot every global random-number generator training uses."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore generators saved by :func:`rng_state` (CUDA only if available)."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(obj: Any, path: Path) -> None:
    """``torch.save`` to a temporary file, then rename it over ``path``.

    A crash or disconnect mid-write leaves the previous ``path`` intact instead
    of a truncated file that would make ``--resume`` fail.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_resume_state(path: Path, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Load ``last.pt`` for resuming and check it belongs to this run.

    Raises ``SystemExit`` with an actionable message when the file is missing,
    is a weights-only checkpoint from an older version, or was trained with a
    different config.
    """
    if not path.exists():
        raise SystemExit(
            f"--resume: no checkpoint at {path}. last.pt is written at the end of every epoch, "
            "so nothing was saved yet (or --checkpoint-dir points somewhere else). "
            "Start without --resume."
        )
    # weights_only=False: the state holds the config dict and NumPy RNG state.
    # Only resume from checkpoints you produced yourself (unpickling can run code).
    state = torch.load(path, map_location=device, weights_only=False)
    if "optimizer_state" not in state:
        raise SystemExit(
            f"--resume: {path} holds weights only (written by a version without resume support), "
            "so the optimizer and schedule cannot be restored. Start without --resume."
        )
    diffs = config_differences(state["config"], cfg)
    if diffs:
        raise SystemExit(
            f"--resume: the config differs from the run that wrote {path}:\n  "
            + "\n  ".join(diffs)
            + "\nRe-run with the same flags as the original run (e.g. the same --epochs)."
        )
    return state


@torch.no_grad()
def validate(model, loader, cfg, device, autocast) -> float:
    """Mean spleen Dice over the validation set using sliding-window inference."""
    model.eval()
    scorer = DiceScorer(cfg["model"]["out_channels"])
    for batch in loader:
        image, label = batch["image"].to(device), batch["label"].to(device)
        with autocast():
            logits = predict_volume(
                model, image, cfg["data"]["roi_size"],
                cfg["inference"]["sw_batch_size"], cfg["inference"]["overlap"],
            )
        scorer(logits.float(), label)
    return scorer.aggregate()


def train(cfg: dict[str, Any], device: torch.device, resume: str | Path | None = None) -> dict[str, Any]:
    """Run the full training loop and return the training history.

    ``resume`` is a path to a ``last.pt`` written by an earlier, interrupted
    call with the same config; ``"auto"`` means ``<checkpoint_dir>/last.pt``.
    Training then continues at the epoch after the one saved.
    """
    set_seed(cfg["seed"])
    validate_roi(cfg["data"]["roi_size"])
    ckpt_dir = resolve_path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Check the resume file before caching the dataset (minutes on Colab), so a
    # wrong path or flag fails immediately.
    state, resume_path = None, None
    if resume is not None:
        resume_path = ckpt_dir / "last.pt" if resume == "auto" else Path(resume)
        state = load_resume_state(resume_path, cfg, device)
    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model(cfg["model"]).to(device)
    print(f"Model: SwinUNETR feature_size={cfg['model']['feature_size']}, "
          f"{count_parameters(model):,} parameters, device={device}")
    print(f"Data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val volumes")

    # DiceCE = Dice loss (directly optimises overlap, robust to the tiny size of
    # the spleen) + cross-entropy (smooth, well-behaved gradients early on).
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    epochs = cfg["train"]["epochs"]
    # Cosine decay: large steps early, fine adjustments at the end.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision only on CUDA: it is where the speed-up (Tensor Cores) is.
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else nullcontext

    history: dict[str, Any] = {"epoch": [], "train_loss": [], "val_epoch": [], "val_dice": []}
    best_dice, best_epoch = -1.0, -1
    start_epoch, prior_seconds, resumed_from = 1, 0.0, []

    if state is not None:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        best_dice, best_epoch = state["best_dice"], state["best_epoch"]
        prior_seconds = state["elapsed_seconds"]
        resumed_from = state.get("resumed_from", []) + [state["epoch"]]
        start_epoch = state["epoch"] + 1
        set_rng_state(state["rng_state"])
        best_msg = f"best val Dice {best_dice:.4f} at epoch {best_epoch}" if best_epoch > 0 else "no validation yet"
        print(f"Resuming from {resume_path}: epoch {state['epoch']}/{epochs} done, {best_msg}", flush=True)
        if start_epoch > epochs:
            print("That run already finished all its epochs; writing its history and stopping.")

    start = time.time() - prior_seconds  # elapsed time keeps counting across resumes

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            image, label = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = loss_fn(model(image), label)
            # GradScaler multiplies the loss so small fp16 gradients don't round
            # to zero, then un-scales before the optimizer step. No-op without AMP.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            steps += 1
        scheduler.step()
        epoch_loss /= max(steps, 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        msg = f"epoch {epoch:4d}/{epochs}  loss {epoch_loss:.4f}  lr {scheduler.get_last_lr()[0]:.2e}"

        if epoch % cfg["train"]["val_interval"] == 0 or epoch == epochs:
            dice = validate(model, val_loader, cfg, device, autocast)
            history["val_epoch"].append(epoch)
            history["val_dice"].append(dice)
            msg += f"  val_dice {dice:.4f}"
            if dice > best_dice:
                best_dice, best_epoch = dice, epoch
                torch.save(
                    {"model_state": model.state_dict(), "config": cfg, "epoch": epoch, "val_dice": dice},
                    ckpt_dir / "best.pt",
                )
                msg += "  (new best, saved)"

        # Everything needed to continue from here, overwritten every epoch.
        # It also keeps "model_state" + "config", so load_checkpoint() can read it.
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": rng_state(),
                "config": cfg,
                "epoch": epoch,
                "history": history,
                "best_dice": best_dice,
                "best_epoch": best_epoch,
                "elapsed_seconds": time.time() - start,
                "resumed_from": resumed_from,
            },
            ckpt_dir / "last.pt",
        )
        print(msg, f"  [{time.time() - start:.0f}s]", flush=True)

    history.update(
        resumed_from_epochs=resumed_from,
        best_val_dice=best_dice,
        best_epoch=best_epoch,
        train_minutes=(time.time() - start) / 60,
        n_train=len(train_loader.dataset),
        n_val=len(val_loader.dataset),
        n_parameters=count_parameters(model),
        config=cfg,
        environment=environment_info(),
    )
    out = save_json(history, resolve_path(cfg["results_dir"]) / "training_history.json")
    print(f"Best val Dice {best_dice:.4f} at epoch {best_epoch}. "
          f"Checkpoint: {ckpt_dir / 'best.pt'}  History: {out}")
    return history


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    try:
        train(cfg, get_device(args.device), resume=args.resume)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "\nCUDA out of memory: SwinUNETR (feature_size="
            f"{cfg['model']['feature_size']}) with {cfg['data']['num_samples']} crops of "
            f"{cfg['data']['roi_size']} per step does not fit on this GPU.\n"
            "Re-run with --use-checkpoint (same model, less memory, slower). If it still "
            "does not fit, add --feature-size 24 and report that in the README."
        )


if __name__ == "__main__":
    main()
