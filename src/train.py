"""Train SwinUNETR on MSD Task09 Spleen.

Examples
--------
    # Tiny end-to-end check on a laptop (CPU or Apple GPU); ~13 min on a MacBook Air:
    python -m src.train --smoke-test

    # Same, without downloading the real dataset (uses fake volumes):
    python -m src.train --smoke-test --synthetic

    # Full training (meant for a CUDA GPU such as Colab's T4):
    python -m src.train --config configs/default.yaml

What one epoch does
-------------------
1. For every training volume, take ``num_samples`` random 96^3 crops
   (half centred on spleen, half on background) and augment them.
2. Forward pass -> DiceCE loss -> backward pass -> AdamW step.
   On CUDA this runs in mixed precision (fp16 matmuls, fp32 master weights).
3. Every ``val_interval`` epochs, segment each *whole* validation volume with
   sliding-window inference and compute the mean spleen Dice.
4. If that Dice is the best so far, save ``best.pt``.

Outputs: ``<checkpoint_dir>/best.pt``, ``<checkpoint_dir>/last.pt`` and
``<results_dir>/training_history.json`` (loss + Dice per epoch).
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from typing import Any

import torch
from monai.losses import DiceCELoss

from src.data import SYNTHETIC_TASK, build_dataloaders
from src.evaluation import DiceScorer, predict_volume
from src.model import build_model, count_parameters, validate_roi
from src.utils import environment_info, get_device, load_config, resolve_path, save_json, set_seed


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
    if args.no_amp:
        cfg["train"]["amp"] = False
    return cfg


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


def train(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Run the full training loop and return the training history."""
    set_seed(cfg["seed"])
    validate_roi(cfg["data"]["roi_size"])
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

    ckpt_dir = resolve_path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: dict[str, Any] = {"epoch": [], "train_loss": [], "val_epoch": [], "val_dice": []}
    best_dice, best_epoch = -1.0, -1
    start = time.time()

    for epoch in range(1, epochs + 1):
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
        print(msg, f"  [{time.time() - start:.0f}s]", flush=True)

    torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epochs}, ckpt_dir / "last.pt")
    history.update(
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
    train(cfg, get_device(args.device))


if __name__ == "__main__":
    main()
