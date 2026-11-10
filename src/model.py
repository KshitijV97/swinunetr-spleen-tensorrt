"""SwinUNETR model construction.

SwinUNETR (Hatamizadeh et al., 2022, https://arxiv.org/abs/2201.01266) pairs a
Swin Transformer encoder with a convolutional U-Net-style decoder:

    input (1, 96, 96, 96)
      -> patch embedding: 2x2x2 patches -> 48^3 tokens of width `feature_size`
      -> 4 Swin stages, each halving resolution and doubling width
         (windowed self-attention, alternating with shifted windows so
          information flows between neighbouring windows)
      -> CNN decoder with skip connections from every encoder stage
      -> 1x1x1 conv to `out_channels` logits per voxel  (2, 96, 96, 96)

docs/reference/architecture.md documents this in detail; docs/learn/07-swin-and-swinunetr.md
explains it from first principles.

Run ``python -m src.model`` to print the parameter count for the default config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from monai.networks.nets import SwinUNETR

# 5 levels of 2x downsampling (patch embedding + 4 patch-merging stages)
# means the input size must be divisible by 2**5.
INPUT_DIVISOR = 32


def validate_roi(roi_size: Sequence[int]) -> None:
    """Raise a clear error if a crop size would break SwinUNETR's downsampling."""
    # Analogy: like a 
    #  checking every guest before the club opens. The
    # network halves the volume 5 times, so each side must split evenly; a side
    # like 100 would end up as a fractional voxel. Better to turn it away at the
    # door (here) than fail with a cryptic shape error deep inside the model.
    # `s % 32` is 0 (falsy) when OK, non-zero (truthy) when a side is bad.
    bad = [s for s in roi_size if s % INPUT_DIVISOR]
    if bad:
        raise ValueError(
            f"roi_size {list(roi_size)} must be divisible by {INPUT_DIVISOR} in every dimension"
        )


def build_model(model_cfg: dict[str, Any]) -> SwinUNETR:
    """Create a SwinUNETR from the ``model:`` block of the config.

    ``feature_size`` is the main capacity knob: 48 is the size used in the
    paper (~62M parameters); 24 is ~4x cheaper; 12 is only for smoke tests.
    It must be divisible by 12 because the first stage has 3 attention heads
    and the width is later split across 6/12/24 heads.
    """
    feature_size = model_cfg["feature_size"]
    if feature_size % 12:
        raise ValueError(f"feature_size must be divisible by 12, got {feature_size}")
    return SwinUNETR(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        feature_size=feature_size,
        use_checkpoint=model_cfg.get("use_checkpoint", False),
        dropout_path_rate=model_cfg.get("dropout_path_rate", 0.0),
        spatial_dims=3,
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[SwinUNETR, dict[str, Any]]:
    """Rebuild a model from a checkpoint written by ``src/train.py``.

    The checkpoint stores the config it was trained with, so the model is
    always rebuilt with the right ``feature_size`` even if the YAML file has
    changed since.

    Parameters
    ----------
    path : str or pathlib.Path
        Checkpoint file, e.g. ``checkpoints/best.pt``.
    device : torch.device or str, default "cpu"
        Where to place the weights, e.g. ``"cuda"``. The returned model lives
        on this device.

    Returns
    -------
    model : SwinUNETR
        The network with trained weights loaded, in eval mode.
    ckpt : dict
        The full checkpoint dict: ``model_state`` (weights), ``config`` (the
        training config), ``epoch`` and, for ``best.pt``, ``val_dice``.
        Callers use ``ckpt["config"]`` to read the shapes used in training.

    Raises
    ------
    SystemExit
        If ``path`` does not exist, with a message telling the user to train
        first. (This ends the program rather than raising a catchable error.)

    Notes
    -----
    Technically, a checkpoint holds only numbers (the ``state_dict``: a map from
    layer-parameter names to tensors), not the code of the network. So loading
    has two parts: build an empty network of the right architecture, then copy
    the saved numbers into it. ``load_state_dict`` is strict by default, so it
    errors if the architecture does not match the saved names and shapes.

    ``weights_only=False`` lets ``torch.load`` unpickle arbitrary Python
    objects, which is needed because the checkpoint contains the config dict.
    Only load checkpoints you trust, since unpickling can execute code.

    Analogy: the checkpoint is the instruction sheet and the saved settings of
    a machine. You first build a blank machine from the sheet (architecture),
    then dial in the saved settings (weights), and switch it to "run" mode.
    """
    # Fail early with an actionable message instead of a raw FileNotFoundError.
    if not Path(path).exists():
        raise SystemExit(
            f"No checkpoint at {path}. Training has not saved a model yet: run "
            "`python -m src.train` first and check its output for errors. best.pt is "
            "written at the first validation (epoch `train.val_interval`)."
        )
    # Read the file. map_location puts tensors straight on `device`, so a model
    # trained on GPU still loads on a CPU-only machine.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Rebuild the architecture from the config saved *inside* the checkpoint
    # (not the current YAML), so the layer shapes match the saved weights.
    model = build_model(ckpt["config"]["model"])
    # Copy the trained weights into the freshly built (randomly initialised) model.
    model.load_state_dict(ckpt["model_state"])
    # Ensure the model is on `device`, and switch to eval mode: this disables
    # dropout / stochastic depth so outputs are deterministic for inference/export.
    model.to(device).eval()
    return model, ckpt


if __name__ == "__main__":
    from src.utils import load_config

    cfg = load_config()
    net = build_model(cfg["model"])
    print(f"SwinUNETR feature_size={cfg['model']['feature_size']}: {count_parameters(net):,} parameters")
