"""Small helpers shared by every script: config loading, device choice, seeding.

Keeping these in one place means `train.py`, `export_onnx.py` and
`benchmark.py` all interpret the YAML config and the `--smoke-test` flag in
exactly the same way.
"""

from __future__ import annotations

import copy
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Repository root (the folder that contains `src/`). Relative paths in the
# config are resolved against it, so scripts work no matter which directory
# you launch them from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a copy of ``base``.

    Nested dictionaries are merged key by key instead of being replaced, so a
    smoke-test override of ``data.roi_size`` leaves ``data.pixdim`` untouched.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None, smoke_test: bool = False) -> dict[str, Any]:
    """Load the YAML config and, if requested, apply its ``smoke_test`` block.

    The ``smoke_test`` section is always removed from the returned dict so the
    rest of the code never has to think about it.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        YAML file to read. Defaults to ``configs/default.yaml`` in the repo.
    smoke_test : bool, default False
        If True, merge the file's ``smoke_test`` block over the normal values
        (see :func:`deep_update`), giving a small, fast configuration for
        quick end-to-end checks. If False, those overrides are discarded.

    Returns
    -------
    dict
        The configuration. It has no ``smoke_test`` section; instead it has a
        ``smoke_test`` key holding the boolean flag, so callers can still tell
        which mode they are in.

    Notes
    -----
    Analogy: the config file is a recipe with a "mini batch" note at the
    bottom. Normally you cook the full recipe and ignore the note; with
    ``smoke_test=True`` you apply the note's substitutions first. Either way
    the note is torn off, and a sticker records which version you cooked.
    """
    # Use the caller's file, or fall back to the repo's default config.
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Remove the smoke_test block (it may be missing or empty) so it never
    # leaks into the returned config.
    smoke_overrides = cfg.pop("smoke_test", {}) or {}
    if smoke_test:
        # Merge key by key: only the listed values change, the rest are kept.
        cfg = deep_update(cfg, smoke_overrides)
    # Record which mode was used (this replaces the removed block).
    cfg["smoke_test"] = smoke_test
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Turn a config path (usually relative) into an absolute path under the repo."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def get_device(preference: str = "auto") -> torch.device:
    """Pick the compute device.

    ``auto`` prefers an NVIDIA GPU (CUDA), then an Apple Silicon GPU (MPS),
    then the CPU. Pass ``cpu``, ``mps`` or ``cuda`` to force one.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch and MONAI so runs are repeatable.

    Note: GPU kernels are not all deterministic, so two CUDA runs can still
    differ slightly in the last decimal places.
    """
    from monai.utils import set_determinism

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_determinism(seed=seed)


def environment_info() -> dict[str, Any]:
    """Collect versions and hardware names, saved next to every result file.

    Benchmarks are meaningless without knowing what they ran on.
    """
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import monai

        info["monai"] = monai.__version__
    except ImportError:  # pragma: no cover - monai is a hard dependency
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as pretty-printed JSON, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
