"""Small helpers shared by every script: config loading, device choice, seeding.

Keeping these in one place means `train.py`, `export_onnx.py` and
`benchmark.py` all interpret the YAML config and the `--smoke-test` flag in
exactly the same way.
"""

from __future__ import annotations

import copy
import hashlib
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

    Args:
        base: The original configuration. It is never modified.
        overrides: Values to layer on top of ``base``. Keys missing from
            ``base`` are added; keys present in both are overwritten, except
            that when both values are dicts they are merged recursively.

    Returns:
        A new dictionary containing the merged result. Neither input is
        mutated, and the result shares no nested objects with them.

    Example:
        >>> base = {"data": {"roi_size": [96, 96, 96], "pixdim": [1.5, 1.5, 2.0]},
        ...         "epochs": 150}
        >>> overrides = {"data": {"roi_size": [32, 32, 32]}, "epochs": 1}
        >>> deep_update(base, overrides)
        {'data': {'roi_size': [32, 32, 32], 'pixdim': [1.5, 1.5, 2.0]}, 'epochs': 1}

        A plain ``{**base, **overrides}`` would instead replace the whole
        ``data`` dict and drop ``pixdim``.
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
    """Turn a config path (usually relative) into an absolute path under the repo.

    Relative paths are anchored at the repository root (``REPO_ROOT``), not the
    current working directory, so scripts behave the same wherever they are
    launched from. A leading ``~`` is expanded to the home directory, and
    absolute paths are returned unchanged.

    Parameters
    ----------
    path : str or pathlib.Path
        A path such as ``"checkpoints/best.pt"`` from the config or CLI.

    Returns
    -------
    pathlib.Path
        An absolute path. The file does not need to exist, and nothing is
        created.

    Examples
    --------
    With the repo at ``/home/me/swinunetr-spleen-tensorrt``::

        >>> resolve_path("checkpoints/best.pt")
        PosixPath('/home/me/swinunetr-spleen-tensorrt/checkpoints/best.pt')
        >>> resolve_path("/data/spleen")      # absolute: unchanged
        PosixPath('/data/spleen')
        >>> resolve_path("~/spleen")          # ~ expanded to the home directory
        PosixPath('/home/me/spleen')

    Analogy: a street address with no city is assumed to be in *this* town
    (the repo), while a full address (absolute) is used as written.
    """
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

    Analogy: shuffling several decks of cards. Each deck (Python, NumPy, PyTorch, MONAI) has its own shuffler. To get the identical shuffle every time, you reset all of them to the same starting arrangement. A deck that was left out would shuffle differently.
    """
    # Imported here so importing utils does not require MONAI up front.
    from monai.utils import set_determinism

    # Each library keeps its own random-number generator, so each needs a seed.
    random.seed(seed)           # Python's built-in `random`
    np.random.seed(seed)        # NumPy (e.g. data shuffling)
    torch.manual_seed(seed)     # PyTorch CPU and CUDA (e.g. weight init, dropout)
    # MONAI: seeds its own transforms and also sets PyTorch/cuDNN to deterministic
    # mode (slower, but repeatable).
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


def file_sha256(path: str | Path, chunk: int = 2**20) -> str:
    """Hex SHA-256 of a file's bytes, read in 1 MiB chunks.

    Used as a fingerprint: two files with the same hash have identical
    contents, so a report or a cached engine can name exactly which
    checkpoint or ONNX file it came from.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as pretty-printed JSON, creating parent folders as needed.

    Parameters
    ----------
    obj : Any
        The data to save. It must be JSON-serializable (dicts, lists, strings,
        numbers, booleans, ``None``). NumPy scalars and arrays are not, and
        raise ``TypeError``; convert them first, e.g. with ``float(...)``.
    path : str or pathlib.Path
        Destination file. An existing file at this path is overwritten.

    Returns
    -------
    pathlib.Path
        ``path`` as a ``Path``, so callers can print or reuse it.

    Notes
    -----
    The file is written directly, not via a temporary file, so a crash
    mid-write can leave a partial file.

    Analogy: a clerk who builds the missing filing-cabinet drawer, then writes
    your notes on a fresh, neatly indented sheet, replacing any old sheet with
    the same name.
    """
    # Accept str or Path; work with Path from here on.
    path = Path(path)
    # Create the destination folder (and parents) if missing; no error if it
    # already exists, so repeated runs are safe.
    path.parent.mkdir(parents=True, exist_ok=True)
    # "w" opens for writing and overwrites any existing file.
    with open(path, "w") as f:
        # indent=2 makes the output human-readable instead of one long line.
        json.dump(obj, f, indent=2)
    return path
