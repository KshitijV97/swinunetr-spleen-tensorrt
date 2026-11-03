"""Dataset download, train/validation split and MONAI preprocessing pipelines.

Dataset
-------
Medical Segmentation Decathlon (MSD) Task09 Spleen: 41 labelled abdominal CT
volumes (plus 20 unlabelled test volumes we do not use). Each label volume
marks every voxel as 0 = background or 1 = spleen. docs/reference/data.md
documents each transform; docs/learn/01-the-medical-problem.md and
02-images-as-signals.md explain CT, Hounsfield units and resampling from scratch.

Usage
-----
    python -m src.data --download          # fetch + extract (~1.5 GB) into data/
    python -m src.data --summary           # print the split that training will use
    python -m src.data --synthetic 4       # write 4 fake volumes (no download needed)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from monai import transforms as T
from monai.data import CacheDataset, DataLoader, Dataset

from src.utils import load_config, resolve_path

# Name of the fake dataset written by `make_synthetic_dataset` (set
# `data.task` to this, or pass `--synthetic` to train.py, to use it).
SYNTHETIC_TASK = "Synthetic_Spleen"


# -----------------------------------------------------------------------------
# Download and file lists
# -----------------------------------------------------------------------------
def download_spleen(root_dir: str | Path, task: str = "Task09_Spleen") -> Path:
    """Download and extract the MSD task into ``root_dir`` if it is not there yet.

    The URL (an S3 mirror hosted by the MONAI team) and its MD5 checksum are
    read from MONAI itself so they stay in sync with the library.
    Returns the task directory, e.g. ``data/Task09_Spleen``. The tar file is
    verified against MONAI's published MD5 checksum before extraction.
    """
    from monai.apps import DecathlonDataset, download_and_extract

    root_dir = resolve_path(root_dir)
    task_dir = root_dir / task
    if (task_dir / "dataset.json").exists():
        return task_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract(
        url=DecathlonDataset.resource[task],
        filepath=root_dir / f"{task}.tar",
        output_dir=root_dir,
        hash_val=DecathlonDataset.md5[task],
        hash_type="md5",  # MONAI ships an MD5; newer MONAI defaults to sha256
    )
    return task_dir


def load_datalist(task_dir: str | Path) -> list[dict[str, str]]:
    """Read ``dataset.json`` and return ``[{"image": path, "label": path}, ...]``.

    We use the JSON file rather than globbing ``imagesTr/`` because the MSD
    tarballs contain hidden macOS ``._*.nii.gz`` files that are not images.
    """
    task_dir = Path(task_dir)
    with open(task_dir / "dataset.json") as f:
        meta = json.load(f)
    return [
        {
            "image": str((task_dir / item["image"]).resolve()),
            "label": str((task_dir / item["label"]).resolve()),
        }
        for item in meta["training"]
    ]


def split_datalist(
    items: list[dict[str, str]],
    val_fraction: float,
    seed: int,
    max_volumes: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deterministically shuffle and split cases into train / validation lists.

    The split depends only on ``seed`` (not on file-system order), so the same
    8 volumes are held out every time, on every machine. At least one volume
    always ends up on each side.
    """
    items = sorted(items, key=lambda d: d["image"])
    random.Random(seed).shuffle(items)
    if max_volumes:
        items = items[:max_volumes]
    if len(items) < 2:
        raise ValueError(f"Need at least 2 volumes to split, got {len(items)}")
    n_val = min(max(1, round(len(items) * val_fraction)), len(items) - 1)
    return items[n_val:], items[:n_val]


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def _common_transforms(data_cfg: dict[str, Any]) -> list:
    """Deterministic preprocessing shared by training and validation.

    Order matters:
      1. load the NIfTI files and put the channel axis first -> (1, X, Y, Z)
      2. reorient to RAS so "left/right/up" mean the same thing for every scan
      3. resample to a fixed voxel spacing so 1 voxel = the same physical size
      4. clip CT intensities to a soft-tissue window and rescale to [0, 1]
      5. crop away the air around the patient to save memory and compute
    """
    a_min, a_max = data_cfg["hu_window"]
    return [
        T.LoadImaged(keys=["image", "label"]),
        T.EnsureChannelFirstd(keys=["image", "label"]),
        T.Orientationd(keys=["image", "label"], axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        T.Spacingd(
            keys=["image", "label"],
            pixdim=data_cfg["pixdim"],
            # Images are interpolated smoothly; labels must stay integers, so
            # they use nearest-neighbour interpolation.
            mode=("bilinear", "nearest"),
        ),
        T.ScaleIntensityRanged(
            keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True
        ),
        T.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    ]


def get_train_transforms(data_cfg: dict[str, Any]) -> T.Compose:
    """Preprocessing + random patch sampling + light augmentation for training.

    Full CT volumes do not fit through a 3D transformer on a 16 GB GPU, so we
    train on random ``roi_size`` crops. ``RandCropByPosNegLabeld`` picks crop
    centres half on spleen and half on background; without it most crops would
    contain no spleen at all (it occupies well under 1% of the volume).
    """
    roi = data_cfg["roi_size"]
    pos, neg = data_cfg["pos_neg_ratio"]
    return T.Compose(
        _common_transforms(data_cfg)
        + [
            # Guarantee the volume is at least one crop big (small scans / synthetic data).
            T.SpatialPadd(keys=["image", "label"], spatial_size=roi),
            T.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi,
                pos=pos,
                neg=neg,
                num_samples=data_cfg["num_samples"],
                image_key="image",
                image_threshold=0,
            ),
            T.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            T.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            T.EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def get_val_transforms(data_cfg: dict[str, Any]) -> T.Compose:
    """Deterministic preprocessing only. Validation sees whole volumes, which are
    segmented with sliding-window inference (see ``src/train.py``)."""
    return T.Compose(
        _common_transforms(data_cfg) + [T.EnsureTyped(keys=["image", "label"], track_meta=False)]
    )


# -----------------------------------------------------------------------------
# Synthetic data (for tests and download-free smoke tests)
# -----------------------------------------------------------------------------
def make_synthetic_dataset(
    root_dir: str | Path,
    n_volumes: int = 2,
    shape: tuple[int, int, int] = (112, 112, 80),
    seed: int = 0,
) -> Path:
    """Write tiny fake CT volumes in the same folder layout as the real MSD task.

    Each volume is an ellipsoidal "body" (~40 HU) surrounded by air (-1000 HU)
    with a brighter ellipsoidal "spleen" (~100 HU) inside, plus noise. It is
    obviously not anatomy - it only exists so the pipeline can be exercised
    without a 1.5 GB download.
    """
    import nibabel as nib

    rng = np.random.default_rng(seed)
    task_dir = resolve_path(root_dir) / SYNTHETIC_TASK
    (task_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (task_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    centre = np.array(shape) / 2
    affine = np.diag([1.5, 1.5, 2.0, 1.0])  # already at the target spacing
    entries = []
    for i in range(n_volumes):
        body = (
            ((zz - centre[0]) / (shape[0] * 0.45)) ** 2
            + ((yy - centre[1]) / (shape[1] * 0.45)) ** 2
            + ((xx - centre[2]) / (shape[2] * 0.45)) ** 2
        ) <= 1
        offset = rng.uniform(-8, 8, size=3)
        radii = rng.uniform(8, 14, size=3)
        spleen = (
            ((zz - centre[0] - offset[0]) / radii[0]) ** 2
            + ((yy - centre[1] - offset[1]) / radii[1]) ** 2
            + ((xx - centre[2] - offset[2]) / radii[2]) ** 2
        ) <= 1
        image = np.full(shape, -1000.0, dtype=np.float32)
        image[body] = 40.0
        image[spleen] = 100.0
        image += rng.normal(0, 15, size=shape).astype(np.float32)
        label = spleen.astype(np.uint8)

        name = f"synthetic_{i:02d}.nii.gz"
        nib.save(nib.Nifti1Image(image, affine), task_dir / "imagesTr" / name)
        nib.save(nib.Nifti1Image(label, affine), task_dir / "labelsTr" / name)
        entries.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})

    with open(task_dir / "dataset.json", "w") as f:
        json.dump({"name": SYNTHETIC_TASK, "training": entries}, f, indent=2)
    return task_dir


# -----------------------------------------------------------------------------
# Putting it together
# -----------------------------------------------------------------------------
def get_task_dir(cfg: dict[str, Any], download: bool = True) -> Path:
    """Return the dataset folder, downloading (or synthesising) it on first use."""
    data_cfg = cfg["data"]
    if data_cfg["task"] == SYNTHETIC_TASK:
        task_dir = resolve_path(data_cfg["root_dir"]) / SYNTHETIC_TASK
        if not (task_dir / "dataset.json").exists():
            make_synthetic_dataset(data_cfg["root_dir"], n_volumes=max(2, data_cfg["max_volumes"] or 2))
        return task_dir
    task_dir = resolve_path(data_cfg["root_dir"]) / data_cfg["task"]
    if not (task_dir / "dataset.json").exists():
        if not download:
            raise FileNotFoundError(f"{task_dir} not found. Run `python -m src.data --download`.")
        task_dir = download_spleen(data_cfg["root_dir"], data_cfg["task"])
    return task_dir


def get_splits(cfg: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """File lists for training and validation, as defined by the config."""
    data_cfg = cfg["data"]
    items = load_datalist(get_task_dir(cfg))
    return split_datalist(items, data_cfg["val_fraction"], cfg["seed"], data_cfg["max_volumes"])


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Create the training and validation DataLoaders.

    ``CacheDataset`` runs the deterministic part of the pipeline once and keeps
    the result in RAM, so every epoch only pays for the random crops. This is
    the single biggest speed-up for 3D medical training.
    """
    data_cfg = cfg["data"]
    train_files, val_files = get_splits(cfg)
    workers = data_cfg["num_workers"]

    def make_ds(files, transform):
        if data_cfg["cache_rate"] > 0:
            return CacheDataset(files, transform, cache_rate=data_cfg["cache_rate"], num_workers=workers)
        return Dataset(files, transform)

    train_ds = make_ds(train_files, get_train_transforms(data_cfg))
    val_ds = make_ds(val_files, get_val_transforms(data_cfg))
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=workers
    )
    # Validation volumes all have different shapes, so they must go one at a time.
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=workers)
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / inspect the MSD Spleen dataset.")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--download", action="store_true", help="download and extract the dataset")
    parser.add_argument("--summary", action="store_true", help="print the train/val split")
    parser.add_argument("--synthetic", type=int, metavar="N", help="write N synthetic volumes instead")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        path = make_synthetic_dataset(cfg["data"]["root_dir"], n_volumes=args.synthetic)
        print(f"Wrote {args.synthetic} synthetic volumes to {path}")
        return
    if args.download:
        path = download_spleen(cfg["data"]["root_dir"], cfg["data"]["task"])
        print(f"Dataset ready at {path}")
    if args.summary or not args.download:
        train_files, val_files = get_splits(cfg)
        print(f"train: {len(train_files)} volumes, val: {len(val_files)} volumes")
        for item in val_files:
            print("  val:", Path(item["image"]).name)


if __name__ == "__main__":
    main()
