"""Tests for dataset splitting and the MONAI preprocessing pipelines."""

import pytest
import torch

from src.data import get_train_transforms, get_val_transforms, split_datalist


def test_split_is_deterministic_and_disjoint():
    items = [{"image": f"img_{i}.nii.gz", "label": f"lbl_{i}.nii.gz"} for i in range(41)]
    train_a, val_a = split_datalist(items, val_fraction=0.2, seed=42)
    train_b, val_b = split_datalist(list(reversed(items)), val_fraction=0.2, seed=42)
    assert val_a == val_b, "split must not depend on input order"
    assert len(train_a) == 33 and len(val_a) == 8
    assert not {d["image"] for d in train_a} & {d["image"] for d in val_a}


def test_split_respects_max_volumes():
    items = [{"image": f"img_{i}", "label": f"lbl_{i}"} for i in range(10)]
    train, val = split_datalist(items, val_fraction=0.5, seed=0, max_volumes=2)
    assert len(train) == 1 and len(val) == 1


def test_split_rejects_single_volume():
    with pytest.raises(ValueError):
        split_datalist([{"image": "a", "label": "b"}], val_fraction=0.5, seed=0)


def test_train_transforms_output(smoke_cfg, synthetic_items):
    data_cfg = smoke_cfg["data"]
    samples = get_train_transforms(data_cfg)(synthetic_items[0])
    # RandCropByPosNegLabeld returns a list of `num_samples` crops.
    assert isinstance(samples, list) and len(samples) == data_cfg["num_samples"]
    for s in samples:
        assert s["image"].shape == (1, *data_cfg["roi_size"])
        assert s["label"].shape == (1, *data_cfg["roi_size"])
        # Intensities are windowed and scaled into [0, 1] (+ small random shift).
        assert s["image"].min() >= -0.11 and s["image"].max() <= 1.11
        assert set(torch.unique(s["label"]).tolist()) <= {0.0, 1.0}


def test_val_transforms_keep_label_aligned(smoke_cfg, synthetic_items):
    out = get_val_transforms(smoke_cfg["data"])(synthetic_items[0])
    assert out["image"].shape == out["label"].shape
    assert out["image"].shape[0] == 1
    assert 0.0 <= float(out["image"].min()) and float(out["image"].max()) <= 1.0
    assert out["label"].sum() > 0, "foreground cropping must not remove the spleen"
