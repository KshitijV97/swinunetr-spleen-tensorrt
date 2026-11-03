"""Render the "what does this project do?" figure used at the top of the README.

It shows one real CT scan from MSD Task09 Spleen next to the same scan with
the spleen highlighted. The highlight is the **radiologist's label** (the
answer the model learns from), not a model prediction.

Usage (needs the dataset in data/ and matplotlib, which is not a project
dependency):

    pip install matplotlib
    python -m scripts.make_overview_figure

The image is derived from the Medical Segmentation Decathlon (CC-BY-SA 4.0).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from src.utils import REPO_ROOT  # noqa: E402

CASE = "spleen_12"  # a training case with a clearly visible spleen
OUT = REPO_ROOT / "docs" / "images" / "ct_spleen_example.png"


def load_ras(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Load a NIfTI file reoriented to RAS (+x = patient's right ... see docs)."""
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj), img.header.get_zooms()


def axial(volume: np.ndarray, z: int) -> np.ndarray:
    """Axial slice in radiological display convention: patient's front at the
    top, patient's right on the viewer's left (as if facing the patient)."""
    return volume[:, :, z].T[::-1, ::-1]


def coronal(volume: np.ndarray, y: int) -> np.ndarray:
    """Coronal slice (front view): head at the top, patient's right on the left."""
    return volume[:, y, :].T[::-1, ::-1]


def main() -> None:
    task = REPO_ROOT / "data" / "Task09_Spleen"
    image, zooms = load_ras(task / "imagesTr" / f"{CASE}.nii.gz")
    label, _ = load_ras(task / "labelsTr" / f"{CASE}.nii.gz")
    window = dict(cmap="gray", vmin=-160, vmax=240)  # soft-tissue display window (HU)

    z = int(label.sum(axis=(0, 1)).argmax())  # slice with the most spleen
    y = int(label.sum(axis=(0, 2)).argmax())
    aspect_cor = zooms[2] / zooms[0]  # slices are thicker than pixels are wide

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    panels = [
        (axial(image, z), None, 1.0, "1. Input: one slice of a CT scan\n(a horizontal cross-section of the belly)"),
        (axial(image, z), axial(label, z), 1.0, "2. Goal: mark every pixel of the spleen\n(outline drawn by a radiologist)"),
        (coronal(image, y), coronal(label, y), aspect_cor, "3. Same patient, front view: the spleen\nis a 3D shape spread over many slices"),
    ]
    for ax, (img, lab, aspect, title) in zip(axes, panels):
        ax.imshow(img, aspect=aspect, **window)
        if lab is not None:
            ax.imshow(np.ma.masked_where(lab == 0, lab), cmap="autumn", alpha=0.55, aspect=aspect)
            ax.contour(lab, levels=[0.5], colors="orange", linewidths=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axes[:2]:
        ax.text(0.02, 0.97, "front", color="w", fontsize=9, transform=ax.transAxes, va="top")
        ax.text(0.02, 0.03, "patient's right", color="w", fontsize=9, transform=ax.transAxes)
        ax.text(0.98, 0.03, "patient's left", color="w", fontsize=9, transform=ax.transAxes, ha="right")
    axes[2].text(0.02, 0.97, "head", color="w", fontsize=9, transform=axes[2].transAxes, va="top")
    fig.text(0.5, 0.01, f"Real scan ({CASE}) from the Medical Segmentation Decathlon, CC-BY-SA 4.0. "
             "Orange = ground-truth label, not a model prediction.", ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
