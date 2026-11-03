# The data: MSD Task09 Spleen and how it is preprocessed

Code: [`src/data.py`](../src/data.py). Config: the `data:` block of
[`configs/default.yaml`](../configs/default.yaml).

---

## 1. The dataset

The **Medical Segmentation Decathlon** (MSD) is a set of ten public 3D
segmentation tasks released to benchmark general-purpose methods
(Antonelli et al., *Nature Communications* 2022,
<https://www.nature.com/articles/s41467-022-30695-9>; website
<http://medicaldecathlon.com>).

**Task09 Spleen**:

| Property | Value |
|---|---|
| Modality | Contrast-enhanced abdominal CT (portal-venous phase) |
| Source | Memorial Sloan Kettering Cancer Center; patients undergoing chemotherapy for liver metastases |
| Labelled volumes | 41 (`imagesTr/` + `labelsTr/`) |
| Unlabelled test volumes | 20 (`imagesTs/`; not used here because their labels are not public) |
| Labels | 0 = background, 1 = spleen |
| Format | NIfTI (`.nii.gz`), one file per volume |
| Download size | ~1.5 GB tar |
| License | CC-BY-SA 4.0 (see the MSD website) |

The data is **not** in this repository (`data/` is in `.gitignore`).
`python -m src.data --download` fetches it from the MONAI team's S3 mirror and
checks its MD5 checksum before extracting it into `data/Task09_Spleen/`.

### Train / validation split

`split_datalist` sorts the 41 cases by filename, shuffles them with a fixed
seed (`seed: 42`) and holds out `val_fraction = 0.2`, giving **33 training / 8
validation** volumes. The same seed gives the same split on every machine.
Print it with `python -m src.data --summary`.

There is no separate test set. With only 41 labelled volumes, every Dice
score this project reports is a **validation** score, and the README's
limitations section says so. A stronger protocol would be 5-fold
cross-validation (see "Ideas for extension" in the README).

## 2. A 60-second primer on CT

- A CT volume is a 3D grid of **voxels**. Each voxel stores X-ray attenuation
  in **Hounsfield units (HU)**: air ≈ −1000, fat ≈ −100, water = 0, soft
  tissue ≈ +20 to +80, contrast-enhanced organs ≈ +100 to +200, bone up to
  +1000 and beyond.
- Voxels are usually **not cubes**: in-plane spacing is often 0.6–0.9 mm,
  while slices can be 1.5–8 mm apart. Spacing differs from scan to scan.
- Each file carries an **affine matrix** that maps voxel indices to physical
  millimetres and tells you which way is left, anterior, superior and so on.

## 3. The preprocessing pipeline, transform by transform

These steps are shared by training and validation (`_common_transforms`):

| # | Transform | What it does | Why |
|---|---|---|---|
| 1 | `LoadImaged` | Reads the NIfTI image and label, keeping the affine as metadata | — |
| 2 | `EnsureChannelFirstd` | `(X, Y, Z)` → `(1, X, Y, Z)` | PyTorch convention: channels first |
| 3 | `Orientationd("RAS")` | Flips and permutes the axes so +x points to the patient's **R**ight, +y to **A**nterior, +z to **S**uperior (NiBabel's "RAS+" convention) | Every scan ends up with the same anatomical axis directions, so the model never has to learn mirrored or rotated anatomy caused only by how the file was saved |
| 4 | `Spacingd(1.5, 1.5, 2.0 mm)` | Resamples to a fixed physical voxel size: **bilinear** interpolation for the image, **nearest-neighbour** for the label | A 96³ crop then always covers the same physical region (144 × 144 × 192 mm), and labels stay 0/1 instead of being blurred into fractions |
| 5 | `ScaleIntensityRanged(-57…164 → 0…1, clip)` | Clips HU to [−57, 164], then maps that range linearly to [0, 1] | Spends the input range on soft tissue and contrast-enhanced organs. Air and bone carry little information about the spleen |
| 6 | `CropForegroundd` | Crops to the bounding box of voxels > 0 after windowing | Removes the air around the patient: less memory, faster training |

The following run only during training (`get_train_transforms`), after the steps above:

| Transform | What it does | Why |
|---|---|---|
| `SpatialPadd(roi_size)` | Pads volumes that are smaller than one crop | Guarantees every crop is exactly `roi_size` |
| `RandCropByPosNegLabeld(pos=1, neg=1, num_samples=4)` | Takes 4 random 96³ crops per volume, half centred on a spleen voxel and half on a background voxel | The spleen is a small fraction of the scan, so uniform random crops would mostly miss it. Balanced sampling gives the model enough positive examples |
| `RandFlipd` on each axis (p = 0.2 each) | Mirrors the crop | Cheap augmentation that reduces over-fitting on 33 volumes |
| `RandRotate90d` (p = 0.2) | Rotates by 90°, 180° or 270° in a plane | Same idea |
| `RandShiftIntensityd(±0.1, p = 0.5)` | Adds a random constant to all intensities | Robustness to scanner and contrast-timing differences |

Validation (`get_val_transforms`) uses only the deterministic steps and keeps
the **whole** volume, which is then segmented with sliding-window inference
(see [TRAINING.md](TRAINING.md#sliding-window-inference)).

### Where the numbers come from

- The **HU window −57…164** and **spacing 1.5 × 1.5 × 2.0 mm** are the values
  used in MONAI's official spleen segmentation tutorial. They are sensible
  defaults, not tuned for this repository.
- **96³ crops** are SwinUNETR's standard input size, and the size must be
  divisible by 32 (see [ARCHITECTURE.md](ARCHITECTURE.md)).
- **4 crops per volume** keep one optimisation step at 4 samples while only
  loading one volume, which fits in a T4's 16 GB with mixed precision.

## 4. Caching: why training is not I/O bound

Loading, reorienting and resampling a CT volume takes on the order of seconds,
far longer than one training step. `CacheDataset` runs the deterministic
steps **once** and keeps the results in RAM. Each epoch then only pays for the
random crops and augmentations. `data.cache_rate` controls how much is cached
(1.0 = everything). If Colab runs out of system RAM, lower it or cap
`data.max_volumes`.

## 5. Where Dice is measured (an honest caveat)

Validation Dice is computed **after** preprocessing: on the reoriented,
resampled (1.5 × 1.5 × 2.0 mm) and foreground-cropped grid. It is not computed
on each scan's original voxel grid. For a fair comparison with papers that
report Dice in native space, you would invert the spacing and orientation
transforms (`monai.transforms.Invertd`) before scoring. The difference is
usually small for a large organ like the spleen, but it is not zero. This
applies equally to every backend in the benchmark, so the PyTorch vs TensorRT
comparison is still apples to apples.

## 6. Synthetic data (for tests and download-free smoke tests)

`make_synthetic_dataset` writes fake volumes in the same folder layout as MSD.
Each one is an ellipsoidal "body" at ~40 HU in air, with a brighter
ellipsoidal "spleen" at ~100 HU inside, plus Gaussian noise. It is
**not anatomy**. It exists so that:

- `pytest` runs in seconds without the 1.5 GB download;
- `python -m src.train --smoke-test --synthetic` can check the whole pipeline
  on a fresh laptop.

Never report numbers obtained on synthetic data.
