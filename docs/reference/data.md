# Data and preprocessing

Code: [`src/data.py`](../../src/data.py). Config: `data:` in
[`configs/default.yaml`](../../configs/default.yaml).

---

## 1. Cohort

Medical Segmentation Decathlon, Task09 Spleen (Antonelli et al., *Nat.
Commun.* 13, 4128, 2022; Simpson et al., arXiv:1902.09063).

| Property | Value |
|---|---|
| Modality / phase | Contrast-enhanced abdominal CT, portal-venous phase |
| Source population | Memorial Sloan Kettering; patients receiving chemotherapy for liver metastases |
| Labelled cases | 41 (`imagesTr/`, `labelsTr/`) |
| Unlabelled cases | 20 (`imagesTs/`, labels withheld, unused) |
| Classes | 0 background, 1 spleen |
| Format | NIfTI-1, one volume per file, affine in header |
| Slice thickness | Heterogeneous across cases (in-plane sub-mm, through-plane up to several mm) |
| License | CC-BY-SA 4.0 |

`python -m src.data --download` fetches the tarball from the MONAI S3 mirror
and verifies it against the MD5 that MONAI ships (`DecathlonDataset.md5`).
The file list comes from `dataset.json`, not a glob, because the archive
contains AppleDouble `._*.nii.gz` files.

## 2. Split

`split_datalist`: sort by path, `random.Random(42).shuffle`, take the first
$\mathrm{round}(0.2 \cdot 41) = 8$ as validation. The result is
independent of filesystem order (unit-tested) and identical across machines.
**33 train / 8 validation, no test set.** All reported Dice values are
validation scores on the same 8 cases used for checkpoint selection (see
[inference-and-evaluation.md §4](inference-and-evaluation.md#4-threats-to-validity)).

## 3. Deterministic pipeline (train and validation)

| # | Transform | Parameters | Rationale |
|---|---|---|---|
| 1 | `LoadImaged` | image + label | Keeps the affine as `MetaTensor` metadata |
| 2 | `EnsureChannelFirstd` | | $(X,Y,Z) \to (1,X,Y,Z)$ |
| 3 | `Orientationd` | `RAS`, explicit `labels=(("L","R"),("P","A"),("I","S"))` | Canonical anatomical axes. The explicit labels pin NiBabel's RAS+ convention and silence MONAI's deprecation path |
| 4 | `Spacingd` | 1.5 × 1.5 × 2.0 mm; `bilinear` image, `nearest` label | Fixed physical field of view per crop: 96³ voxels = 144 × 144 × 192 mm. Nearest keeps labels binary |
| 5 | `ScaleIntensityRanged` | $[-57, 164]$ HU $\to [0,1]$, clipped | Soft-tissue window centred on enhanced parenchyma |
| 6 | `CropForegroundd` | source = image, `select_fn` = $x > 0$, `allow_smaller=True` | Bounding box of voxels above −57 HU: removes air and table margin |

Steps 1–6 run once per case and are cached (§5). Spacing and window follow
MONAI's spleen tutorial and were not tuned on this split.

Notes:

- Step 6 runs after windowing, so "foreground" means HU > −57. Lung bases
  and bowel gas inside the body survive (the box is the body's bounding box,
  not a body mask).
- Through-plane resampling to 2.0 mm upsamples thick-slice cases. The
  bilinear interpolation (trilinear in 3D) introduces no new information. It
  only equalises the grid.

## 4. Stochastic pipeline (training only)

| Transform | Parameters | Rationale |
|---|---|---|
| `SpatialPadd` | to `roi_size` | Guarantees at least one full crop (small or synthetic volumes) |
| `RandCropByPosNegLabeld` | 96³, `pos=1, neg=1`, `num_samples=4`, `image_threshold=0` | Each case yields 4 crops. Centre voxels are drawn uniformly from label = 1 (pos) or from label = 0 ∧ image > 0 (neg), each with probability 1/2 per crop. The spleen is well under 1 % of the cropped volume, so uniform sampling would be dominated by negatives |
| `RandFlipd` | each axis independently, $p = 0.2$ | |
| `RandRotate90d` | $p = 0.2$, $k \in \{1,2,3\}$, axes (0, 1) | In-plane only (MONAI default `spatial_axes=(0, 1)`) |
| `RandShiftIntensityd` | offset $\sim U(-0.1, 0.1)$, $p = 0.5$ | Global intensity shift in normalised units (≈ ±22 HU) for contrast-timing/scanner variation |

The flips and 90° rotations break left-right anatomical priors: a flipped
crop puts the spleen on the patient's right. That is standard in the MONAI
recipe and acts as regularisation on 33 cases. It also means the model cannot
rely on absolute laterality, which is arguably desirable for patch-based
training but worth knowing.

Batching: `batch_size = 1` case, and MONAI's `list_data_collate` flattens the 4
crops, so the effective minibatch is $4 \times 1 \times 96^3$.

## 5. Caching

`CacheDataset(cache_rate=1.0)` stores the output of the deterministic prefix
(up to and including `CropForegroundd`) in RAM, so each epoch runs only
the stochastic suffix. Validation volumes are cached whole. Lower
`cache_rate` or set `max_volumes` if host RAM is limited (Colab).

## 6. Validation inputs

Validation uses steps 1–6 only and keeps the whole foreground-cropped volume,
which goes to sliding-window inference with `batch_size = 1` (shapes differ
between cases). Dice is therefore computed **in the resampled, cropped
space** and not on the native grid. See
[inference-and-evaluation.md](inference-and-evaluation.md#3-dice-as-computed-here).

## 7. Synthetic fixtures

`make_synthetic_dataset` writes MSD-layout NIfTI files: an ellipsoidal body
(40 HU) in air (−1000 HU), a randomly placed and sized ellipsoidal "spleen"
(100 HU), additive Gaussian noise ($\sigma = 15$ HU), affine already at
1.5 × 1.5 × 2.0 mm. They exist for unit tests and the CPU smoke test in CI.
They carry no anatomical signal, and numbers obtained on them are never
reported.
