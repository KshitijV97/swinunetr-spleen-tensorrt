# SwinUNETR Spleen Segmentation → ONNX → TensorRT

[![tests](https://github.com/kshitijv97/swinunetr-spleen-tensorrt/actions/workflows/tests.yml/badge.svg)](https://github.com/kshitijv97/swinunetr-spleen-tensorrt/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kshitijv97/swinunetr-spleen-tensorrt/blob/main/notebooks/colab_train_and_benchmark.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, reproducible pipeline for **3D spleen segmentation in
contrast-enhanced CT**: a SwinUNETR trained from scratch with MONAI on MSD
Task09, exported to ONNX with a numerical parity check, compiled to a
**TensorRT FP16** engine, and benchmarked against PyTorch FP32 and AMP on an
NVIDIA T4 for **per-volume latency and Dice**.

The scope is deliberately narrow: one cohort, one architecture, one deployment
path. The effort goes into a correct pipeline and a measurement protocol that
says exactly what each number means.

> **Status:** implementation, tests, CPU CI and the local smoke test are
> complete. The full T4 training and benchmark have **not been run yet**.
> Every result cell below reads *TBD* until they come from
> [the Colab notebook](notebooks/colab_train_and_benchmark.ipynb). Nothing here
> is estimated.

**Documentation comes in two tracks:**

| | For | Start at |
|---|---|---|
| **Technical reference** | Readers fluent in DL, medical imaging and GPU inference who want the design decisions, exact settings and caveats | [docs/reference/](docs/reference/README.md) |
| **Learning track** | Engineers new to the field: CT physics, signals, ML, transformers, training and GPU deployment, from first principles | [docs/learn/](docs/learn/README.md) |

---

## Motivation

Pre-operative organ segmentation feeds surgical planning, registration and
navigation, and virtual-fixture constraints in robot-assisted surgery. The
spleen is a relevant at-risk organ: highly vascular, and a recognised site of
iatrogenic injury in left-upper-quadrant and colorectal procedures (for
example, splenic flexure mobilisation). Clinical use needs two things: an
accurate 3D shape, and inference fast enough on the target GPU for the
workflow it serves. Hence the second half of this project: TensorRT, which is
the same toolchain used on Jetson-class embedded GPUs.

## Method

| Stage | Choice | Details |
|---|---|---|
| Data | MSD Task09 Spleen, 41 labelled portal-venous CTs; fixed seeded split **33 train / 8 val** | [data.md](docs/reference/data.md) |
| Preprocessing | RAS reorientation; resample to 1.5 × 1.5 × 2.0 mm (trilinear / nearest); HU window [−57, 164] → [0, 1]; foreground crop | [data.md §3](docs/reference/data.md#3-deterministic-pipeline-train-and-validation) |
| Sampling / augmentation | 4 × 96³ crops per case, 1:1 spleen/background centres; flips, in-plane 90° rotations, intensity shift | [data.md §4](docs/reference/data.md#4-stochastic-pipeline-training-only) |
| Model | SwinUNETR, $F = 48$, depths (2,2,2,2), heads (3,6,12,24), window 7: **62.2 M params**, trained from scratch | [architecture.md](docs/reference/architecture.md) |
| Objective / optimiser | Dice + CE (MONAI `DiceCELoss`), AdamW 1e-4 / wd 1e-5, cosine to 0 over 150 epochs, FP16 AMP with dynamic loss scaling | [training.md](docs/reference/training.md) |
| Inference | Sliding window 96³, overlap 0.5, Gaussian blending ($\sigma = 0.125 \cdot 96$), batch 4, argmax, no post-processing | [inference-and-evaluation.md](docs/reference/inference-and-evaluation.md) |
| Metric | Foreground Dice per volume, mean over 8 validation cases, in resampled space | [inference-and-evaluation.md §3](docs/reference/inference-and-evaluation.md#3-dice-as-computed-here) |
| Export | ONNX opset 18, static $(4,1,96,96,96)$, verified against PyTorch with ONNX Runtime (`allclose`, atol = rtol = 1e-3) | [deployment.md §2](docs/reference/deployment.md#2-onnx-export) |
| Engine | TensorRT 10, FP16 builder flag, per-layer precision by the builder; zero-copy integration on PyTorch's CUDA stream | [deployment.md §3–4](docs/reference/deployment.md#3-tensorrt-engine-build) |
| Benchmark | Synchronised wall-clock latency of whole-volume sliding-window inference (GPU-resident input), 2 warm-up + 8 vols × 3 repeats; mean / p50 / p95; Dice per backend | [benchmarking.md](docs/reference/benchmarking.md) |

```
MSD Spleen ─▶ MONAI preprocessing ─▶ SwinUNETR training ─▶ best.pt ─┬─▶ PyTorch FP32 / AMP ──┐
                                                                    └─▶ ONNX (+ parity) ─▶ TensorRT FP16 ─┴─▶ latency + Dice
```

## Results

> ⚠️ **TBD: run notebook.** Values come from
> `notebooks/colab_train_and_benchmark.ipynb` on a Colab T4. The field-to-cell
> mapping is in [docs/author-notes/reporting-results.md](docs/author-notes/reporting-results.md).

### Training

| | Value |
|---|---|
| Train / validation volumes | 33 / 8 |
| Model | SwinUNETR, $F = 48$, 62.2 M parameters |
| Epochs trained | TBD: run notebook |
| Training time (T4) | TBD: run notebook |
| Best validation Dice (AMP, during training) | TBD: run notebook |

<!-- After the Colab run, uncomment: ![Training curve](results/training_curve.png) -->

### Inference (NVIDIA T4, 8 validation volumes × 3 repeats)

Latency = whole-volume sliding-window inference (96³, batch 4, overlap 0.5),
input already on the GPU, after warm-up, CUDA-synchronised. Preprocessing,
I/O and post-processing are excluded.

| Backend | Mean latency / volume (ms) | p95 (ms) | Speed-up vs FP32 | Mean Dice |
|---|---:|---:|---:|---:|
| PyTorch FP32 | TBD: run notebook | TBD: run notebook | 1.00× | TBD: run notebook |
| PyTorch AMP (FP16 autocast) | TBD: run notebook | TBD: run notebook | TBD: run notebook | TBD: run notebook |
| TensorRT FP16 | TBD: run notebook | TBD: run notebook | TBD: run notebook | TBD: run notebook |

### Export

| | Value |
|---|---|
| ONNX vs PyTorch max abs. logit difference | TBD: run notebook |
| ONNX vs PyTorch argmax agreement | TBD: run notebook |
| TensorRT engine build time / size | TBD: run notebook |
| Software (driver / CUDA / PyTorch / TensorRT) | TBD: run notebook |

## Limitations

- **Small, single-centre cohort.** 41 portal-venous CTs from one institution
  and one patient population (liver metastases under chemotherapy). No claim
  of external validity across scanners, phases or pathology.
- **Validation, not test.** One fixed 33/8 split. The checkpoint is selected
  on the same 8 cases it is reported on (optimistic bias). $n = 8$ gives wide
  uncertainty, and per-case Dice is in `benchmark.json`.
- **Resampled-space Dice, no surface metric.** Dice is computed on the
  1.5 × 1.5 × 2 mm cropped grid, not natively. No HD95/NSD is reported.
- **One GPU, model-only latency.** Shared, virtualised Colab T4 with that
  day's software stack. Preprocessing and I/O are excluded, and speed-ups
  will differ on other hardware and TensorRT versions.
- **Static engine shape.** $(4, 1, 96^3)$ only, which is sufficient for
  sliding-window inference but not a dynamic-shape deployment.
- **Not a medical device.** Research and education only. Not for clinical
  decisions.

## Reproduce

**Local (CPU/MPS, no NVIDIA GPU).** This is a pipeline check with a tiny
model ($F = 12$), 2 volumes, 64³ crops and 2 epochs. The outputs go to the
git-ignored `results/smoke/`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.train --smoke-test --synthetic      # drop --synthetic to use 2 real volumes (1.5 GB download)
python -m src.export_onnx --smoke-test
python -m src.benchmark --smoke-test --synthetic  # AMP / TensorRT recorded as "skipped: needs CUDA"
```

**Full run (Colab T4).** Open the notebook with the badge above, select
*Runtime → Change runtime type → T4 GPU*, then *Run all*. It installs pinned
dependencies and TensorRT, tests, downloads, trains, exports, benchmarks and
packages `results.zip`. See [colab.md](docs/reference/colab.md). On any
CUDA Linux host:

```bash
pip install -r requirements-colab.txt    # plus a CUDA build of torch
python -m src.data --download && python -m src.train && python -m src.export_onnx && python -m src.benchmark
```

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml),
and common ones can be overridden on the CLI (`--epochs`, `--feature-size`,
`--lr`, ...). The merged config and environment are embedded in every
checkpoint and result file.

## Repository layout

```
configs/default.yaml        hyperparameters (+ smoke-test overrides)
src/                        data, model, evaluation, train, export_onnx, trt_utils, benchmark, utils
tests/                      pytest on synthetic data (CPU, ~30 s)
notebooks/                  Colab T4 end-to-end run
docs/reference/             technical reference (expert audience)
docs/learn/                 learning track (first principles)
docs/author-notes/          how to transcribe and phrase results
results/                    committed results after the T4 run
```

Data, checkpoints, ONNX files and engines are never committed. Details are in
[code-and-reproducibility.md](docs/reference/code-and-reproducibility.md).

## Extensions

5-fold CV, native-space Dice (`Invertd`) and HD95/NSD. Self-supervised
pre-trained Swin weights (Tang et al., 2022) and a 3D U-Net / nnU-Net
baseline. INT8 with calibration, FP32-pinned normalisation layers, dynamic
batch profiles and CUDA graphs. GPU-side preprocessing with end-to-end
latency. Jetson Orin deployment and a ROS 2 node. Multi-organ (BTCV, AMOS,
TotalSegmentator).

## References

- Hatamizadeh, A. et al. *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.* BrainLes 2021. arXiv:2201.01266.
- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- Tang, Y. et al. *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis.* CVPR 2022.
- Antonelli, M. et al. *The Medical Segmentation Decathlon.* Nat. Commun. 13, 4128 (2022). Data: <http://medicaldecathlon.com> (CC-BY-SA 4.0).
- [MONAI](https://monai.io) (model, transforms, spleen tutorial defaults), [ONNX](https://onnx.ai), [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).

Code is released under the [MIT License](LICENSE). The dataset is not
included and keeps its own license.
