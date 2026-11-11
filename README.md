# Finding the spleen in 3D CT scans, fast

**SwinUNETR spleen segmentation → ONNX → TensorRT FP16, benchmarked on an NVIDIA T4**

[![tests](https://github.com/kshitijv97/swinunetr-spleen-tensorrt/actions/workflows/tests.yml/badge.svg)](https://github.com/kshitijv97/swinunetr-spleen-tensorrt/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kshitijv97/swinunetr-spleen-tensorrt/blob/main/notebooks/colab_train_and_benchmark.ipynb)
[![License: All rights reserved](https://img.shields.io/badge/License-All%20rights%20reserved-red.svg)](LICENSE)

> **New here? Start with the plain-English section below.** It assumes no
> background in medicine, engineering or AI. Technical readers can jump
> straight to [For technical readers](#for-technical-readers).

---

## What this project does, in plain English

A **CT scan** is a 3D X-ray picture of the inside of the body. It is a stack of
grey cross-sectional images, like slices of a loaf of bread. This project
teaches a computer to look at a CT scan of the belly and **colour in every
part of it that is the spleen** (a fist-sized organ under the left ribs). It
then makes that computer program run **much faster** on the kind of graphics
chip used in medical and robotic equipment, and measures carefully whether
the speed-up cost any accuracy.

![A real CT slice, the same slice with the spleen outlined, and a front view showing the spleen's 3D shape](docs/images/ct_spleen_example.png)

*Above: a real scan from the public dataset used here. The orange area was
outlined by a radiologist. That is the "answer key" the computer learns from
(this is not the computer's own output). The computer has to produce outlines
like this by itself, on every slice, for scans it has never seen.*

### Why would anyone want this?

- **Surgical robots and image-guided surgery need a map.** Before and during
  an operation, software has to know where the organs are, especially
  delicate ones a surgeon must not cut. The spleen is one of them: it is
  full of blood, and accidental injury to it is a known risk in some
  abdominal operations.
- **Drawing outlines by hand is slow.** One scan has dozens to hundreds of
  slices. A computer that does it automatically saves expert time and makes
  3D organ models possible.
- **Speed matters.** A program that takes a long time per scan is fine for
  planning an operation the next day, but not for anything interactive. So
  making it fast is half of this project.

### What happens, step by step

| Step | What it means in everyday terms | Where it happens |
|---|---|---|
| **1. Get examples** | Download 41 real, anonymised CT scans in which experts have already outlined the spleen. They come from a public research dataset. | `src/data.py` |
| **2. Tidy them up** | Make every scan comparable: same orientation, same pixel size, same brightness range. This is like straightening and resizing photos before comparing them. | `src/data.py` |
| **3. Teach the computer** | Show the computer 33 of the scans over and over. Each time, it guesses the outline, compares it with the expert's, and nudges its internal settings to do better. This is called **training**. | `src/train.py` |
| **4. Test it fairly** | Check it on the other 8 scans, which it never learned from, like an exam with questions you have not seen before. | `src/train.py`, `src/evaluation.py` |
| **5. Make it portable** | Save the trained program in a standard format (**ONNX**) that other tools can read, and double-check the saved copy gives the same answers as the original. | `src/export_onnx.py` |
| **6. Make it fast** | Hand it to NVIDIA's optimiser (**TensorRT**), which rewrites it specifically for one graphics chip, removing wasted steps and doing arithmetic with fewer digits (**FP16**) where that is safe. | `src/trt_utils.py` |
| **7. Measure honestly** | Time the original and optimised versions on the same 8 test scans, and check the outlines are still just as accurate. | `src/benchmark.py` |

### How do we know if it works?

The project answers two questions, each with one number:

1. **How accurate are the outlines?** This is measured with the **Dice
   score**: how much the computer's outline overlaps the expert's. **1.0
   means a perfect match, 0 means no overlap at all.** Think of two
   overlapping stickers: the more they line up, the closer to 1.
2. **How fast is it?** This is measured as **latency**: how many seconds the
   computer needs to outline the spleen in one whole scan. Lower is better.
   The **speed-up** is how many times faster the optimised version is than
   the original.

The goal is for the optimised version to be clearly faster **with
essentially the same Dice score**. If it got faster but less accurate, the
results will say so.

> **Current status, plainly:** all the code is written, tested and checked
> on a laptop, and the speed measurements have been run on a Colab T4. The
> speed results are solid. The **accuracy is not**: the model was trained for
> only 20 of the planned 150 epochs, so its Dice (0.587) is far below what a
> fully trained model should reach. See the Results section below. No number
> in this repository is guessed or made up.

### What this project is *not*

- **Not a medical device.** It has not been tested on patients and must not
  be used to make any medical decision.
- **Not proof it works everywhere.** It learns from only 41 scans from one
  hospital. Scans from other hospitals, machines or patients could behave
  differently.
- **Not a new AI model.** The model design (SwinUNETR) comes from published
  research and the open-source MONAI library. This project's work is the
  complete, tested pipeline around it: preparing the data, training,
  verified export, GPU optimisation and honest measurement.

### Jargon decoder

| You will see | It means |
|---|---|
| **CT scan** | A 3D X-ray image made of many thin cross-sectional slices. Brighter means denser tissue. |
| **Segmentation** | Labelling every pixel (here, every tiny 3D cube, or *voxel*) as "spleen" or "not spleen". |
| **SwinUNETR** | The type of AI model used. It looks at the scan in small 3D blocks and learns how the blocks relate to each other, then draws a detailed outline. |
| **Training** | Letting the model learn from examples with known answers. |
| **Validation scans** | Scans held back from training and used only to test it. |
| **Dice score** | The overlap score between computer and expert outlines, from 0 (none) to 1 (perfect). |
| **GPU** | A graphics chip. It is very good at the huge number of simple calculations AI needs. |
| **NVIDIA T4** | The specific GPU used for the measurements. Google Colab provides it for free. |
| **PyTorch / MONAI** | The open-source software used to build and train the model (MONAI is PyTorch for medical images). |
| **ONNX** | A standard file format for AI models, like PDF for documents. Many tools can open it. |
| **TensorRT** | NVIDIA software that rewrites a model to run as fast as possible on one specific NVIDIA GPU. |
| **FP32 / FP16** | Doing arithmetic with about 7 significant digits (FP32) or about 3 (FP16). FP16 is faster and usually just as good for AI. |
| **AMP** | "Automatic mixed precision": using FP16 where safe and FP32 where needed, inside PyTorch. |
| **Latency** | Time to process one scan. |
| **Smoke test** | A tiny, quick run that only checks that every step of the pipeline works, not that the results are good. |

### Where to go next

| If you are... | Read |
|---|---|
| Curious, with no technical background | [**The project explained for everyone**](docs/overview-for-everyone.md): a longer, friendly walk-through with an FAQ |
| An engineer new to medical imaging or AI | The [**learning track**](docs/learn/README.md): 13 chapters from first principles |
| Already an expert | The [**technical reference**](docs/reference/README.md), or keep reading below |
| Wanting to run it | [Reproduce](#reproduce) below, or the Colab button at the top |

---

## For technical readers

An end-to-end, reproducible pipeline for **3D spleen segmentation in
contrast-enhanced CT**: a SwinUNETR trained from scratch with MONAI on MSD
Task09, exported to ONNX with a numerical parity check, compiled to a
**TensorRT FP16** engine, and benchmarked against PyTorch FP32 and AMP on an
NVIDIA T4 for **per-volume latency and Dice**.

The scope is deliberately narrow: one cohort, one architecture, one deployment
path. The effort goes into a correct pipeline and a measurement protocol that
says exactly what each number means.

> **Status:** implementation, tests, CPU CI and the local smoke test are
> complete. The T4 benchmark has been run from
> [the Colab notebook](notebooks/colab_train_and_benchmark.ipynb), but on a
> **short 20-epoch training run** (see Results). The speed-up numbers are
> measured; the Dice is from an under-trained model and is not
> representative of what SwinUNETR can achieve on this task. Nothing here is
> estimated.

**Documentation comes in three tracks:**

| | For | Start at |
|---|---|---|
| **Technical reference** | Readers fluent in DL, medical imaging and GPU inference who want the design decisions, exact settings and caveats | [docs/reference/](docs/reference/README.md) |
| **Learning track** | Engineers new to the field: CT physics, signals, ML, transformers, training and GPU deployment, from first principles | [docs/learn/](docs/learn/README.md) |
| **For everyone** | Readers with no technical or medical background | [docs/overview-for-everyone.md](docs/overview-for-everyone.md) |

---

### Motivation

Pre-operative organ segmentation feeds surgical planning, registration and
navigation, and virtual-fixture constraints in robot-assisted surgery. The
spleen is a relevant at-risk organ: highly vascular, and a recognised site of
iatrogenic injury in left-upper-quadrant and colorectal procedures (for
example, splenic flexure mobilisation). Clinical use needs two things: an
accurate 3D shape, and inference fast enough on the target GPU for the
workflow it serves. Hence the second half of this project: TensorRT, which is
the same toolchain used on Jetson-class embedded GPUs.

### Method

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
| Parity gate | Every backend vs PyTorch FP32 (TF32 off), one tolerance tier per precision (fp32 strict elementwise; fp16/bf16/int8 judged on class flips, confident flips and Dice drop); fixed-seed golden batches + optional whole volumes; non-zero exit on failure | [deployment.md §2.1](docs/reference/deployment.md#21-parity-gate) |
| Engine | TensorRT 10, FP16 builder flag, per-layer precision by the builder; zero-copy integration on PyTorch's CUDA stream | [deployment.md §3–4](docs/reference/deployment.md#3-tensorrt-engine-build) |
| Benchmark | Synchronised wall-clock latency of whole-volume sliding-window inference (GPU-resident input), 2 warm-up + 8 vols × 3 repeats; mean / p50 / p95; Dice per backend | [benchmarking.md](docs/reference/benchmarking.md) |

```
MSD Spleen ─▶ MONAI preprocessing ─▶ SwinUNETR training ─▶ best.pt ─┬─▶ PyTorch FP32 / AMP ──┐
                                                                    └─▶ ONNX (+ parity) ─▶ TensorRT FP16 ─┴─▶ latency + Dice
```

### Results

> ⚠️ **Read this first: the model is under-trained.** These values come from
> `notebooks/colab_train_and_benchmark.ipynb` on a Colab T4 with
> `--epochs 20` instead of the configured 150. The mean validation Dice is
> **0.587**, far below the roughly 0.9+ expected for a fully trained model on
> this dataset. The **latency and speed-up figures are valid** (they depend on
> the network's compute, not on how well it is trained), and Dice is the same
> across all three backends, so precision changes did not cost accuracy. But
> **do not read 0.587 as the accuracy this architecture can reach.** Why this
> happened and how to fix it is under
> [Why the Dice is low](#why-the-dice-is-low-and-how-to-fix-it).
> The field-to-cell mapping is in
> [docs/author-notes/reporting-results.md](docs/author-notes/reporting-results.md).

#### Training

| | Value |
|---|---|
| Train / validation volumes | 33 / 8 |
| Model | SwinUNETR, $F = 48$, 62.2 M parameters |
| Epochs trained | **20** (config default is 150; shortened run) |
| Training time (T4) | 42.6 min (gradient checkpointing on) |
| Best validation Dice (AMP, during training) | 0.5874 at epoch 20 (the last epoch) |

![Training curve](results/training_curve.png)

Validation Dice at epochs 5 / 10 / 15 / 20 was 0.345 / 0.426 / 0.451 / 0.587.
It was still rising when training stopped, and the best checkpoint is the
final epoch, so the run ended before the model converged.

#### Inference (NVIDIA T4, 8 validation volumes × 3 repeats)

Latency = whole-volume sliding-window inference (96³, batch 4, overlap 0.5),
input already on the GPU, after warm-up, CUDA-synchronised. Preprocessing,
I/O and post-processing are excluded.

| Backend | Mean latency / volume (ms) | p95 (ms) | Speed-up vs FP32 | Mean Dice |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 34476.1 | 59137.6 | 1.00× | 0.5873 |
| PyTorch AMP (FP16 autocast) | 14587.1 | 24866.0 | 2.36× | 0.5874 |
| TensorRT FP16 | 6182.1 | 10212.9 | 5.58× | 0.5873 |

#### Export

| | Value |
|---|---|
| ONNX vs PyTorch max abs. logit difference | 7.63e-05 |
| ONNX vs PyTorch argmax agreement | 99.99997 % |
| TensorRT engine build time / size | 513 s / 125.7 MB |
| Software | Tesla T4, CUDA 13.0, cuDNN 9.28, PyTorch 2.14.1+cu130, TensorRT 10.16.1.11, MONAI 1.6.1, Python 3.13.16 |

#### Reading these results

- **Speed-up is real.** TensorRT FP16 is 5.58× faster than FP32 and 2.36×
  faster than AMP on the same volumes. AMP's gain comes mostly from FP16 Tensor
  Cores; TensorRT's extra gain over AMP comes from layer fusion, kernel
  selection and less framework overhead.
- **Precision did not change the answer.** Dice is 0.5873 / 0.5874 / 0.5873
  across FP32 / AMP / TensorRT, and ONNX agrees with PyTorch on all but a
  handful of voxels. This holds regardless of how well the model is trained.
- **Accuracy is poor and uneven.** Per-volume Dice ranges from 0.29 to 0.87:
  three volumes are above 0.7 and three are below 0.34, which is typical of a
  model that has not finished learning.
- **p95 is wide because scans differ in size**, not because of jitter
  (larger scans need more sliding windows).

### Why the Dice is low, and how to fix it

**Cause.** `results/training_history.json` records `epochs: 20`, not the
default 150. Training loss was still falling (0.67 to 0.29) and validation
Dice was still climbing (0.345 to 0.587) at the final epoch. The model had not
converged. Three things made this worse:

- A SwinUNETR with 62 M parameters trained **from scratch** on only 33 volumes
  needs many epochs; there are no pretrained weights to start from.
- The cosine learning-rate schedule is set over the number of epochs, so in a
  20-epoch run it decayed to zero by epoch 20 instead of staying high for
  much longer.
- Validation ran every 5 epochs, so only four Dice measurements exist.

This is a training-budget problem, not a deployment bug: the export and
benchmark code reproduced the same Dice on all three backends.

**Remediation.**

1. **Retrain with the full 150 epochs.** Resume cannot extend this run,
   because `--resume` requires the epoch count to match and the schedule has
   already reached zero. Start a fresh run
   (`python -m src.train --epochs 150`).
2. **Plan for the time.** 20 epochs took 42.6 min, so 150 epochs is roughly
   5 to 6 hours on a T4. That is close to Colab's session limit.
3. **Keep checkpoints off the VM.** Set `--checkpoint-dir` to a mounted
   Google Drive folder so a disconnect doesn't lose `last.pt`, and use
   `--resume` (same flags) if the session drops.
4. **Free 150-epoch plan.** The step-by-step plan (background run on Kaggle's
   free GPU, resume strategy, Colab fallback, and how to finish the benchmark
   afterwards) is in
   [docs/reference/free-training-150-epochs.md](docs/reference/free-training-150-epochs.md);
   the notebook is
   [notebooks/kaggle_train_150.ipynb](notebooks/kaggle_train_150.ipynb). It is
   written but **has not been run yet**.
5. **Then rerun stage 8** and update the table above. Latency should barely
   change; Dice should rise substantially. If it still stays far below 0.9
   after 150 epochs, check preprocessing and the train/validation split
   before changing the model.

### Limitations

- **Small, single-centre cohort.** 41 portal-venous CTs from one institution
  and one patient population (liver metastases under chemotherapy). No claim
  of external validity across scanners, phases or pathology.
- **Under-trained model in the published numbers.** The reported Dice comes
  from a 20-epoch run (configured: 150). It illustrates the speed and
  precision comparison, not the accuracy this architecture can achieve.
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

### Reproduce

**Local (CPU/MPS, no NVIDIA GPU).** This is a pipeline check with a tiny
model ($F = 12$), 2 volumes, 64³ crops and 2 epochs. The outputs go to the
git-ignored `results/smoke/`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.train --smoke-test --synthetic      # drop --synthetic to use 2 real volumes (1.5 GB download)
python -m src.export_onnx --smoke-test
python -m src.parity --smoke-test --synthetic --volumes 1   # TensorRT recorded as "skipped: needs CUDA"
python -m src.benchmark --smoke-test --synthetic  # AMP / TensorRT recorded as "skipped: needs CUDA"
```

**Full run (Colab T4).** Open the notebook with the badge above, select
*Runtime → Change runtime type → T4 GPU*, then *Run all*. It installs pinned
dependencies and TensorRT, tests, downloads, trains, exports, benchmarks and
packages `results.zip`. Training saves its full state after every epoch, so
after a disconnect you can resume where it stopped (`RESUME = True` in the
notebook, `--resume` on the CLI). See [colab.md](docs/reference/colab.md). On any
CUDA Linux host:

```bash
pip install -r requirements-colab.txt    # plus a CUDA build of torch
python -m src.data --download && python -m src.train && python -m src.export_onnx && python -m src.parity --volumes 8 && python -m src.benchmark
```

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml),
and common ones can be overridden on the CLI (`--epochs`, `--feature-size`,
`--lr`, ...). The merged config and environment are embedded in every
checkpoint and result file.

### Repository layout

```
configs/default.yaml        hyperparameters (+ smoke-test overrides)
src/                        data, model, evaluation, train, export_onnx, trt_utils, benchmark, utils
tests/                      pytest on synthetic data (CPU, ~2 min; the resume test trains a tiny model)
notebooks/                  Colab T4 end-to-end run
scripts/                    make_overview_figure.py (renders docs/images/ct_spleen_example.png)
docs/overview-for-everyone.md  plain-English guide (no background needed)
docs/reference/             technical reference (expert audience)
docs/learn/                 learning track (first principles)
docs/author-notes/          how to transcribe and phrase results
results/                    committed results after the T4 run
CHANGELOG.md                release notes
```

Data, checkpoints, ONNX files and engines are never committed. Details are in
[code-and-reproducibility.md](docs/reference/code-and-reproducibility.md).

### Extensions

5-fold CV, native-space Dice (`Invertd`) and HD95/NSD. Self-supervised
pre-trained Swin weights (Tang et al., 2022) and a 3D U-Net / nnU-Net
baseline. INT8 with calibration, FP32-pinned normalisation layers, dynamic
batch profiles and CUDA graphs. GPU-side preprocessing with end-to-end
latency. Jetson Orin deployment and a ROS 2 node. Multi-organ (BTCV, AMOS,
TotalSegmentator).

### References

- Hatamizadeh, A. et al. *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.* BrainLes 2021. arXiv:2201.01266.
- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- Tang, Y. et al. *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis.* CVPR 2022.
- Antonelli, M. et al. *The Medical Segmentation Decathlon.* Nat. Commun. 13, 4128 (2022). Data: <http://medicaldecathlon.com> (CC-BY-SA 4.0).
- [MONAI](https://monai.io) (model, transforms, spleen tutorial defaults), [ONNX](https://onnx.ai), [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).

Copyright (c) 2026 Kshitij Vengurlekar. All rights reserved. This code is
**not open source**: you may not use, copy, modify or distribute it without
the author's prior written consent. See [LICENSE](LICENSE). The dataset is
not included and keeps its own license.
