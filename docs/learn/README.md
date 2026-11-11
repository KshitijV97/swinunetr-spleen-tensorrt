# Learning track: understand this whole project from first principles

**Who this is for:** you've written production software for years, and you
studied electrical engineering a while ago. You remember roughly what a
convolution, a Fourier transform and a floating-point number are, but the
details have faded. You've never trained a neural network or looked at a CT
scan. By the end of this track you should be able to explain every line of
this repository, and why it is there, to a colleague at a whiteboard.

**How it's written:** plain language first, then the maths, and the maths
always comes with a small worked example. Wherever an idea maps onto
something you already know from EE or software, the chapter says so. A box
called **In the code** at the end of each section points to the exact file
and function.

---

## The project in one paragraph

A CT scanner produces a 3D grid of numbers describing how much each tiny
cube of the body absorbs X-rays. We want a program that takes that grid and
marks every cube that belongs to the **spleen**, an organ that surgeons must
avoid damaging. Writing that program by hand is hopeless, so we **learn** it
from 41 scans that radiologists have already labelled. The learned program
is a neural network called **SwinUNETR**. Once it works, we make it run fast
on an NVIDIA GPU by converting it to **ONNX** (a portable format) and
compiling it with **TensorRT** (an optimising compiler for neural networks).
Then we measure carefully how much faster it got and whether it got any less
accurate.

## Reading order

Each chapter builds on the previous ones. If you already know a topic, skim
its summary box and move on.

| # | Chapter | You'll learn | Time |
|---|---|---|---|
| 1 | [The medical problem](01-the-medical-problem.md) | What the spleen is, why surgeons care, how CT works physically, Hounsfield units, scan files | 30 min |
| 2 | [Images as signals](02-images-as-signals.md) | Voxels as samples, spacing as sampling period, resampling as interpolation, windowing, 3D convolution as a filter | 30 min |
| 3 | [Machine learning from zero](03-machine-learning-from-zero.md) | Learning = fitting a function, loss, gradient descent (the LMS filter you already know), backprop as the chain rule, overfitting, train/val/test | 45 min |
| 4 | [Neural network building blocks](04-neural-network-building-blocks.md) | Linear layers, activations, softmax, cross-entropy, normalisation, residual connections | 40 min |
| 5 | [CNNs and the U-Net](05-cnns-and-unet.md) | Learned filters, receptive fields, downsampling/upsampling, why U-Nets are the go-to for segmentation | 35 min |
| 6 | [Attention and transformers](06-attention-and-transformers.md) | Tokens, queries/keys/values as a soft hash map, why divide by $\sqrt{d}$, multi-head attention, the quadratic cost | 45 min |
| 7 | [Swin and SwinUNETR](07-swin-and-swinunetr.md) | Windowed and shifted attention, patch merging, the full network with every tensor shape | 45 min |
| 8 | [Training this model](08-training-this-model.md) | Dice loss, AdamW, learning-rate schedules, floating-point formats and mixed precision, memory budgeting, what one epoch does, surviving interruptions with `--resume` | 60 min |
| 9 | [Inference and evaluation](09-inference-and-evaluation.md) | Sliding windows, blending, Dice, why 8 validation scans is a small number, what "good" means clinically | 30 min |
| 10 | [GPUs and deployment](10-gpus-and-deployment.md) | GPU architecture, Tensor Cores, the roofline model, async execution, ONNX as an IR, TensorRT as a compiler | 50 min |
| 11 | [Is the fast model still the same model?](11-the-parity-gate.md) | The automated parity gate: why outputs can't match exactly, margins and confident flips, tolerance tiers, golden inputs vs whole scans, stale-engine protection, how to explain it in an interview | 35 min |
| 12 | [Benchmarking honestly](12-benchmarking-honestly.md) | Measuring asynchronous hardware, warm-up, percentiles, Amdahl's law, how to read the results table | 25 min |
| 13 | [Software engineering of this repo](13-software-engineering-of-this-repo.md) | Config layering, provenance, determinism, guarded imports, designing for interruption, test strategy and fault injection, CI, how to extend | 25 min |
| 14 | [Where to go next](14-where-to-go-next.md) | Books, courses, papers and exercises, in priority order | 10 min |
| — | [Glossary](glossary.md) | Every term, one line each | reference |

## A translation table for EE and software people

Much of deep learning reuses ideas you already know under different names.

| Deep-learning term | What you already know | Chapter |
|---|---|---|
| Convolution layer | FIR filter (3D, multi-channel, with learned taps) | 2, 5 |
| Stride-2 convolution | Filter + decimation by 2 | 5 |
| Transposed convolution | Zero-stuffing upsampler + interpolation filter | 5 |
| Gradient descent | LMS adaptive filter / iterative optimisation | 3 |
| Backpropagation | Chain rule; reverse-mode automatic differentiation; adjoint sensitivity analysis | 3 |
| Learning rate | Step size $\mu$ in LMS: too large diverges, too small crawls | 3 |
| Loss function | Error signal / cost function / objective | 3 |
| Attention | Content-addressable memory; a differentiable hash-map lookup | 6 |
| Softmax | A "soft" argmax; the Boltzmann distribution | 4 |
| Normalisation layers | Automatic gain control | 4 |
| Residual connection | A feed-forward bypass path: output = input + correction | 4 |
| Mixed precision | Fixed/floating-point word-length trade-offs in DSP | 8 |
| ONNX | A compiler intermediate representation (like LLVM IR) | 10 |
| TensorRT engine | A binary compiled with `-march=native` for one specific GPU | 10 |
| Sliding-window inference | Overlap-add block processing | 9 |
| Parity gate | A regression test with tolerances, run in CI; a build cache keyed on content hashes | 11 |
| Data augmentation | Generating extra test vectors by symmetry | 3, 8 |
| Overfitting | Fitting noise: a 40th-order polynomial through 41 points | 3 |

## How to use this track with the code

1. Set up the local environment (see the main [README](../../README.md#reproduce))
   and run `pytest -q`. Everything works on a Mac without a GPU.
2. Read a chapter, then open the files named in its **In the code** boxes.
3. Run the smoke test (`python -m src.train --smoke-test --synthetic`) and
   watch the log while reading chapter 8.
4. Use the [technical reference](../reference/README.md) as the compact
   "spec sheet" once a chapter has given you the intuition.
