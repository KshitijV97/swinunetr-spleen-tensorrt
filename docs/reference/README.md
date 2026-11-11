# Technical reference

Reference documentation for readers who already know deep learning, 3D medical
image segmentation and GPU inference. These pages describe **what this
repository does and why**: design decisions, exact settings, equations where
they help, and the threats to validity of every reported number. They do not
teach background concepts. For that, see the [learning track](../learn/README.md).

| Page | Contents |
|---|---|
| [data.md](data.md) | MSD Task09 cohort, split, preprocessing pipeline with parameters, patch sampling, caching, synthetic fixtures |
| [architecture.md](architecture.md) | SwinUNETR as instantiated here: tensor shapes, window and shift sizes per stage, decoder wiring, parameter counts, export-relevant details |
| [training.md](training.md) | Objective (DiceCE as MONAI computes it), optimiser, schedule, AMP, checkpoint selection, config system, hyperparameter table, interruption and `--resume` |
| [inference-and-evaluation.md](inference-and-evaluation.md) | Sliding-window inference and Gaussian blending, Dice definition and aggregation, evaluation protocol and its caveats |
| [deployment.md](deployment.md) | ONNX export and verification, TensorRT FP16 engine build, runtime integration, CUDA guard |
| [benchmarking.md](benchmarking.md) | Latency measurement protocol, what is included and excluded, statistics, how to read the table |
| [code-and-reproducibility.md](code-and-reproducibility.md) | Module map, data flow, determinism, provenance metadata, tests, CI |
| [colab.md](colab.md) | Running the full pipeline on a Colab T4, with troubleshooting |

Notation used throughout: $C$ channels, $D \times H \times W$ spatial size,
$F$ = `model.feature_size`, $P$ predicted foreground mask, $G$ ground-truth
mask, $\lvert\cdot\rvert$ voxel count.
