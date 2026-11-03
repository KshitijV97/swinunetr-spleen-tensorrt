# Reporting results: what to copy where

After the Colab run you will have five files in `results/`. This page lists
which numbers go into the README and your résumé, and how to phrase them
honestly.

---

## 1. Numbers for the README "Results" section

| README placeholder | Where to find it |
|---|---|
| Benchmark table (all rows) | Copy `results/benchmark.md` as-is |
| Best validation Dice (AMP, during training) | `training_history.json → best_val_dice` (and `best_epoch`) |
| Epochs trained / training time | `training_history.json → config.train.epochs`, `train_minutes` |
| Train / val volumes | `training_history.json → n_train`, `n_val` (should be 33 / 8) |
| Parameters | `training_history.json → n_parameters` |
| ONNX max abs diff and argmax agreement | `onnx_export.json → max_abs_diff`, `argmax_agreement` |
| TensorRT engine build time and size | `benchmark.json → tensorrt_engine.build_seconds`, `engine_size_mb` |
| GPU, CUDA, PyTorch, TensorRT versions | `benchmark.json → environment` |
| Training curve | `results/training_curve.png` (embed it in the README) |

The notebook's section 10 prints most of these for you.

Delete the "TBD" markers only once each number is in place.

## 2. The headline numbers

For a one-line summary or a résumé bullet, these are the numbers that matter:

1. **Validation Dice of TensorRT FP16** (`benchmark.json`, `tensorrt_fp16 → dice_mean`), next to the **PyTorch FP32 Dice**: shows accuracy was preserved.
2. **Mean latency per volume for PyTorch FP32 and TensorRT FP16**, and the
   **speed-up** (FP32 latency ÷ TensorRT latency, also in `benchmark.md`).
3. **The GPU** (T4) and the input setup (96³ windows, sliding window, 50% overlap).

## 3. Résumé bullet templates

Replace every `<...>` with your real numbers. Do not round favourably, and do
not use a number you did not measure.

> Trained a SwinUNETR (Swin Transformer + CNN decoder, <62M> params) for 3D
> spleen segmentation on MSD CT data with MONAI, reaching <0.XX> validation
> Dice (8 held-out volumes).

> Exported the model to ONNX (verified against PyTorch within 1e-3) and
> compiled a TensorRT FP16 engine, cutting sliding-window inference latency
> on an NVIDIA T4 from <X> s to <Y> s per CT volume (<Z>× speed-up) with a
> Dice change of <±0.00X>.

> Built a reproducible pipeline (config-driven training, CI-friendly CPU
> smoke test, unit tests, Colab notebook) for training, export and
> benchmarking of 3D medical segmentation models.

## 4. How to talk about it honestly

- Say **validation** Dice, and say it is 8 volumes.
- Say **T4**, and that latency is model inference only (preprocessing excluded).
- If you trained for fewer epochs than the default, say how many.
- If TensorRT FP16 lowered Dice, report it. "FP16 reduced Dice by 0.004;
  next step would be keeping LayerNorm in FP32" is a stronger interview
  answer than a suspiciously perfect table.
- If AMP and TensorRT are close, that is also an interesting finding:
  much of TensorRT's gain on a T4 comes from FP16 Tensor Cores, which AMP also
  uses. The remaining difference is layer fusion and kernel auto-tuning.

## 5. Questions you should be ready for

- *Why SwinUNETR rather than a 3D U-Net?* See [the learning track, chapter 7](../learn/07-swin-and-swinunetr.md) and [architecture.md](../reference/architecture.md).
- *Why fixed input shapes for TensorRT?* See [deployment.md §2](../reference/deployment.md#2-onnx-export).
- *What exactly does your latency number include?* See [benchmarking.md §2](../reference/benchmarking.md#2-unit-of-measurement).
- *How would this run inside a robot?* Rebuild the engine on the target GPU
  (e.g. Jetson Orin), add preprocessing on the GPU, possibly INT8, and
  integrate with the robot's perception stack (e.g. a ROS 2 node).
- *What are the weaknesses?* Small dataset, single split, validation-based
  checkpoint selection, Dice in resampled space, one GPU type. All are listed
  in the README's limitations section.
