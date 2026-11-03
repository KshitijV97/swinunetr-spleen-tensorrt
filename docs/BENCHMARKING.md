# Benchmark methodology

Code: [`src/benchmark.py`](../src/benchmark.py). This page states exactly
what the latency and Dice numbers mean, so nobody (including a future you in
an interview) over-reads them.

---

## 1. What is compared

| Backend | Weights | Precision | Runs on |
|---|---|---|---|
| `pytorch_fp32` | `best.pt` | FP32 everywhere (cuDNN may use TF32 internally, PyTorch's default on Ampere and newer GPUs; the T4 is Turing, which has no TF32) | Any device |
| `pytorch_amp` | `best.pt` | `torch.autocast(float16)`: FP16 matmuls/convs, FP32 for sensitive ops | CUDA only |
| `tensorrt_fp16` | ONNX export of `best.pt`, compiled by TensorRT | TensorRT chooses FP16 or FP32 per layer | CUDA + TensorRT only |

Backends that cannot run on the current machine are recorded as
`"status": "skipped"` with a reason. They are never estimated or filled in.

## 2. What one latency sample is

**One sample = sliding-window inference over one whole preprocessed
validation volume, measured with wall-clock time.**

Included:
- every forward pass of the network over all windows of the volume
  (`sw_batch_size = 4` windows per pass, 50% overlap);
- MONAI's sliding-window bookkeeping: cutting windows, Gaussian blending,
  stitching (identical for every backend);
- for TensorRT, copying each batch of windows into the engine's input buffer
  and the output back into a PyTorch tensor.

Excluded:
- reading the NIfTI file, reorientation, resampling, intensity scaling and
  cropping (done once, up front; every volume is already on the GPU);
- argmax, Dice computation and moving results to CPU;
- the TensorRT engine build (reported separately as `build_seconds`).

So these are **model-inference latencies**, not end-to-end application
latencies. A real system would add I/O and preprocessing time.

## 3. How timing is done

1. All validation volumes are preprocessed and moved to the GPU once.
2. **Warm-up:** the first `warmup_volumes` (2) volumes are processed without
   timing. The first CUDA calls pay one-off costs: context creation, cuDNN
   algorithm search, memory-pool growth, TensorRT lazy initialisation.
3. For each timed volume: `torch.cuda.synchronize()`, start
   `time.perf_counter()`, run inference, `torch.cuda.synchronize()`, stop.
   GPU calls return before the GPU has finished, so without the
   synchronisation you would time only the Python launching the kernels.
4. The whole validation set is timed `repeats` (3) times, so there are
   8 volumes × 3 = 24 samples per backend.
5. Reported statistics: mean, median (p50), 95th percentile (p95), standard
   deviation and peak GPU memory. All raw samples are kept in
   `benchmark.json` (`latency_ms_all`).

Before TensorRT runs, the PyTorch model is deleted and the CUDA cache
emptied, so TensorRT is not competing for memory.

## 4. How to read the numbers

- **Latency depends on volume size.** A bigger scan needs more windows. The
  mean is therefore a mean over *these 8 scans*, and the spread (std, p95)
  mostly reflects scan sizes, not jitter.
- **p95 with few samples is rough.** With 24 samples, p95 sits between the
  largest few values. Treat it as "the slow end", not a precise tail estimate.
- **Speed-up** in `benchmark.md` = FP32 mean latency ÷ backend mean latency.
- **Dice** is computed per volume and averaged, exactly as in training
  validation (see [DATA.md](DATA.md#5-where-dice-is-measured-an-honest-caveat)
  for the "resampled space" caveat). Identical preprocessing and
  sliding-window settings for every backend mean any Dice difference comes
  from numerical precision alone.
- **Hardware matters.** Every number is specific to a Colab T4 with
  whatever driver, CUDA, PyTorch and TensorRT versions Colab provided that day.
  These are recorded in `benchmark.json → environment`. Colab GPUs are shared
  virtual machines, so expect some run-to-run variation; re-running and
  reporting both runs is more honest than picking the best one.

## 5. Output files

- `results/benchmark.json`: full report (config used, environment,
  per-backend summary statistics, per-volume Dice and every latency sample,
  TensorRT engine build time and size).
- `results/benchmark.md`: the Markdown table that goes into the README.
- `results/onnx_export.json`: ONNX verification numbers.
- `results/training_history.json`: training curve and best validation Dice.

## 6. What would make the benchmark stronger

- More GPUs (A100, L4, Jetson Orin) and an end-to-end timing that includes
  preprocessing.
- Batch-size and `overlap` sweeps (overlap 0.25 roughly halves the number of
  windows compared with 0.5).
- CUDA graphs / `torch.compile` as additional PyTorch baselines.
- Reporting confidence intervals across several Colab sessions.
