# Benchmark protocol

Code: [`src/benchmark.py`](../../src/benchmark.py).

---

## 1. Backends

| Backend | Artifact | Numerics | Requires |
|---|---|---|---|
| `pytorch_fp32` | `best.pt` | FP32 (T4 has no TF32) | any device |
| `pytorch_amp` | `best.pt` | `autocast(float16)`: FP16 conv/matmul, FP32 norms/softmax | CUDA |
| `tensorrt_fp16` | ONNX → engine | Per-layer FP16/FP32 chosen by the builder | CUDA + TensorRT |

Unavailable backends are recorded as `skipped` with a reason. They are never
estimated.

## 2. Unit of measurement

**One sample = wall-clock latency of `sliding_window_inference` over one
whole preprocessed validation volume that is already resident on the GPU.**

| Included | Excluded |
|---|---|
| All forward passes (batch 4, 50 % overlap) | NIfTI I/O, reorientation, resampling, windowing, cropping |
| MONAI window extraction, Gaussian weighting, accumulation, normalisation (identical across backends) | argmax, Dice, D2H transfer |
| TensorRT: D2D copy into the bound input buffer, output clone | Engine build (reported separately as `build_seconds`) |
| Python dispatch overhead | Process start-up, CUDA context creation (absorbed by warm-up) |

These are **model-inference latencies**, not end-to-end system latencies.

## 3. Procedure

1. Preprocess all 8 validation volumes and move them to the device.
2. Warm-up: `warmup_volumes = 2` untimed inferences (context, cuDNN heuristics,
   allocator growth, TensorRT lazy init).
3. For each sample: `torch.cuda.synchronize()`, `perf_counter()`,
   inference, `synchronize()`, `perf_counter()`.
4. `repeats = 3` passes over the 8 volumes give 24 samples per backend. Dice
   is computed on the first pass only (deterministic given the backend).
5. Before TensorRT runs, the PyTorch model is deleted and the cache emptied.
   `peak_gpu_mem_mb` (`torch.cuda.max_memory_allocated`) is recorded per CUDA
   backend. It counts only PyTorch-allocator memory, so for `tensorrt_fp16`
   it covers the I/O buffers and volumes but not TensorRT's internal
   activation memory.

Statistics: mean, p50, p95 (`numpy.percentile`, linear interpolation),
population std. All raw samples are kept in `latency_ms_all`.

## 4. Interpretation

- **Variance is dominated by volume size, not jitter.** Latency is
  proportional to the window count, which varies across the 8 cases. The std
  and p95 describe the case mix. For jitter, compare repeats of the same
  volume (indices $i$, $i + 8$, $i + 16$ in `latency_ms_all`).
- **p95 from 24 samples** sits among the top two values. Read it as "the
  slow end of this case mix".
- **Speed-up** = mean FP32 latency ÷ mean backend latency. Because every
  backend sees the same volumes, the ratio of means equals a
  window-count-weighted average of per-volume speed-ups.
- **Attribution.** On Turing, most of the FP32 → FP16 gain comes from Tensor
  Cores, which AMP also uses. The TRT-vs-AMP gap isolates fusion, tactic
  selection, removal of framework dispatch and reduced memory traffic.
  Reporting all three backends makes this decomposition visible.
- **Constant overhead.** MONAI's Python-side blending is shared by all
  backends and does not shrink with a faster network. As the network
  accelerates, this Amdahl term bounds the achievable end-to-end speed-up.
- **Environment.** Colab T4s are virtualised and shared, and clock behaviour
  is not controlled. Driver, CUDA, cuDNN, torch and TensorRT versions are
  recorded in `benchmark.json → environment`. Prefer reporting multiple
  sessions over a best-of.

## 5. Outputs

| File | Content |
|---|---|
| `results/benchmark.json` | Config subset, environment, per-backend stats, per-volume Dice, all samples, engine build info |
| `results/benchmark.md` | README table |
| `results/onnx_export.json` | Export parity metrics |
| `results/training_history.json` | Training curve, best validation Dice |

## 6. Extensions

Overlap and `sw_batch_size` sweeps. Dynamic-batch engine. CUDA graphs and
`torch.compile` baselines. GPU-side preprocessing and end-to-end timing.
Additional targets (L4, A100, Jetson Orin). Confidence intervals across
sessions. Inter-backend voxel agreement on real volumes.
