# 12. Benchmarking honestly

> **Summary.** A latency number is only meaningful if you know exactly what
> it includes. GPU timing has specific traps:
> - asynchronous execution: you must synchronise;
> - one-off start-up costs: warm up first;
> - variable inputs: bigger scans take longer.
>
> This chapter explains how [`src/benchmark.py`](../../src/benchmark.py)
> avoids them, how to read mean/p95/speed-up, and why Amdahl's law caps the
> speed-up you'll see.

---

## 12.1 Define the measurement first

**One latency sample = sliding-window inference over one whole,
already-preprocessed validation scan that is already on the GPU.**

| Included | Not included |
|---|---|
| Every forward pass of the network over all windows | Reading the file, reorienting, resampling, windowing |
| MONAI's window cutting, Gaussian weighting and stitching | Argmax, Dice and copying results back to the CPU |
| For TensorRT: copying windows into its input buffer and the output back out | Building the TensorRT engine (reported separately) |

So these are **model-inference latencies**, not "time from scan file to
answer". A real system adds I/O and preprocessing. Anyone quoting the numbers
should say "model latency on a T4".

## 12.2 Trap 1: asynchronous execution

Chapter 10 explained that CUDA calls return immediately. A naive timer

```python
t0 = time.perf_counter(); model(x); t1 = time.perf_counter()   # WRONG on GPU
```

measures how long Python took to *queue* the work. The fix is to wait for the
GPU before starting and before stopping the clock:

```python
torch.cuda.synchronize(); t0 = time.perf_counter()
logits = infer(image)
torch.cuda.synchronize(); t1 = time.perf_counter()
```

That's exactly what `run_backend` does (via `_sync`, which also handles
Apple's MPS backend).

## 12.3 Trap 2: start-up costs

The first GPU calls pay one-off costs: creating the CUDA context, cuDNN
choosing algorithms, the memory allocator growing its pool, TensorRT
initialising lazily. Timing them would make the first backend look slow for
reasons unrelated to steady-state speed. So the first **2 scans are processed
untimed** (warm-up).

Also, before the TensorRT run the PyTorch model is deleted and the GPU cache
emptied, so the two don't compete for memory.

## 12.4 Trap 3: inputs of different sizes

Latency is roughly proportional to the number of windows, which depends on
scan size (chapter 9). So across the 8 validation scans the spread of
latencies mostly reflects **different scan sizes**, not timing noise. To see
real jitter, compare the same scan across repeats. The benchmark times all 8
scans 3 times (24 samples) and keeps every raw sample in
`benchmark.json → latency_ms_all`, so scan $i$ appears at positions $i$,
$i + 8$ and $i + 16$.

## 12.5 Reading the statistics

| Statistic | Meaning | Caveat here |
|---|---|---|
| Mean | Average over 24 samples | An average over *this* mix of scan sizes |
| p50 (median) | Half the samples are faster | Robust to one slow outlier |
| p95 | 95 % of samples are at or below this | With 24 samples, that's essentially the 2nd-slowest. Read it as "the slow end", not a precise tail estimate |
| Std | Spread | Mostly the scan-size mix |
| Speed-up | FP32 mean ÷ backend mean | Same scans for every backend, so the ratio is fair |

## 12.6 Amdahl's law: why the speed-up is capped

Only part of the measured time is the network. MONAI's window bookkeeping
(cutting, weighting, accumulating, dividing) runs the same way for every
backend and doesn't get faster when the network does. If a fraction $f$ of
the FP32 time is network and the network gets $s$ times faster, the overall
speed-up is

```math
S = \frac{1}{(1 - f) + f/s}
```

Example: if the network is 95 % of the time ($f = 0.95$) and TensorRT makes
it 5× faster, the overall speed-up is $1 / (0.05 + 0.19) \approx 4.2\times$,
not 5×. The faster the network, the more the fixed overhead matters. Keep
that in mind before calling a measured speed-up "disappointing".

## 12.7 Accuracy is part of the benchmark

A faster model that's less accurate isn't a free win. Every backend is scored
with the same Dice on the same scans with the same windows, so any Dice
difference comes from **numerical precision alone**. Interpreting it:

- A difference in the 4th decimal place: FP16 is harmless for this task.
- A difference in the 2nd or 3rd decimal place: report it, then look into
  forcing sensitive layers (LayerNorm, softmax) to FP32.
- Remember from chapter 8 that two GPU runs of the *same* backend can differ
  by about 1e-3. Smaller differences than that aren't findings.

## 12.8 The environment is part of the result

Colab's T4s are shared, virtualised machines, and the software versions
change over time. Every result file records Python, PyTorch, MONAI, CUDA,
cuDNN, TensorRT and the GPU name (`environment` in `benchmark.json`). Without
that, a benchmark number can't be interpreted or reproduced. If you run twice
and get different numbers, report both rather than picking the better one.

## 12.9 A checklist for any benchmark you read

1. What exactly is timed? (Start and end points, what's excluded.)
2. Was the GPU synchronised? Was there a warm-up?
3. How many samples? Are the inputs representative?
4. Is accuracy reported alongside speed?
5. Is there a fair baseline (here: AMP, not just FP32)?
6. What hardware and software versions?

This project's [benchmark reference](../reference/benchmarking.md) answers
each of these explicitly.

---

**In the code**
- [`src/benchmark.py`](../../src/benchmark.py) → `run_backend` (warm-up,
  synchronised timing, Dice, statistics) and `to_markdown` (the README
  table).
- Benchmark settings: [`configs/default.yaml`](../../configs/default.yaml),
  `benchmark:` block (`warmup_volumes`, `repeats`).

**Check yourself**
1. What would you measure if you forgot `torch.cuda.synchronize()`?
2. Why is the std of latencies across scans not a measure of noise?
3. With $f = 0.9$ and a 10× faster network, what's the overall speed-up?
4. TensorRT Dice is 0.0004 lower than FP32. Is that a finding?

Next: [13. Software engineering of this repo →](13-software-engineering-of-this-repo.md)
