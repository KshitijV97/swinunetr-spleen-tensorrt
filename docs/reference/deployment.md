# Deployment: ONNX export and TensorRT FP16

Code: [`src/export_onnx.py`](../../src/export_onnx.py),
[`src/trt_utils.py`](../../src/trt_utils.py),
[`src/benchmark.py`](../../src/benchmark.py).

---

## 1. Path

```
best.pt ──torch.onnx.export (TorchScript tracer, opset 18)──▶ model.onnx (FP32, static shape)
        ──TensorRT 10 builder (FP16 flag)──▶ model.engine (sm_75-specific)
        ──TRTPredictor (execute_async_v3 on torch's stream)──▶ sliding_window_inference
```

## 2. ONNX export

| Choice | Value | Rationale |
|---|---|---|
| Input shape | $(4, 1, 96, 96, 96)$, no `dynamic_axes` | Equals the sliding-window call shape. Swin window partition, padding and masks fold into constants. One TensorRT optimisation profile |
| Exporter | TorchScript tracer (`dynamo=False`) | Mature for this model. `torch.export` gains nothing for a static graph |
| Opset | 18 | ≥ 17 gives native `LayerNormalization` |
| Constant folding | on | Folds shape arithmetic and attention masks |
| Weights | FP32 | Precision is decided at engine build |
| Device | CUDA when available | Tracing at batch 4 on CPU is RAM-heavy on Colab |
| I/O names | `image` → `logits` | |

Expected `TracerWarning`s point at MONAI's `if pad_d1 > 0 ...` and similar
branches. They are frozen to the traced shape, which is correct for a static
graph.

### Verification

1. `onnx.checker.check_model`.
2. The same $U(0,1)$ input (seeded) goes through PyTorch (cuDNN TF32 disabled
   for the reference; a no-op on T4) and ONNX Runtime CPU EP.
3. Pass iff every logit satisfies $\lvert z_\text{pt} - z_\text{ort}\rvert \le 10^{-3} + 10^{-3}\lvert z_\text{pt}\rvert$
   (the `numpy.allclose` rule). The script also reports max/mean absolute
   difference, the fraction within tolerance and argmax agreement.
4. A failure exits non-zero.

Random inputs exercise the arithmetic, not realistic activation statistics.
Real-data parity is measured downstream by per-backend Dice. On the
smoke-test model the max absolute difference was ~1e-5.

### 2.1 Parity gate

`python -m src.parity` (`src/parity.py`) is the full gate. The export check above is a quick sanity
check at export time; this gate judges every deployed backend and is meant to
block a change in CI.

**Reference.** The PyTorch checkpoint in true FP32 (`cudnn.allow_tf32` and
`cuda.matmul.allow_tf32` both off).

**Candidates and tiers.** Each backend is judged by the tier of its precision
(`parity.tiers` in `configs/default.yaml`):

| Backend | Tier | Default |
|---|---|---|
| `onnxruntime` (CPU EP) | `fp32` | yes |
| `tensorrt_fp16` | `fp16` | yes (skipped without CUDA/TensorRT) |
| `pytorch_autocast` | `fp16` on CUDA, `bf16` on CPU | opt-in: CPU bf16 is ~15x slower than fp32 on Apple Silicon |
| (TensorRT INT8) | `int8` | tier reserved, backend not yet implemented |

**Checks** (a check runs only if its tier sets the threshold):

| Metric | Meaning | Why |
|---|---|---|
| `fraction_within_tolerance` | share of logits with $\lvert z_\text{ref}-z\rvert \le \text{atol} + \text{rtol}\lvert z_\text{ref}\rvert$ | 1.0 for fp32 (a miss means a different graph); < 1 for reduced precision, where every logit legitimately moves |
| `cosine_similarity` | angle between flattened logit tensors | catches global distortions that elementwise counts dilute |
| `flip_rate` | share of voxels whose argmax changed | the task-level effect of numeric error |
| `confident_flips` | flips where the reference top-2 margin > `margin_threshold` | must be 0 in every tier: a flip on a confident voxel cannot be rounding noise |
| `foreground_dice_between_backends` | Dice of the two foreground masks | agreement on the organ itself, not the background |
| `dice_drop` (volumes only) | Dice vs label, reference minus candidate | the clinical metric must not regress |

A non-finite candidate output (NaN/Inf, typically an FP16 overflow) fails
immediately.

**Inputs.** `golden_batches` fixed-seed $U(0,1)$ batches of the exported shape
(no dataset needed), plus `--volumes N` whole validation volumes through
sliding-window inference (real activation statistics and Dice).

**Outputs.** `results/parity.json` (all metrics, flips by margin bucket,
checkpoint and ONNX SHA-256, engine build record, environment, tiers used) and `results/parity.md` (a
check table for a CI summary). Exit status 1 if any backend that ran fails,
or if no backend could run.

**Measured on CPU** (smoke-test model and a tiny untrained model; `tests/test_parity.py`):

| Candidate | fp32 tier | fp16 tier |
|---|---|---|
| ONNX Runtime | pass (max diff 1.0e-5, flip rate 1.3e-6 golden / 9.4e-6 volume) | |
| Weights, input and output rounded through float16 | **fail** (flip rate 3.1e-4) | pass (max diff 2.9e-3, 0 confident flips) |
| Output layer weights × 1.01 | **fail** | **fail** (flip rate) |

The fp16 thresholds and the int8 tier are starting values: calibrate them on
a T4 by running the gate over several healthy checkpoints and setting each
threshold with headroom above the worst observed value.

## 3. TensorRT engine build

`build_engine` (TensorRT 10 API, pinned `tensorrt-cu{12,13}==10.16.1.11` to match PyTorch's CUDA major):

1. `create_network(0)` (explicit batch is the only mode in TRT 10), then
   `OnnxParser`. Parse errors are surfaced verbatim.
2. Workspace pool limit `benchmark.workspace_gb` (4 GiB).
3. `BuilderFlag.FP16`. No `OBEY_PRECISION_CONSTRAINTS` and no per-layer
   overrides, so TensorRT chooses FP16 or FP32 per layer by tactic timing,
   subject to its own accuracy heuristics for normalisation layers.
4. `build_serialized_network`: graph fusion (conv + bias + activation,
   attention/MHA patterns where recognised, pointwise chains), precision
   selection and tactic auto-tuning on the build GPU.
5. Serialised to `checkpoints/swinunetr_spleen_fp16.engine`. Build time and
   size go to `benchmark.json → tensorrt_engine`.

The engine is built on first use and reused afterwards. `build_engine`
writes a build record next to it (`<engine>.json`: ONNX SHA-256, TensorRT
version, GPU name, precision); `ensure_engine` reuses the engine only if that
record matches the current ONNX file, TensorRT and GPU, and otherwise
rebuilds and records why (`rebuilt`, `rebuild_reason`). Both the benchmark
and the parity gate go through `ensure_engine`, so neither can test an engine
built from an older export. `--rebuild-engine` forces a rebuild. Engines are
specific to the GPU architecture and TensorRT version, so they are git-ignored.

**Not done:** INT8 (would need calibration data and an accuracy study),
explicit FP32 pinning of LayerNorm/softmax, dynamic-shape profiles, CUDA
graphs, timing-cache reuse.

## 4. Runtime integration

`TRTPredictor` presents the engine as a `Callable[[Tensor], Tensor]`, so it
plugs into `sliding_window_inference` unchanged:

- Named-tensor API: I/O names, shapes and modes come from the engine.
- Input and output buffers are pre-allocated `torch` CUDA tensors, bound once
  with `set_tensor_address`. No pycuda or cuda-python dependency.
- `execute_async_v3(torch.cuda.current_stream().cuda_stream)`: TensorRT work
  is ordered on PyTorch's stream, so no extra synchronisation is needed for
  correctness.
- Fixed batch: partial batches are zero-padded to 4 and the output is sliced.
  Larger batches are chunked. Results are `.clone()`d because the output
  buffer is reused.
- Per call there is one device-to-device copy into the input buffer and one
  clone out. The benchmark latency includes both (see
  [benchmarking.md](benchmarking.md)).

## 5. CUDA / TensorRT guard

The pipeline must run on macOS without NVIDIA software:

- No module-level `import tensorrt` anywhere.
- `_require_tensorrt()` checks `torch.cuda.is_available()`, then imports, and
  otherwise raises `RuntimeError` with an actionable message.
- `benchmark.py` imports `src.trt_utils` only inside the CUDA branch and
  records `{"status": "skipped", "reason": ...}` for unavailable backends.
- `test_tensorrt_module_imports_without_cuda` guards this on CPU CI.

## 6. FP16 risk analysis

FP16: 10-bit mantissa (unit roundoff $2^{-11} \approx 4.9 \times 10^{-4}$),
max 65,504, min normal $6.1 \times 10^{-5}$. Concerns specific to this
network:

- **LayerNorm variance** over 48–768 channels can overflow if computed
  naively in FP16. TensorRT's normalisation layer accumulates in FP32 when it
  is given a native `LayerNormalization` node, which is the reason for
  opset ≥ 17.
- **Softmax over 343 entries plus relative position bias** is a max-subtracted
  exponentiation. It is stable, with precision loss mostly in small
  probabilities.
- **Decoder logits** are compared by argmax on 2 channels, which is robust to
  small perturbations except near the decision boundary, i.e. at organ
  surfaces. Any FP16 Dice delta should therefore show up as surface voxel flips.

If `tensorrt_fp16` Dice falls measurably below `pytorch_fp32`, the next steps
are, in order: FP32 pinning of normalisation and softmax layers with
`OBEY_PRECISION_CONSTRAINTS`, then layer-wise output comparison with Polygraphy.

## 7. Troubleshooting

| Problem | Cause / fix |
|---|---|
| `ModuleNotFoundError: tensorrt` | `pip install tensorrt-cu<12 or 13>==10.16.1.11`, matching `torch.version.cuda` (the notebook does this) |
| Parser: unsupported op | Opset < 17 or stale `onnx`. Re-export |
| `Could not load ...engine` | Built on another GPU or TensorRT version. Normally caught by the build record; if the record is wrong, use `--rebuild-engine` |
| LayerNorm-in-FP16 overflow warning | Check Dice. If it is degraded, pin those layers to FP32 |
| Build OOM / killed | Lower `workspace_gb` |
