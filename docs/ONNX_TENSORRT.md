# ONNX export and TensorRT optimisation, explained

Code: [`src/export_onnx.py`](../src/export_onnx.py), [`src/trt_utils.py`](../src/trt_utils.py),
[`src/benchmark.py`](../src/benchmark.py).

---

## 1. Why bother?

A model that segments well in a training notebook is not yet a model you can
put in a surgical robot's perception loop. Deployment cares about
**latency** (how long one scan takes), **predictability** (no Python overhead
spikes), **memory**, and **portability** (C++ runtimes, embedded GPUs such as
NVIDIA Jetson). The standard NVIDIA path is:

```
PyTorch model ──torch.onnx.export──▶ model.onnx ──TensorRT builder──▶ model.engine ──TensorRT runtime──▶ predictions
   (training)                       (portable graph)                 (GPU-specific, optimised)
```

## 2. Step 1: export to ONNX

[ONNX](https://onnx.ai) is a framework-neutral file format that describes a
network as a graph of standard operators (Conv, MatMul, Softmax,
LayerNormalization, ...) plus the trained weights.

```bash
python -m src.export_onnx               # checkpoints/best.pt → checkpoints/swinunetr_spleen.onnx
python -m src.export_onnx --smoke-test  # the tiny smoke-test model
```

Design decisions:

| Decision | Reason |
|---|---|
| **Fixed input shape** `(sw_batch_size, 1, 96, 96, 96)` = `(4, 1, 96, 96, 96)` | Sliding-window inference only ever calls the network with this shape. Swin's window partitioning computes reshapes from the input size, and with fixed shapes those become constants, which keeps the graph simple. TensorRT can then tune kernels for exactly one shape |
| **Opset 18** | Opset ≥ 17 has a native `LayerNormalization` op (used throughout the Swin blocks) instead of a fragile chain of reduce/sub/div ops |
| **TorchScript exporter** (`dynamo=False`) | The mature tracer-based exporter handles SwinUNETR reliably. The newer `torch.export` path is not needed for a fixed-shape model |
| **FP32 weights in the ONNX file** | Precision is TensorRT's decision at build time, so the ONNX file stays a faithful copy of the PyTorch model |
| **Exported on GPU when available** | Tracing the 62M-parameter model at batch 4 on CPU needs a lot of RAM on Colab |

During export you will see `TracerWarning`s pointing at lines in MONAI's
Swin code (`if pad_d1 > 0 ...`). The tracer records Python `if` statements
with the branch taken for *this* input shape. Because the shape is fixed, that
is exactly what we want, and the warnings are expected.

### Verification

Export is not trusted blindly. `verify()`:

1. runs `onnx.checker.check_model` (is the file structurally valid?);
2. feeds the **same random input** to PyTorch and to ONNX Runtime (CPU);
3. requires every logit to satisfy `|pytorch − onnx| ≤ atol + rtol·|pytorch|`
   with `atol = rtol = 1e-3`, and reports the max/mean absolute difference and
   the fraction of voxels whose predicted class (argmax) agrees.

Small differences are normal: the runtimes add numbers in different orders,
and floating-point addition is not associative. On NVIDIA GPUs, cuDNN may
also use **TF32** for "FP32" convolutions (only 10 mantissa bits), so
verification turns TF32 off for the PyTorch reference.

The report is saved to `results/onnx_export.json`. On the local smoke-test
model the max absolute difference was ~1e-5.

## 3. Step 2: build a TensorRT engine

[TensorRT](https://developer.nvidia.com/tensorrt) is NVIDIA's inference
compiler. `build_engine()` in `src/trt_utils.py`:

1. parses the ONNX file into a TensorRT network (`trt.OnnxParser`);
2. sets a workspace memory limit (`benchmark.workspace_gb`, default 4 GB of
   scratch memory the builder may use while trying kernels);
3. enables the **FP16** builder flag;
4. calls `build_serialized_network`, which
   - **fuses layers** (e.g. convolution + bias + activation become one kernel;
     attention sub-graphs become fused kernels),
   - **chooses precisions** layer by layer: FP16 where it is safe, FP32
     where needed,
   - **auto-tunes**: times several kernel implementations ("tactics") for each
     layer on *this* GPU and keeps the fastest;
5. writes the result to `checkpoints/swinunetr_spleen_fp16.engine`.

This takes a few minutes on a T4. The engine is built automatically the first
time you run `python -m src.benchmark` and reused afterwards (add
`--rebuild-engine` to force a rebuild).

> **Engines are not portable.** An engine is tied to the GPU model and the
> TensorRT version that built it. That is why `.engine` files are
> git-ignored: always rebuild on the target machine.

### What FP16 means for accuracy

FP16 has a 10-bit mantissa (about 3 decimal digits) and a maximum value of
65,504. Most convolution and matmul layers are fine with it, and the T4's
Tensor Cores run FP16 much faster than FP32. Softmax, normalisation and large
reductions are where overflow or precision loss can show up. The benchmark
therefore reports **Dice for every backend**: if TensorRT FP16 Dice matches
PyTorch FP32 Dice to 3–4 decimal places, the precision change is harmless for
this task. If it drops, that is a finding worth reporting, not hiding.

## 4. Step 3: run the engine from Python

`TRTPredictor` makes the engine behave like a PyTorch model, so it can be
dropped into MONAI's `sliding_window_inference` unchanged:

- It uses the **TensorRT 10 named-tensor API**: it looks up the input/output
  tensor names, pre-allocates PyTorch CUDA tensors of the right shapes, and
  passes their raw device pointers with `context.set_tensor_address`.
- `execute_async_v3` is launched on **PyTorch's current CUDA stream**, so
  TensorRT work and PyTorch work are correctly ordered without extra
  synchronisation code or extra packages (no pycuda / cuda-python).
- **Fixed batch handling:** the engine only accepts batch 4. Sliding-window
  inference may hand it a final batch of 1–3 windows, so `TRTPredictor`
  zero-pads up to 4 and slices the output; larger batches are chunked.

## 5. Why TensorRT is guarded

The project must run on a Mac with no NVIDIA GPU, where `tensorrt` cannot
even be installed. So:

- nothing imports `tensorrt` at module level;
- `src/trt_utils.py` imports it **inside** functions, after checking
  `torch.cuda.is_available()`, and raises a clear `RuntimeError` otherwise;
- `src/benchmark.py` only imports `src.trt_utils` inside the
  `if device.type == "cuda"` branch, and records TensorRT as
  `"skipped: needs CUDA"` on other machines;
- a unit test (`test_tensorrt_module_imports_without_cuda`) checks the module
  imports cleanly on a CUDA-less machine.

## 6. Common problems

| Problem | Cause / fix |
|---|---|
| `ModuleNotFoundError: tensorrt` on Colab | Run `pip install -r requirements-colab.txt` (it pins `tensorrt-cu12==10.16.1.11`) |
| `TensorRT could not parse ...onnx` with an unsupported-op error | Make sure the opset is ≥ 17. Re-export after upgrading `onnx` if needed |
| `Could not load ...engine` | The engine was built on a different GPU or TensorRT version. Run with `--rebuild-engine` |
| TensorRT warns that a LayerNorm runs in FP16 and may overflow | Check the Dice. If it matches FP32, it is harmless. If not, a next step is to force those layers to FP32 (`layer.precision`, with `BuilderFlag.OBEY_PRECISION_CONSTRAINTS`) |
| Engine build is killed / out of memory | Lower `benchmark.workspace_gb` |

## 7. Where this could go next

- **INT8 quantisation** with a calibration set (more speed, needs care for accuracy).
- **Dynamic batch dimension** with a TensorRT optimisation profile.
- **Torch-TensorRT** (`torch.compile(backend="tensorrt")`) as an alternative path.
- **Jetson Orin** deployment: rebuild the engine on the device itself.
- **Triton Inference Server** to serve the engine behind an API.
