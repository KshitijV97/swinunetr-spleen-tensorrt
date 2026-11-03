# 10. GPUs and deployment

> **Summary.** A GPU is a throughput machine: thousands of simple cores and
> special **Tensor Cores** for matrix maths, fed by a memory system that is
> often the real bottleneck. The **roofline model** tells you which ops are
> limited by compute and which by memory bandwidth. To deploy, we:
> 1. **export** the PyTorch model to **ONNX**, a portable graph format (think
>    compiler IR), and verify that it computes the same thing;
> 2. **compile** it with **TensorRT**, an optimising compiler that fuses
>    operations, picks FP16 where safe and auto-tunes kernels for this exact
>    GPU (much like FFTW's planner).
>
> The result is an **engine**: a binary tied to one GPU model and one
> TensorRT version.

---

## 10.1 CPU vs GPU

| | CPU | GPU (NVIDIA T4) |
|---|---|---|
| Design goal | Low latency per thread | High throughput over many threads |
| Cores | A few to dozens, complex (out-of-order, big caches, branch prediction) | 2,560 simple CUDA cores in 40 **streaming multiprocessors** (SMs), plus 320 Tensor Cores |
| Execution | Threads run independently | Threads run in lock-step groups of 32 (**warps**), the same instruction on different data (SIMT) |
| Memory | Large caches hide DRAM latency | Enormous bandwidth (320 GB/s on the T4) and thousands of threads in flight to hide latency |

A neural network is mostly the same arithmetic on millions of numbers, which
is exactly the shape of work GPUs are built for.

## 10.2 The T4, the GPU this project targets

| Spec | Value |
|---|---|
| Architecture | Turing (compute capability 7.5) |
| Memory | 16 GB GDDR6, 320 GB/s |
| FP32 peak | ~8.1 TFLOPS |
| FP16 Tensor Core peak | ~65 TFLOPS |
| INT8 Tensor Core peak | ~130 TOPS |
| Power | 70 W |

The T4 is a low-power inference card, and it's what free Google Colab
offers. It has no BF16 or TF32 support, so "fast" means FP16 on the T4.

**Tensor Cores** are hardware units that compute small matrix
multiply-accumulates ($D = A \times B + C$ on little tiles) in one
operation, with FP16 inputs and FP32 accumulation. That's where the ~8×
FP16-vs-FP32 gap comes from. Convolutions and attention are re-expressed as
matrix multiplies to use them.

## 10.3 The roofline model: compute-bound or memory-bound?

Every operation has an **arithmetic intensity**: FLOPs performed per byte
moved to or from memory. A GPU has a peak compute rate $\pi$ and a peak
bandwidth $\beta$. The achievable rate is

```math
\text{attainable FLOP/s} = \min\left(\pi,\; \beta \times \text{intensity}\right)
```

The crossover ("ridge point") on the T4 is about $65 \times 10^{12} / 320 \times 10^9 \approx 200$
FLOP/byte for FP16, and about 25 for FP32.

| Operation | Intensity | Verdict on T4 |
|---|---|---|
| Elementwise add, GELU, residual (FP16) | < 1 FLOP/byte | **Memory-bound**: the cores mostly wait |
| LayerNorm, softmax | a few FLOP/byte | Memory-bound |
| Large matmul / convolution with many channels | hundreds or more | **Compute-bound**: the cores are busy |

This explains why **operator fusion** (doing several elementwise ops in one
pass over memory instead of one pass each) gives big speed-ups. It is the
main thing TensorRT does beyond using FP16. An EE equivalent is the
difference between processing a signal in one pipelined pass versus writing
it to RAM and reading it back between every stage.

### A back-of-envelope latency bound

This is an **estimate for intuition, not a measurement**. Published
comparisons put SwinUNETR ($F = 48$, 96³ input) at a few hundred GFLOPs per
window. Take ~350 GFLOPs. The 50-window volume from chapter 9 then needs
about 17.5 TFLOP. At the T4's FP32 peak that's at least ~2 s, and at the
FP16 Tensor Core peak at least ~0.3 s. Real kernels reach a fraction of peak,
and memory-bound ops add more. So expect real latencies of a few times these
bounds. Use them to sanity-check the measured numbers when they arrive. If
TensorRT ever appears faster than the FP16 bound, something is wrong with the
measurement.

## 10.4 GPUs are asynchronous

When Python calls `model(x)` on CUDA, PyTorch doesn't wait for the GPU. It
**enqueues kernels** onto a **stream** (a FIFO command queue) and returns
immediately. The GPU drains the queue in order. Think of a producer-consumer
queue, or a DMA command list.

Consequences:
- **Timing needs synchronisation.** Without `torch.cuda.synchronize()` you
  measure how fast Python can enqueue work, not how fast the GPU does it
  (chapter 11).
- **Launch overhead.** Each kernel launch costs a few microseconds of CPU
  time. A model made of many tiny ops can be limited by launches rather than
  by maths. Fusion helps here too.
- **Ordering across libraries.** If PyTorch and TensorRT work go on the
  same stream, they're automatically ordered with no extra syncing. Our
  TensorRT wrapper relies on this.

## 10.5 Step 1: export to ONNX

**ONNX** (Open Neural Network Exchange) is a file format that describes a
network as a graph of standard operators (`Conv`, `MatMul`, `Softmax`,
`LayerNormalization`, ...) plus the trained weights. It plays the role of a
compiler's **intermediate representation**, like LLVM IR: frameworks
produce it, and runtimes and compilers (ONNX Runtime, TensorRT, OpenVINO)
consume it. The **opset** version says which operator definitions are used.
We use opset 18, because LayerNorm became a single native op in opset 17.

**How export works: tracing.** `torch.onnx.export` runs the model once on an
example input and **records every tensor operation that actually executes**,
like recording a macro. Consequences:

- Python control flow is **frozen**. An `if` in the model is recorded as
  whichever branch ran. MONAI's Swin code has `if padding_needed:` branches
  that depend on the input size, and the exporter prints `TracerWarning`s
  about them. Because we export **one fixed shape**, the frozen branch is
  correct for every call we'll ever make. That's one reason for the fixed
  shape.
- Shapes become constants, so window partitioning, padding sizes and
  attention masks get pre-computed ("constant folding"). The graph is
  simpler, and TensorRT can specialise kernels for exactly one shape.

We export with input shape `(4, 1, 96, 96, 96)`, exactly what
sliding-window inference feeds the model (chapter 9).

### Verifying the export

Never trust a conversion blindly. `verify()` in
[`src/export_onnx.py`](../../src/export_onnx.py):

1. checks the file is structurally valid (`onnx.checker`);
2. feeds the same random input to PyTorch and to **ONNX Runtime** (an
   independent implementation);
3. requires every output to satisfy $\lvert a - b\rvert \le 10^{-3} + 10^{-3}\lvert a\rvert$.

Why not demand exact equality? **Floating-point addition isn't
associative.** In FP32, $(10^8 + 1) - 10^8 = 0$ but $(10^8 - 10^8) + 1 = 1$,
because near $10^8$ the representable numbers are 8 apart. Two runtimes that
sum the same numbers in different orders (and they do, for speed) give
slightly different answers. Both are "correct". The tolerance says how
different is acceptable.

## 10.6 Step 2: compile with TensorRT

**TensorRT** is NVIDIA's optimising compiler and runtime for neural
networks. The **builder** takes the ONNX graph and:

1. **Parses** it into its own network representation.
2. **Optimises the graph:** fuses layers (conv + bias + activation become one
   kernel, chains of elementwise ops merge, attention patterns map to fused
   kernels), folds constants and removes dead nodes.
3. **Chooses precision per layer.** With the FP16 flag it may run each layer
   in FP16 or FP32. It prefers FP16 for speed but keeps sensitive layers in
   FP32 where it sees a need.
4. **Auto-tunes.** For each layer it **times several kernel implementations**
   (called *tactics*) on the actual GPU and keeps the fastest. If you know
   FFTW's planner, which benchmarks FFT algorithms on your machine and saves
   "wisdom", it's the same idea.
5. **Plans memory**, re-using buffers between layers whose lifetimes don't
   overlap.
6. **Serialises** the result into an **engine** file.

This takes minutes on a T4, because of all the timing.

**Engines aren't portable.** An engine is tuned for one GPU architecture and
one TensorRT version, like a binary compiled with `-march=native` and
profile-guided optimisation for one machine. That's why `.engine` files are
git-ignored and rebuilt wherever they run. For a robot with a Jetson Orin,
you'd rebuild on the Jetson.

### The FP16 risk

FP16 overflows at 65,504 (chapter 8). In this network the risky places are
**LayerNorm** (a variance, i.e. a sum of squares) and **softmax**
(exponentials). Exporting LayerNorm as a single op lets TensorRT use its
dedicated normalisation layer, which accumulates in higher precision.
Whether accuracy survives is an empirical question. That's why the benchmark
reports **Dice for every backend**, and a Dice drop under FP16 would be
reported, not hidden.

## 10.7 Step 3: run the engine from Python

[`src/trt_utils.py`](../../src/trt_utils.py) → `TRTPredictor` makes the
engine look like a PyTorch model (a callable from tensor to tensor), so it
drops into the same sliding-window code:

- It allocates **input and output buffers as PyTorch CUDA tensors** once, and
  hands their raw GPU addresses to TensorRT (`set_tensor_address`). That's
  like passing a raw pointer to a C library over FFI. No extra CUDA Python
  package is needed.
- It launches the engine on **PyTorch's current stream**
  (`execute_async_v3`), so ordering with surrounding PyTorch work is
  automatic (§10.4).
- The engine only accepts **batch 4**. Sliding-window inference can hand it a
  last batch of 1–3 windows, so the wrapper zero-pads to 4 and slices the
  result.

## 10.8 Making it work on a laptop without a GPU

TensorRT can't even be installed on a Mac. The code never imports it at the
top of a file. It imports it **inside functions, after checking that CUDA
exists**, and otherwise raises a clear error. The benchmark records TensorRT
as "skipped: needs CUDA" rather than crashing or inventing a number, and a
unit test checks this on every CI run. This is the same pattern as an
optional plugin or a feature flag around a platform-specific dependency.

## 10.9 The three backends we compare

| Backend | What runs | Why include it |
|---|---|---|
| PyTorch FP32 | The trained model, 32-bit everywhere | Baseline |
| PyTorch AMP | The same model with FP16 autocast | Isolates the gain from FP16 Tensor Cores alone |
| TensorRT FP16 | The compiled engine | FP16 **plus** fusion, auto-tuning and no Python per op |

The AMP row is what makes the comparison honest. If TensorRT is only a
little faster than AMP, most of the speed-up came from FP16, and only the
remaining gap is due to compilation.

---

**In the code**
- Export and verification: [`src/export_onnx.py`](../../src/export_onnx.py).
- Engine build and runtime wrapper: [`src/trt_utils.py`](../../src/trt_utils.py).
- Where the three backends are run:
  [`src/benchmark.py`](../../src/benchmark.py) → `main`.
- The compact version: [deployment reference](../reference/deployment.md).

**Check yourself**
1. Is LayerNorm compute-bound or memory-bound on a T4? Why does that make
   fusion attractive?
2. Why does exporting with a fixed input shape make the `TracerWarning`s
   harmless?
3. Why can't you build a TensorRT engine on Colab and use it on a Jetson?
4. If TensorRT FP16 is 3× faster than FP32 and AMP is 2.5× faster, what does
   that say about where the speed-up comes from?

Next: [11. Benchmarking honestly →](11-benchmarking-honestly.md)
