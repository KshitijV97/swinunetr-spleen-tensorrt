# 11. Is the fast model still the same model? (the parity gate)

> **Summary.** Chapter 10 turned the PyTorch model into an ONNX file and then
> into a TensorRT engine. Each conversion could quietly change what the model
> computes. The **parity gate** is an automatic test that runs the original
> and each converted version on the same inputs, compares the outputs, and
> returns **pass** or **fail**. Because the outputs can never match exactly,
> it uses **looser rules for lower precision** (FP16) than for full precision
> (FP32). Its one rule that never relaxes is that a voxel the model was **sure**
> about must not change its answer. A failure stops the pipeline, just like a
> failing unit test.

---

## 11.1 The question, in one sentence

> *After converting the model, does it still give the same answers?*

The first idea is to check that the outputs are equal. That fails for two
reasons, and both are worth understanding before anything else.

**Reason 1: the order of adding changes the result.** In floating point,
`(a + b) + c` and `a + (b + c)` can differ in the last digit. Try it in
Python:

```python
>>> (0.1 + 0.2) + 0.3
0.6000000000000001
>>> 0.1 + (0.2 + 0.3)
0.6
```

A convolution sums thousands of products. PyTorch, ONNX Runtime and
TensorRT each add them in a different order, so even a perfect conversion
gives slightly different numbers. That is not a bug. It's arithmetic.

**Reason 2: FP16 stores fewer digits.** FP32 keeps about 7 significant
digits and FP16 about 3 (chapter 8). FP16 can store numbers between 4 and 8
only in steps of 1/256 ≈ 0.0039. So a logit of `5.3` is stored as
`5.30078125`. Every value moves a little. Multiply that by about 50 layers
and the final logits typically differ by 0.01 or more.

So the real question becomes:

> *Are the differences small enough to be harmless, and are they the kind
> of difference that rounding causes, rather than the kind a bug causes?*

The rest of the chapter is about telling those two apart with numbers.

## 11.2 The reference: what counts as "the original"

Every comparison is against **the PyTorch model running in true FP32**.

The word "true" matters. Newer NVIDIA GPUs (Ampere onwards) have a mode
called **TF32** that quietly runs "FP32" matrix maths with only about 3
digits, to go faster. If that mode were on, the reference would be noisy
itself, like calibrating a ruler against a ruler that is also slightly off. The gate turns
TF32 off while computing the reference and switches it back afterwards.

> **In the code:** `fp32_reference()` in [`src/parity.py`](../../src/parity.py).

## 11.3 What the gate measures

The network outputs, for every voxel, one **logit** per class: a raw score
for "background" and a raw score for "spleen" (chapter 4). The higher score
wins. Call the reference logits *ref* and the converted model's logits *out*.
The gate computes six numbers from them. Each one catches a different kind of
problem.

### (a) Fraction of logits within tolerance

For each logit, is the gap small enough?

$$\lvert \text{ref} - \text{out} \rvert \;\le\; \text{atol} + \text{rtol}\cdot\lvert\text{ref}\rvert$$

`atol` is an **absolute** allowance and `rtol` is a **relative** one (a
percentage of the value). It's the same formula as NumPy's `np.allclose`.

*Worked example.* The reference logit is 5.0.
- FP32 rules (atol 0.001, rtol 0.001): allowed gap = 0.001 + 0.001 × 5 = **0.006**.
- FP16 rules (atol 0.05, rtol 0.01): allowed gap = 0.05 + 0.01 × 5 = **0.10**.

If `out` is 5.04, it passes under FP16 rules and fails under FP32 rules. The
gate reports the share of all logits that pass: 1.0 means all of them.

### (b) Cosine similarity

Treat all the logits as one very long list of numbers (a vector), for both
*ref* and *out*, and measure the angle between the two vectors. A value of 1.0
means they point the same way. This catches a small distortion spread across
the whole output, which check (a) can miss if each individual gap is just
under the limit.

It has one blind spot: multiplying every logit by 2 leaves the angle
unchanged. That's fine, because other checks catch it. No single number is
trusted on its own.

### (c) Flip rate

For each voxel, does the converted model pick the **same class**? A *flip*
is a voxel where the answer changed (background ↔ spleen). The flip rate is
the share of voxels that flipped.

This is the check closest to what we actually care about, because a
surgeon sees the label map, not the logits. Under FP16 rules the limit is
1 in 1,000. One 96³ window has about 885,000 voxels, so that allows up to
about 885 flips per window.

### (d) Confident flips: the key idea

Not all flips are equal. Look at two voxels:

| Voxel | Reference logits (background, spleen) | Gap between top two (the **margin**) | Model's feeling |
|---|---|---|---|
| A | (2.00, 2.03) | 0.03 | "It could be either." Usually right on the organ's edge |
| B | (−3.0, 4.0) | 7.0 | "Definitely spleen." |

For a voxel to flip, the numeric error has to be bigger than its margin.
FP16 errors are about 0.01 to 0.05. That can easily flip voxel A, and it
doesn't matter, because the model was unsure anyway. Voxel B can't flip from
rounding, because that would need an error of 7. **If B flips, something is
broken.**

So the gate splits flips by margin. A voxel counts as **confident** if its
margin is above 1.0. For two classes, a margin of 1.0 means the model gives
the winner about a 73 % probability. The rule:

> **Confident flips must be exactly 0, under every set of rules.**

This is the most useful single idea in the gate. It separates "rounding
noise at the edges" from "the model is computing something different",
without having to guess a percentage.

There is a link back to training here. A model that ends up with **large
margins** on most voxels is easier to deploy, because rounding can only flip
the few voxels near the boundary.

### (e) Spleen agreement between the two models

Take each model's spleen mask (voxels where spleen wins) and compute the
Dice overlap between **the two masks** (Dice is explained in chapter 9).
The flip rate counts all voxels, and most are background, so a model that
flipped half the spleen could still have a tiny flip rate. This check
looks only at the organ.

### (f) Dice drop (whole scans only)

On real scans with a radiologist's outline, compute Dice against that outline
for the original and for the converted model. **Dice drop** is the original
minus the converted. Under FP16 rules it must be at most 0.005. So
0.950 → 0.946 passes, and 0.950 → 0.940 fails.

### (g) NaN or infinity: immediate fail

FP16's largest number is 65,504. If a value inside the network goes past
that, it becomes infinity, and the output can then become NaN ("not a
number"). Any NaN or infinity in the output fails the gate on the spot.

## 11.4 Different rules for different precisions ("tiers")

The same set of thresholds can't fit both FP32 and FP16:

- Thresholds strict enough to catch a bug in an FP32 conversion would fail
  **every** healthy FP16 engine.
- Thresholds loose enough for FP16 would let real FP32 bugs through.

So each precision gets its own set, called a **tier**, in
[`configs/default.yaml`](../../configs/default.yaml) under `parity.tiers`:

| Tier | Used for | In plain words |
|---|---|---|
| `fp32` | ONNX Runtime | Every logit must be within ±0.1 %. Almost nothing may flip. Any real difference means the conversion changed the graph. |
| `fp16` | TensorRT FP16 | Logits may move by a few percent. At most 1 in 1,000 voxels may flip, and none of the confident ones. Dice may drop by at most 0.005. |
| `bf16` | PyTorch on CPU with bfloat16 | Like `fp16` but looser, because bfloat16 keeps even fewer digits. |
| `int8` | (reserved for later) | Thresholds are written down but no INT8 engine exists yet. |

**How we know the tiers work.** The tests run three fake "conversions" of a
small model on a CPU:

| Fake conversion | `fp32` tier | `fp16` tier | What it shows |
|---|---|---|---|
| Identical model | pass | pass | No false alarm |
| Every weight and value rounded through FP16 | **fail** | pass | Honest FP16 noise is accepted by the FP16 rules and rejected by the strict ones. That is the reason tiers exist. |
| Output layer multiplied by 1.01 (a 1 % bug) | **fail** | **fail** | Even a small genuine bug is caught under the loose rules |

The last row matters most. A gate that never fails proves nothing. You have
to show it **does** fail on a realistic bug.

## 11.5 Two kinds of test input

| | Golden inputs | Whole scans (`--volumes N`) |
|---|---|---|
| What | 3 batches of random numbers, always the same (fixed seed), in the exact shape the engine expects | Real validation scans, run through the full sliding-window pipeline (chapter 9) |
| Cost | Seconds, no dataset needed | Minutes, needs the data |
| Catches | A broken conversion: wrong operation, missing layer, bad padding | Problems that only real data triggers (e.g. FP16 overflow on unusual intensities), plus the real Dice drop |
| EE analogy | Test vectors on the bench | A field test |

Random noise doesn't look like a CT scan, so the values inside the network
won't look like real ones either. That's why the scan-level check exists.

## 11.6 What makes it "automated"

A test is only a **gate** if nobody has to remember to look at it.

1. **Pass/fail is a process exit code.** `python -m src.parity` exits with
   status 0 on pass and 1 on fail, like `pytest`. Any CI system can block a
   merge on it. It also fails if *no* backend could run, so "nothing was
   tested" never shows up as a pass.
2. **It runs in CI on every push** (`.github/workflows/tests.yml`): train a
   tiny model, export it, run the gate, run the benchmark. The CI machine has
   no GPU, so there only the ONNX Runtime check actually runs. TensorRT is
   recorded as "skipped: needs CUDA" (§11.7).
3. **It writes a report**: `results/parity.json` (every number) and
   `results/parity.md` (a pass/fail table you can paste into a pull request).
4. **It records exactly what it tested.** The report includes a
   **SHA-256 hash** (a fingerprint of the file's bytes) of the checkpoint and
   of the ONNX file. If two reports show the same hash, they tested the same
   file. If the hash differs, the file changed.

### The stale-engine problem (and its fix)

Building a TensorRT engine takes minutes, so it's saved to disk and reused.
Any cache raises the same question: **how do you know the cached copy is
still up to date?**

The original code reused the engine whenever the file existed. Picture this
sequence:

1. Export model v1 to ONNX and build an engine. The gate passes.
2. Retrain, export v2 to ONNX, and forget `--rebuild-engine`.
3. The gate compares PyTorch **v2** with the TensorRT engine built from
   **v1**. It might pass or fail, but either way it's testing the wrong thing.

**The fix** works the way `make`, `ccache` and Bazel do it: tie the cached
output to a fingerprint of its input. When the engine is built, a small
record is written next to it (`model.engine.json`):

```json
{ "onnx_sha256": "3f9a…", "tensorrt": "10.16.1.11", "gpu": "Tesla T4", "precision": "fp16" }
```

Before reusing the engine, the code compares that record with the current
ONNX file's hash, the installed TensorRT version and the GPU name. If
anything differs, or the record is missing, it rebuilds and says why. The
benchmark uses the same function, so it can't time an old engine either.

> **In the code:** `ensure_engine()` and `engine_stale_reason()` in
> [`src/trt_utils.py`](../../src/trt_utils.py). The tests at the bottom of
> [`tests/test_parity.py`](../../tests/test_parity.py) check each "changed"
> case on a CPU, without TensorRT.

## 11.7 What it does not do yet (be upfront about these)

- **TensorRT is not checked in CI.** CI has no GPU. A real setup needs a
  GPU machine in CI with the **same GPU model and TensorRT version** as
  production, because an engine built on one GPU doesn't run on another.
- **The thresholds are reasoned, not measured.** They come from how many
  digits FP16 keeps, not from data. The proper way to set them: build
  several engines from good checkpoints, record the worst value of each
  metric, and set each threshold a bit beyond it.
- **"Zero confident flips" is a hard count.** On millions of voxels, one
  voxel with a margin of 1.01 could fail an otherwise perfect run. A small
  allowed rate, or a second, higher margin cut-off, would be more robust.
- **Golden inputs are random noise**, as §11.5 says. In CI, even the
  "whole scan" check runs on synthetic scans.
- **The export step has its own, simpler check.** `verify()` in
  [`src/export_onnx.py`](../../src/export_onnx.py) is a quick check
  at export time. It should eventually reuse the gate's comparison code
  so there's one definition of "close enough".
- **No real-GPU results yet.** The full run on a T4 hasn't been done
  (see the main README).

## 11.8 Explaining it out loud (e.g. in an interview)

**The 30-second version:**

> "When you convert a model for deployment, the numbers can't match exactly,
> because each runtime adds in a different order and FP16 keeps fewer
> digits. So I built an automatic gate that runs the original PyTorch model
> and each converted version on the same inputs and compares them. It uses
> strict rules for FP32 and looser rules for FP16. The key rule is that no
> voxel the model was confident about may change its answer: rounding can
> only flip voxels the model was unsure about, so a confident flip means a
> real bug. It exits non-zero on failure, so CI blocks the change, and it
> records hashes of exactly what it tested."

**Questions you're likely to get, with short answers:**

| Question | Answer in plain words |
|---|---|
| Why not just check the outputs are equal? | They can't be: different addition order, and FP16 keeps about 3 digits (§11.1). |
| Why not just compare Dice? | Dice on 8 scans is noisy and doesn't show *where* something broke. Logit checks catch a broken conversion before it shows up in Dice. |
| How do you tell a bug from rounding? | Margins. Rounding can only flip voxels whose top two scores are close. A flip on a voxel with a big gap is a bug (§11.3 d). |
| How do you know the gate isn't too loose? | The tests show a 1 % change to one layer fails even the FP16 rules (§11.4). |
| How did you choose the thresholds? | From FP16's precision, as a starting point. I'd calibrate them on real engines next (§11.7). |
| What if the engine is out of date? | It's fingerprinted with the ONNX hash, TensorRT version and GPU, and rebuilt on any mismatch (§11.6). |
| Does CI test TensorRT? | Not yet: it needs a GPU machine. Today CI checks the ONNX conversion and the gate's own logic (§11.7). |

---

**In the code**
- The gate: [`src/parity.py`](../../src/parity.py). `compare` computes the
  metrics, `judge` applies a tier, and `main` runs everything and writes the report.
- The tiers: `parity:` in [`configs/default.yaml`](../../configs/default.yaml).
- Engine caching and staleness: [`src/trt_utils.py`](../../src/trt_utils.py).
- The tests: [`tests/test_parity.py`](../../tests/test_parity.py).
- The compact version: [deployment reference §2.1](../reference/deployment.md#21-parity-gate).

**Try it** (on a laptop, no GPU):

```bash
python -m src.train --smoke-test --synthetic
python -m src.export_onnx --smoke-test
python -m src.parity --smoke-test --synthetic --volumes 1
```

Open `results/smoke/parity.md` and find the flip rate and the confident
flips for ONNX Runtime.

**Check yourself**
1. A voxel's reference logits are (0.40, 0.42). An FP16 engine gives
   (0.43, 0.41). Is that a flip? Is it a confident flip? Should the gate care?
2. Why can't one set of tolerances serve both ONNX Runtime and TensorRT FP16?
3. Cosine similarity is 1.0, but every logit is exactly twice the reference.
   Which other check catches this?
4. Why does the gate fail when *no* backend could run, instead of passing?
5. You retrain, re-export, and run the gate without `--rebuild-engine`.
   What happens now, and what happened before the build record was added?

Next: [12. Benchmarking honestly →](12-benchmarking-honestly.md)
