# 13. Software engineering of this repo

> **Summary.** This is the chapter where your existing experience applies
> most directly. ML code has the usual engineering needs plus a few of its
> own:
> - results must be **traceable** to exact settings and versions;
> - **randomness** must be controlled;
> - **hardware differences** (Mac vs NVIDIA GPU) must be handled explicitly;
> - tests must run **without** the 1.5 GB dataset or a GPU.
>
> Here's how the repo does each, and how to extend it.

---

## 13.1 Layout and data flow

```
configs/default.yaml   ─┐
                        ├─▶ src/train.py ──best.pt──▶ src/export_onnx.py ──.onnx──▶ src/benchmark.py
MSD data ─▶ src/data.py ┘                     └──────────────best.pt──────────────────▶ │
                                                                         src/trt_utils.py (engine)
results/: training_history.json, onnx_export.json, benchmark.json, benchmark.md
```

| Module | Responsibility |
|---|---|
| `utils.py` | Config loading and merging, path resolution, device choice, seeding, environment capture |
| `data.py` | Download + checksum, split, preprocessing pipelines, DataLoaders, synthetic data |
| `model.py` | Build/validate/load the network |
| `evaluation.py` | Sliding-window inference and Dice, **shared** by training and benchmarking |
| `train.py`, `export_onnx.py`, `benchmark.py` | Three CLIs, one per pipeline stage |
| `trt_utils.py` | Everything TensorRT, isolated behind a guard |

Each stage communicates through **files** (`best.pt`, `.onnx`, JSON
results), not shared state. You can rerun any stage alone, inspect its
output, or run it on a different machine. It's a pipeline of pure-ish
functions with artefacts in between.

## 13.2 Configuration: one source of truth, layered

Settings come from three layers, each overriding the previous one:

1. [`configs/default.yaml`](../../configs/default.yaml): every
   hyperparameter, commented.
2. The `smoke_test:` block at the bottom of the same file, deep-merged on top
   when `--smoke-test` is passed. Only the keys that change are listed, so
   "what's different about the smoke test" is visible at a glance.
3. Command-line flags (`--epochs 80`) for one-off changes.

`deep_update` merges nested dictionaries key by key, so overriding
`data.roi_size` doesn't wipe out `data.pixdim`. This is the same pattern as
layered config in most server frameworks (defaults → environment file →
flags).

## 13.3 Provenance: every number knows where it came from

- The **fully merged config** is saved inside every checkpoint and every
  results JSON.
- `environment_info()` records Python, platform, PyTorch, MONAI, CUDA, cuDNN
  and GPU name, and the benchmark adds the TensorRT version.
- Export and benchmark **rebuild the model from the config stored in the
  checkpoint**, not from the current YAML. If someone edits `feature_size`
  after training, the old checkpoint still loads with the right
  architecture.

The software analogy is embedding the git SHA and build flags in a binary.
For ML results it's the difference between a number and a claim.

## 13.4 Determinism and its limits

`set_seed` seeds Python's `random`, NumPy, PyTorch and MONAI. The train/val
split additionally sorts the file list before shuffling, so it doesn't depend
on filesystem order, and a test checks this by shuffling the input. What
isn't guaranteed: bit-identical GPU results, because some CUDA kernels use
atomic additions in non-deterministic order (chapter 8). The docs say so
rather than overclaiming.

## 13.5 Handling hardware differences explicitly

| Situation | How the code handles it |
|---|---|
| Choosing a device | `get_device("auto")`: CUDA, else Apple MPS, else CPU. Overridable with `--device` |
| Mixed precision on a Mac | Disabled automatically. `GradScaler` and `autocast` become no-ops |
| TensorRT on a Mac | Never imported at module level. Imported inside functions after a CUDA check, with an actionable `RuntimeError` otherwise |
| Benchmark without CUDA | AMP and TensorRT rows are recorded as `skipped` with a reason, **never estimated** |
| Colab vs local dependencies | Two pinned requirement files. Colab's keeps Colab's own CUDA build of PyTorch |

The guarded import is the key trick: optional, platform-specific
dependencies behave like plugins, so the core code runs everywhere.

## 13.6 Testing ML code without data or a GPU

Testing ML code is awkward. "Is the model good?" can't be a unit test, and
real data is large and licensed. The repo tests the things that *can* be
pinned down, using **synthetic NIfTI files** generated on the fly (a bright
ellipsoid "spleen" in an ellipsoid "body"):

| Test | Catches |
|---|---|
| Split is deterministic, disjoint, 33/8 | Data leakage between train and validation; irreproducible splits |
| Train transforms give the right crop count/shape, range, binary labels | Broken preprocessing (the most common silent ML bug) |
| Validation transforms keep image and label aligned | Misaligned masks, which give Dice ≈ 0 with no error message |
| Output shape equals input shape | Architecture/config mismatch |
| Sliding window works on odd sizes | Edge cases in inference |
| Bad `roi_size` / `feature_size` are rejected | Confusing crashes deep inside MONAI |
| Default model has ~62 M params | Accidental config changes |
| TensorRT module imports without CUDA | Regressions in the guard |

On top of unit tests, **CI** (GitHub Actions) runs the whole pipeline
end-to-end on synthetic data on a CPU runner, train → export → benchmark,
for every push. It proves the code *runs*, not that the model is *good*.
Model quality is measured by the Colab run and reported with caveats.

## 13.7 What's deliberately not in git

- The dataset: large, and licensed separately. Anyone can re-download it,
  and the checksum is verified.
- Checkpoints, ONNX files and TensorRT engines: large binaries, and engines
  are machine-specific anyway.
- `results/smoke/`: smoke-test numbers aren't real results, so they shouldn't
  sit next to real ones.

Only small, human-readable results (`results/*.json`, `benchmark.md`, the
training-curve PNG) get committed after the real run.

## 13.8 Honesty as an engineering requirement

A few rules run through the codebase and docs:

- Unavailable measurements are **skipped and labelled**, never filled in.
- README results say **TBD** until a real run produces them.
- Limitations are listed up front, not buried.
- Every metric is defined precisely (what's included, what isn't).

These matter more in ML than in most software, because a plausible-looking
wrong number is easy to produce and hard to detect.

## 13.9 How to extend the project

| You want to... | Change |
|---|---|
| Try a smaller model | `--feature-size 24` (and say so when reporting) |
| Train on another MSD task | `data.task`, plus `hu_window`, `pixdim`, `out_channels` as needed |
| Add a backend (e.g. ONNX Runtime GPU, `torch.compile`) | Write a callable `(n, 1, 96, 96, 96) → (n, 2, 96, 96, 96)` and pass it to `run_backend` |
| Report native-space Dice | Add MONAI's `Invertd` to the shared evaluation path so every backend gets it |
| Add a surface metric | Extend `DiceScorer` (or add a sibling) in `evaluation.py` with MONAI's `HausdorffDistanceMetric` |

**Suggested exercises**, roughly in order of difficulty:
1. Run the smoke test and read every line of its log against chapter 8.
2. Change `overlap` to 0.25 in the smoke benchmark and compare latency.
3. Add a test that the Dice of a mask with itself is 1 and with its
   complement is 0.
4. Add largest-connected-component post-processing behind a flag, and
   measure its effect on the real validation set.
5. Add an ONNX Runtime backend to the benchmark.

---

**In the code**
- [`src/utils.py`](../../src/utils.py): config, devices, seeding,
  provenance.
- [`tests/`](../../tests/) and [`.github/workflows/tests.yml`](../../.github/workflows/tests.yml).
- The compact version: [code and reproducibility reference](../reference/code-and-reproducibility.md).

Next: [14. Where to go next →](14-where-to-go-next.md)
