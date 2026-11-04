# Training

Code: [`src/train.py`](../../src/train.py). Config:
[`configs/default.yaml`](../../configs/default.yaml).

---

## 1. Commands and outputs

```bash
python -m src.train                                  # full run (CUDA)
python -m src.train --epochs 80 --feature-size 24    # CLI overrides
python -m src.train --smoke-test [--synthetic]       # tiny CPU/MPS run
```

| Output | Content |
|---|---|
| `checkpoints/best.pt` | `model_state`, merged `config`, `epoch`, `val_dice` at best validation |
| `checkpoints/last.pt` | Final-epoch weights + config |
| `results/training_history.json` | Per-epoch loss, per-validation Dice, best Dice/epoch, wall time, parameter count, merged config, environment |

## 2. Configuration resolution

`default.yaml` → deep-merge `smoke_test:` block (if `--smoke-test`) → CLI
overrides. The merged dict is embedded in every checkpoint and result file.
`load_checkpoint` rebuilds the network from the **checkpoint's** config, so
export and benchmark cannot drift from training (`feature_size`, `roi_size`).

## 3. Objective

`DiceCELoss(to_onehot_y=True, softmax=True)` with MONAI defaults
(`include_background=True`, `batch=False`, `smooth_nr = smooth_dr = 1e-5`,
`lambda_dice = lambda_ce = 1`, mean reduction):

```math
\mathcal{L} = \mathcal{L}_\text{Dice} + \mathcal{L}_\text{CE}
```

```math
\mathcal{L}_\text{Dice} = \frac{1}{BK}\sum_{b=1}^{B}\sum_{k=1}^{K}
\left(1 - \frac{2\sum_i p_{bik}\, g_{bik} + \epsilon}{\sum_i p_{bik} + \sum_i g_{bik} + \epsilon}\right),
\qquad
\mathcal{L}_\text{CE} = -\frac{1}{B N}\sum_{b,i} \log p_{b i y_{bi}}
```

where $p = \mathrm{softmax}(z)$ over $K = 2$ channels, $g$ is one-hot,
$i$ runs over the $N = 96^3$ voxels of a crop, and $B = 4$ crops.

Points worth knowing:

- **Background is included in the Dice term** (MONAI default). Background
  soft Dice is near 1 almost immediately, so it contributes little gradient.
  The foreground term carries the overlap signal. The validation metric
  excludes background (§7).
- **Per-sample Dice** (`batch=False`). Negative crops with no spleen have
  $\sum g = 0$. The $\epsilon$ smoothing then makes the foreground term reward
  predicting no foreground: $\mathcal{L}_\text{Dice} \to 0$ as $\sum p \to 0$,
  but it is ≈ 1 for any non-trivial false-positive mass. That is the intended
  behaviour for pos/neg sampling.
- CE supplies dense, well-conditioned gradients in early epochs, when the
  soft Dice of a near-uniform prediction is flat.

## 4. Optimisation

| | Setting |
|---|---|
| Optimiser | AdamW, $\eta = 10^{-4}$, $\lambda = 10^{-5}$ (decoupled), $\beta = (0.9, 0.999)$, $\epsilon = 10^{-8}$ |
| Schedule | `CosineAnnealingLR(T_max = epochs)`, stepped per epoch, $\eta_\text{min} = 0$, no warm-up |
| Epochs | 150 × 33 steps (one case, 4 crops per step) ≈ 4,950 steps |
| Gradient clipping | none |
| Weight decay exclusion | none (applies to norms and biases too) |

```math
\eta_t = \tfrac{1}{2}\,\eta_0\left(1 + \cos\frac{\pi t}{T}\right)
```

These are the common SwinUNETR-from-scratch settings for MSD/BTCV-scale data.
PyTorch's AdamW applies decay as $\theta \leftarrow \theta(1 - \eta\lambda)$, so
the per-step shrink here is $10^{-9}$. That is negligible: regularisation
effectively comes from augmentation and architecture, not weight decay.
Each step is a single case, so the gradient is a strongly correlated
4-crop estimate. Lack of warm-up has not been a problem at $\eta = 10^{-4}$
with AdamW.

## 5. Mixed precision

On CUDA only: `torch.autocast("cuda", dtype=torch.float16)` around forward and
loss, plus `torch.amp.GradScaler` with dynamic loss scaling (skipped steps on
inf/NaN, scale backoff). Master weights and optimiser state are FP32. Autocast
keeps softmax, LayerNorm, reductions and the loss in FP32. On CPU/MPS, AMP is
disabled and the scaler is a no-op. The T4 (Turing, sm_75) has FP16 Tensor
Cores and no BF16 or TF32, so FP16 + loss scaling is the only fast option.

**Validation during training also runs under autocast on CUDA.** The
selection metric is therefore the AMP Dice. The benchmark separately reports
FP32, AMP and TensorRT FP16 Dice for the selected checkpoint.

## 6. Validation and checkpoint selection

Every `val_interval = 5` epochs (and at the last epoch): sliding-window
inference on each of the 8 whole validation volumes, mean foreground Dice
(see [inference-and-evaluation.md](inference-and-evaluation.md)). `best.pt`
is written on strict improvement. Export and benchmark always use `best.pt`.
Selection and reporting share the same 8 cases, which biases the reported
Dice upward (max over 30 noisy evaluations). The bias is small for a
saturated task but not zero.

## 7. Hyperparameters

| Key | Default | Note |
|---|---|---|
| `seed` | 42 | Split, crop sampling, init (`set_determinism`) |
| `data.pixdim` | 1.5, 1.5, 2.0 mm | |
| `data.hu_window` | −57, 164 | |
| `data.roi_size` | 96³ | Divisible by 32 |
| `data.num_samples` | 4 | Crops per case per step |
| `data.pos_neg_ratio` | 1:1 | |
| `data.cache_rate` | 1.0 | |
| `model.feature_size` | 48 | 62.2 M parameters |
| `model.use_checkpoint` | false | Activation recomputation (~30 % slower, large memory saving) |
| `train.epochs` | 150 | |
| `train.lr` / `weight_decay` | 1e-4 / 1e-5 | |
| `train.amp` | true | CUDA only |
| `train.val_interval` | 5 | |
| `inference.sw_batch_size` | 4 | Also the ONNX/TensorRT batch dimension |
| `inference.overlap` | 0.5 | |

## 8. Determinism

Python, NumPy, torch and MONAI RNGs are seeded. `cudnn.benchmark` is left at
its default (off) and deterministic algorithms are not forced, so CUDA runs
can differ in the last digits (atomics in some backward kernels). Expect
run-to-run Dice variation of the order of 1e-3. That is relevant when you
compare backends whose Dice differs by a similar amount (see
[benchmarking.md](benchmarking.md)).

## 9. Failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| CUDA OOM | $F = 48$, 4 × 96³ crops, activations | `--use-checkpoint` (or `use_checkpoint: true`), `num_samples: 2`, or $F = 24$ (report it) |
| Host OOM at cache time | `cache_rate: 1.0` on Colab RAM | `cache_rate: 0.5` or `--max-volumes` |
| Dice ≈ 0 after many epochs | Image/label misalignment, wrong window | `pytest`, `python -m src.data --summary`, inspect one case |
| NaN loss under AMP | FP16 overflow | Confirm with `--no-amp`, then lower LR |

Well-trained models commonly reach validation Dice in the mid-0.9s on this
task. This is context, not a claim about this repository.
