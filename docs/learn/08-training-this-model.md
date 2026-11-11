# 8. Training this model

> **Summary.** We train SwinUNETR by repeatedly showing it 4 random 96³ crops
> and nudging its weights to reduce a loss.
> - The loss is **Dice + cross-entropy**: Dice handles the tiny-organ
>   imbalance, and cross-entropy gives smooth gradients.
> - The optimiser is **AdamW**: gradient descent with momentum (a low-pass
>   filter) and per-weight step normalisation (automatic gain control).
> - The learning rate follows a **cosine** curve down to zero.
> - On the GPU we use **mixed precision**: 16-bit floats for speed, a
>   loss-scaling trick so small gradients survive, and 32-bit master weights.
>
> This chapter also budgets GPU memory and traces exactly what one epoch does.

---

## 8.1 The imbalance problem, concretely

After resampling, a spleen of ~200 ml occupies about
$200{,}000\ \text{mm}^3 / (1.5 \times 1.5 \times 2.0\ \text{mm}^3) \approx 44{,}000$ voxels.
A whole cropped abdominal scan is a few million voxels, so the spleen is
**around 1 %**, often less.

A model that predicts "background" everywhere is ~99 % accurate per voxel,
yet it's useless. Plain cross-entropy is an average over voxels, so it is
dominated by the easy background. Two defences are used here:

1. **Balanced crop sampling** (chapter 2): half the crops are centred on
   spleen, so the spleen fills a decent fraction of what the model sees.
2. **A Dice term in the loss**, which measures overlap with the spleen and
   doesn't care how many background voxels there are.

## 8.2 Dice and soft Dice

The **Dice coefficient** between a predicted mask $P$ and the true mask $G$:

```math
\mathrm{Dice} = \frac{2\lvert P \cap G \rvert}{\lvert P \rvert + \lvert G \rvert}
```

It's 1 for perfect overlap and 0 for none. It only counts spleen voxels
(predicted or true), so the sea of background doesn't inflate it. In
information-retrieval terms, it's the **F1 score** (harmonic mean of
precision and recall) computed over voxels.

The mask $P$ comes from argmax, which is a step function with zero gradient
almost everywhere, so you can't train on it directly. **Soft Dice** replaces
the hard mask with probabilities $p_i \in [0, 1]$ from softmax:

```math
\mathcal{L}_\text{Dice} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
```

Now every voxel's probability gets a gradient. A tiny $\epsilon = 10^{-5}$
avoids division by zero, which matters for crops that contain no spleen at
all. For those crops the loss is near 0 if the model predicts no spleen, and
near 1 if it hallucinates any.

## 8.3 The combined loss: DiceCE

```math
\mathcal{L} = \mathcal{L}_\text{Dice} + \mathcal{L}_\text{CE}
```

- **Dice** optimises what we actually measure and is robust to imbalance,
  but early in training, when predictions are a uniform grey, it's noisy and
  flat.
- **Cross-entropy** (chapter 4) gives every voxel a clean, bounded gradient
  ($p - y$), which gets training started reliably.

Together they're the standard recipe for medical segmentation. In MONAI it's
one line: `DiceCELoss(to_onehot_y=True, softmax=True)`. One subtlety: by
default MONAI's Dice term includes the background channel too. Background
Dice is ~1 almost immediately, so it adds little, and the spleen term does
the real work. The *evaluation* metric excludes background (chapter 9).

## 8.4 The optimiser: Adam and AdamW

Plain gradient descent uses one step size for every weight. **Adam** adapts
it per weight using two running averages of the gradient $g$:

```math
m \leftarrow \beta_1 m + (1-\beta_1)\, g \qquad\qquad v \leftarrow \beta_2 v + (1-\beta_2)\, g^2
```

```math
\theta \leftarrow \theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon}
```

($\hat m, \hat v$ are bias-corrected versions. The averages start at zero, so
early on they're scaled up to compensate.)

Read this with your EE hat on:

- $m$ is a **first-order IIR low-pass filter** on the gradient (an
  exponential moving average, pole at $\beta_1 = 0.9$, time constant ≈ 10
  steps). This is **momentum**: it smooths out the noise of mini-batch
  gradients and keeps moving in consistent directions.
- $v$ tracks the gradient's **power** (pole at $\beta_2 = 0.999$, ≈ 1000
  steps). Dividing by $\sqrt{v}$ is **automatic gain control** per weight.
  Weights with large, noisy gradients take smaller steps, and weights with
  tiny gradients take larger ones. The effective step size is roughly $\eta$
  for every weight, whatever its gradient scale.

**AdamW** adds **weight decay**, a gentle pull of every weight toward zero
(regularisation, chapter 3), applied *directly* to the weights instead of
through the gradient: $\theta \leftarrow \theta - \eta\lambda\theta$. That
"decoupling" is what the W stands for. It matters because folding decay into
the gradient would get distorted by Adam's per-weight normalisation.

Our values: $\eta = 10^{-4}$, $\lambda = 10^{-5}$. A side note: PyTorch
multiplies the decay by the learning rate, so each step shrinks weights by a
factor of $1 - 10^{-9}$. That's an extremely weak regulariser. In practice
the augmentation and the architecture do most of the regularising here.

## 8.5 The learning-rate schedule: cosine annealing

Big steps early (explore quickly), small steps late (settle precisely). The
learning rate follows half a cosine from $\eta_0$ to 0 over the 150 epochs:

```math
\eta_t = \tfrac{1}{2}\,\eta_0\left(1 + \cos\frac{\pi t}{T}\right)
```

| Epoch | Learning rate |
|---:|---:|
| 0 | 1.0e-4 |
| 38 | 8.5e-5 |
| 75 | 5.0e-5 |
| 113 | 1.4e-5 |
| 150 | 0 |

The analogy is annealing in metallurgy: cool slowly so the system settles
into a good low-energy state.

## 8.6 Floating-point formats: the word-length trade-off

You've met fixed-point vs floating-point and word-length effects in DSP. GPUs
offer several float formats with different splits of bits between
**exponent** (range) and **mantissa** (precision):

| Format | Sign / exponent / mantissa bits | Max value | Relative precision ($2^{-\text{mantissa}}$) | Notes |
|---|---|---|---|---|
| FP32 | 1 / 8 / 23 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-7}$ | Default |
| **FP16** | 1 / 5 / 10 | **65,504** | $9.8 \times 10^{-4}$ | Fast on the T4's Tensor Cores |
| BF16 | 1 / 8 / 7 | $3.4 \times 10^{38}$ | $7.8 \times 10^{-3}$ | FP32's range, less precision. Not supported by the T4 |
| TF32 | 1 / 8 / 10 | $3.4 \times 10^{38}$ | $9.8 \times 10^{-4}$ | Internal matmul format on newer GPUs. Not on the T4 |

What goes wrong in FP16:
- **Overflow:** anything above 65,504 becomes `inf`. Sums of squares (in
  variance or LayerNorm) and exponentials (in softmax) are the usual culprits.
- **Rounding:** with 10 mantissa bits, $2048 + 1 = 2048$ in FP16, because the
  spacing between representable numbers at 2048 is 2.
- **Underflow:** values below about $6 \times 10^{-8}$ become exactly 0.
  Gradients are often that small.

So why bother? Because the T4's **Tensor Cores** do FP16 matrix multiplies
roughly **8× faster** than its regular FP32 units (≈ 65 vs ≈ 8 TFLOPS
peak), and FP16 halves memory traffic.

## 8.7 Mixed precision (AMP): getting the speed safely

**Automatic mixed precision** uses FP16 where it's safe and FP32 where it
isn't:

1. **Master weights stay FP32.** Tiny updates like $10^{-4} \times g$ would
   round away in FP16.
2. **`torch.autocast`** runs matmuls and convolutions in FP16 and keeps
   softmax, normalisation, reductions and the loss in FP32. It does this
   automatically, op by op.
3. **Loss scaling** fixes gradient underflow. `GradScaler` multiplies the loss
   by a big factor (it starts at $2^{16} = 65{,}536$) before backprop, so all
   gradients are scaled up into FP16's representable range. It divides them
   back down before the optimiser step. If any gradient overflowed to
   `inf`/`nan`, it skips that step and halves the factor. After a long run
   of good steps, it doubles it again.

The EE analogy is setting the gain in front of an ADC: amplify a weak signal
so it uses the converter's dynamic range instead of disappearing below the
LSB, and back off when it clips.

In code ([`src/train.py`](../../src/train.py)):

```python
with autocast():                       # FP16 where safe
    loss = loss_fn(model(image), label)
scaler.scale(loss).backward()          # backprop the scaled loss
scaler.step(optimizer)                 # unscale; skip the step if inf/nan
scaler.update()                        # adjust the scale factor
```

On a Mac (CPU/MPS) AMP is switched off and these calls do nothing.

## 8.8 GPU memory budget

Where do the T4's 16 GB go during training? Rough arithmetic for $F = 48$:

| Item | Size |
|---|---|
| Weights, FP32 (62.2 M × 4 B) | ~250 MB |
| Gradients, FP32 | ~250 MB |
| Adam's $m$ and $v$, FP32 | ~500 MB |
| **Subtotal: model state** | **~1 GB** |
| One stage-1 activation tensor (4 crops × 48³ tokens × 48 ch × 2 B) | ~42 MB, and dozens are saved per block (norms, Q/K/V, MLP at 4× width, ...) |
| Stage-1 attention probabilities (343 windows × 343² × 3 heads × 4 crops × 2 B) | ~1 GB **per block** if stored in full |
| One full-resolution decoder tensor (4 × 48 ch × 96³ × 2 B) | ~340 MB, and several per decoder block |

These are back-of-envelope figures, not measurements, but the conclusion is
robust: **activations dominate**, and they're mostly at the two finest
levels. That's why:
- the batch is only 4 crops, all from one scan;
- the config has `use_checkpoint` (**gradient checkpointing**): don't store
  activations, recompute them during the backward pass. That trades about
  30 % more compute for a large memory saving;
- the troubleshooting table's first fix for "CUDA out of memory" is
  checkpointing, then fewer crops, then a narrower model.

## 8.9 Feeding the GPU: caching

Loading a CT file, reorienting it and resampling it takes on the order of
seconds, far longer than one training step. If we did it every step, the GPU
would sit idle. `CacheDataset` runs those deterministic steps **once** for
every scan, keeps the results in RAM, and each step only does the cheap
random crop and augmentation. It's ordinary memoisation of a pure function.

## 8.10 What one epoch actually does

1. Shuffle the 33 training scans.
2. For each scan (33 steps):
   1. Take its cached preprocessed volume.
   2. Cut 4 random 96³ crops (2 centred on spleen, 2 on background) and
      randomly flip, rotate and intensity-shift them.
   3. Forward pass under autocast, giving logits of shape `(4, 2, 96, 96, 96)`.
   4. DiceCE loss → scaled backward → AdamW step.
3. Step the cosine schedule.
4. Every 5 epochs: **validate**. Run sliding-window inference over the 8
   whole validation scans (chapter 9), compute the mean spleen Dice, and if
   it's the best so far, save `best.pt`.
5. Save `last.pt`, a snapshot of the whole training run so far (§8.13), so
   an interruption loses at most this epoch.

Over 150 epochs that's ~4,950 optimisation steps and 30 validations. Each
log line looks like:

```
epoch   45/150  loss 0.2134  lr 7.94e-05  val_dice 0.9012  (new best, saved)  [2710s]
```

(Illustrative values, not results.) Use the elapsed-seconds column after the
first few epochs to estimate the total training time.

## 8.11 Reading the training curves

`results/training_history.json` stores the loss per epoch and Dice per
validation, and the Colab notebook plots them.

| What you see | What it suggests |
|---|---|
| Loss falls, validation Dice rises then plateaus | Healthy |
| Loss falls, validation Dice falls after a while | Overfitting (the best checkpoint is still saved from the peak) |
| Loss is `nan` | Divergence. Try without AMP to confirm, then lower the learning rate |
| Dice stuck near 0 | A data bug (labels misaligned, wrong window), not a modelling problem |
| Dice very noisy between validations | Expected with only 8 scans: one hard case moves the mean a lot |

The local **smoke test** (tiny model, 2 epochs, synthetic data) scores a Dice
of about 0.02. That's expected. It proves the pipeline runs end to end, and
nothing more.

## 8.12 Reproducibility

The seed fixes the data split, the random crops and the initial weights. GPUs
still use a few non-deterministic kernels (the order of floating-point
additions in parallel reductions varies), so two GPU runs can differ in the
third decimal of Dice. Keep that in mind when you compare numbers that differ
by about that much (chapter 12).

## 8.13 Surviving interruptions: checkpoints and resuming

Training runs for hours on a rented GPU that can disappear: free Colab
sessions disconnect when you close the laptop and wipe the machine. If the
only thing on disk is `best.pt`, an interruption at epoch 120 of 150 means
starting again from epoch 1.

**Why the weights alone aren't enough.** It's tempting to reload `best.pt`
and keep going. But the weights are only part of what training "knows":

| State | What it is | What breaks if you drop it |
|---|---|---|
| Weights | The model itself | Everything |
| AdamW moments $m, v$ and step count | Running averages of each weight's gradient and squared gradient (§8.4) | The optimiser forgets each weight's typical step size. The next updates are badly scaled, like restarting from a cold start |
| Schedule position | Where we are on the cosine curve (§8.5) | The learning rate jumps from its small late value back to $10^{-4}$, shaking a nearly converged model |
| `GradScaler` scale | The current loss multiplier for FP16 (§8.7) | It re-learns the scale from scratch, skipping steps on the way |
| Random-number generators | The state of every "dice" used for shuffling | The data order no longer continues the same sequence |
| History and best Dice | Loss/Dice so far, the best score | The curve has a hole, and a worse model could overwrite `best.pt` |

An EE analogy: pausing a running IIR filter. To continue its output exactly,
you have to save its internal **delay-line state**, not just its
coefficients. The weights are the coefficients; the optimiser moments and
schedule position are the delay line.

**What this repo does.** At the end of every epoch, `train.py` writes all of
the above to `checkpoints/last.pt`. On Colab with Drive mounted, that file
lives on your Drive. After a disconnect you run the same command with
`--resume` added:

```bash
python -m src.train --epochs 20             # interrupted after epoch 11
python -m src.train --epochs 20 --resume    # continues at epoch 12
```

Two engineering details are worth copying in your own projects:

- **Atomic writes.** The file is saved to `last.pt.tmp` and then renamed
  over `last.pt`. A rename is all-or-nothing, so a disconnect *during* the
  save leaves the previous good snapshot, never half a file.
- **Refuse mismatched configs.** If you resume with a different `--epochs`,
  the cosine curve would be stretched halfway through, and the result
  wouldn't correspond to any single, describable training run. So the code
  compares the saved config with the current one and stops with a list of
  differences. Only settings that don't change the maths (file paths,
  number of loader workers, gradient checkpointing) may differ.

**Honest caveat.** A resumed run is equivalent to an uninterrupted one, but
not bit-identical: MONAI's augmentations keep their own random generators,
which restart on resume. The crops after the interruption are different
random draws from the same distribution. That's the same order of
difference as GPU non-determinism (§8.12) and doesn't bias the result.

---

**In the code**
- Everything in this chapter: [`src/train.py`](../../src/train.py) → `train`.
- Hyperparameters: [`configs/default.yaml`](../../configs/default.yaml),
  `train:` and `model:` blocks.
- Caching: [`src/data.py`](../../src/data.py) → `build_dataloaders`.
- Seeding: [`src/utils.py`](../../src/utils.py) → `set_seed`.
- Resuming: [`src/train.py`](../../src/train.py) → `load_resume_state`,
  `atomic_save`, `config_differences`; tested in
  [`tests/test_train_resume.py`](../../tests/test_train_resume.py).
- The compact version: [training reference](../reference/training.md).

**Check yourself**
1. A prediction covers the spleen exactly and also 50 % extra volume around
   it. What's the Dice?
2. Why do master weights stay in FP32 even when the forward pass is FP16?
3. What does `GradScaler` do when it sees an `inf` gradient?
4. Where does most of the training memory go, and what does
   `use_checkpoint` trade for it?
5. You resume from epoch 100 of 150 with the right weights but a fresh
   optimiser and scheduler. What happens to the learning rate on the next
   step, and why is that harmful?

Next: [9. Inference and evaluation →](09-inference-and-evaluation.md)
