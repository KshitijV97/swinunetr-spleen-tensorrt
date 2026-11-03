# Training, explained

Code: [`src/train.py`](../src/train.py), [`src/evaluation.py`](../src/evaluation.py).
Config: [`configs/default.yaml`](../configs/default.yaml).

---

## 1. Commands

```bash
# Local smoke test: tiny model, 2 real volumes (downloads the dataset on first run)
python -m src.train --smoke-test

# Local smoke test without any download (synthetic volumes)
python -m src.train --smoke-test --synthetic

# Full training (CUDA GPU, e.g. Colab T4)
python -m src.train

# Override any of the common settings without editing the YAML
python -m src.train --epochs 80 --feature-size 24 --lr 2e-4
python -m src.train --help        # all flags
```

Outputs:

| File | Content |
|---|---|
| `checkpoints/best.pt` | Weights with the best validation Dice, plus the config and epoch they came from |
| `checkpoints/last.pt` | Weights after the final epoch |
| `results/training_history.json` | Loss per epoch, Dice per validation, best Dice/epoch, training time, parameter count, config, software versions, GPU name |

With `--smoke-test` these go to `checkpoints/smoke/` and `results/smoke/`
(both git-ignored, because smoke-test numbers are meaningless).

## 2. How the config works

1. `configs/default.yaml` is loaded.
2. If `--smoke-test` is given, the `smoke_test:` block at the bottom of the
   YAML is merged on top (only the keys it lists change).
3. Command-line flags such as `--epochs` override single values.
4. The final, merged config is stored **inside** every checkpoint and every
   results JSON, so a result can always be traced back to its exact settings.

## 3. Every hyperparameter

| Key | Default | Meaning and reasoning |
|---|---|---|
| `seed` | 42 | Fixes the data split, crop sampling and weight initialisation |
| `data.val_fraction` | 0.2 | 8 of the 41 volumes are held out for validation |
| `data.cache_rate` | 1.0 | Fraction of preprocessed volumes kept in RAM (see [DATA.md](DATA.md#4-caching-why-training-is-not-io-bound)) |
| `data.pixdim` | 1.5, 1.5, 2.0 mm | Target voxel spacing |
| `data.hu_window` | −57, 164 | CT intensity window |
| `data.roi_size` | 96³ | Crop size for training = window size for inference. Must be divisible by 32 |
| `data.num_samples` | 4 | Crops per volume per step (effective batch = `batch_size × num_samples`) |
| `model.feature_size` | 48 | Network width. 48 ≈ 62M parameters; 24 ≈ 16M |
| `model.use_checkpoint` | false | Gradient checkpointing: recompute activations in the backward pass instead of storing them. Saves a lot of memory and costs roughly a third more time |
| `train.epochs` | 150 | One epoch = one pass over the 33 training volumes (33 steps × 4 crops) |
| `train.lr` | 1e-4 | AdamW learning rate, the usual choice for SwinUNETR fine-tuning and training |
| `train.weight_decay` | 1e-5 | Light L2-style regularisation (decoupled, as in AdamW) |
| `train.amp` | true | Mixed precision on CUDA |
| `train.val_interval` | 5 | Validation is expensive (whole volumes), so it runs every 5 epochs |
| `inference.sw_batch_size` | 4 | Windows per forward pass during sliding-window inference |
| `inference.overlap` | 0.5 | Neighbouring windows overlap by 50% |

## 4. The loss: DiceCE

`DiceCELoss(to_onehot_y=True, softmax=True)` is the **sum** of two losses:

- **Soft Dice loss** = 1 − Dice computed on softmax probabilities. It
  directly optimises the metric we care about and is insensitive to class
  imbalance. If the spleen is 1% of the crop, missing it entirely still
  costs a lot.
- **Cross-entropy** gives smooth, well-scaled gradients for every voxel,
  which stabilises the early epochs when the Dice term is noisy.

## 5. The optimiser and schedule

- **AdamW** is Adam with weight decay applied directly to the weights rather
  than through the gradient. It is the standard optimiser for transformers.
- **Cosine annealing** lowers the learning rate from `lr` to 0 along half a
  cosine over `epochs`: big steps early, fine adjustments at the end.

## 6. Mixed precision (AMP)

On CUDA the forward pass runs under `torch.autocast(float16)`: matrix
multiplications and convolutions use FP16 on the T4's Tensor Cores, while
precision-sensitive ops (softmax, norms, the loss) stay FP32. The master copy
of the weights stays FP32.

`GradScaler` multiplies the loss by a large factor before `backward()` so tiny
gradients do not round to zero in FP16, then divides the gradients back before
the optimiser step. If it detects inf/NaN gradients it skips that step and
lowers the factor. On CPU/MPS, AMP is switched off and these objects do
nothing.

## 7. Sliding-window inference

Validation volumes are whole scans (for example ~250 × 250 × 130 voxels after
resampling), but the network was trained on 96³ crops.
`monai.inferers.sliding_window_inference`:

1. tiles the volume with 96³ windows that overlap by `overlap` (50%);
2. runs the model on `sw_batch_size` windows at a time;
3. blends overlapping predictions with a **Gaussian** weight map (window
   centres count more than their edges, where context is missing);
4. returns logits the size of the full volume.

The same function is used in training validation **and** in the benchmark, so
the Dice scores are directly comparable.

## 8. The metric: Dice

For the predicted spleen mask P and the ground truth G:

```
Dice = 2 |P ∩ G| / (|P| + |G|)
```

It goes from 0 (no overlap) to 1 (identical). Background is excluded
(`include_background=False`), otherwise the millions of correct background
voxels would make every model look excellent. Reported values are the mean
over validation volumes.

As context only (not a claim about this repository): well-trained models
commonly reach validation Dice in the mid-0.9s on MSD Spleen. If your number
is far below that after full training, something is probably wrong (see
troubleshooting below).

## 9. Checkpointing

After each validation, if the mean Dice beats the best so far, `best.pt` is
written. The export and benchmark scripts always use `best.pt`. Selecting the
checkpoint on the same validation set you report is slightly optimistic. This
is common practice with tiny datasets, but say so (the README does).

## 10. Reproducibility

`set_seed` seeds Python, NumPy, PyTorch and MONAI (`set_determinism`). On a
GPU some kernels are still non-deterministic, so expect tiny run-to-run
differences in the last decimal places of Dice.

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | 48-wide model + 4 crops of 96³ | `model.use_checkpoint: true`, or `data.num_samples: 2`, or `--feature-size 24` |
| Colab crashes while "Loading dataset" | System RAM full from caching | `data.cache_rate: 0.5` or `--max-volumes 30` |
| Dice stays near 0 | Labels misaligned with images, or wrong intensity window | Run `pytest`; check that `python -m src.data --summary` lists real files |
| Loss becomes `nan` with AMP | FP16 overflow | Try `--no-amp` to confirm, then lower `--lr` |
| Very slow on Mac | SwinUNETR is heavy; MPS/CPU are not the target | That is expected. Use the smoke test locally and train on Colab |
