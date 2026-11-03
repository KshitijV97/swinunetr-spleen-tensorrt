# Inference and evaluation protocol

Code: [`src/evaluation.py`](../../src/evaluation.py) (shared by training
validation and the benchmark).

---

## 1. Sliding-window inference

`monai.inferers.sliding_window_inference` with `roi_size = 96³`,
`sw_batch_size = 4`, `overlap = 0.5`, `mode = "gaussian"`,
`sigma_scale = 0.125`, constant zero padding.

- **Tiling.** Along an axis of length $L \ge R$ ($R = 96$), the stride is
  $s = R(1 - \text{overlap}) = 48$ and there are $n = \lceil (L - R)/s \rceil + 1$
  window positions. The last window is aligned to the far edge. Axes shorter
  than $R$ are padded. Example: a 250 × 250 × 130 volume gives 5 × 5 × 2 = 50
  windows, i.e. 13 forward passes of batch 4 (the last one partially filled).
- **Blending.** Each window's logits are multiplied by a separable Gaussian
  importance map $w(\mathbf{x}) = \prod_a \exp\!\left(-\tfrac{(x_a - c_a)^2}{2\sigma_a^2}\right)$,
  $\sigma_a = 0.125 R$, accumulated, then normalised by the summed weights:

```math
\hat{z}(\mathbf{x}) = \frac{\sum_j w_j(\mathbf{x})\, z_j(\mathbf{x})}{\sum_j w_j(\mathbf{x})}
```

  This down-weights window borders, where context is truncated by zero padding
  and by the receptive field.
- **Logit averaging.** Blending happens on logits, before softmax/argmax.

Each predictor call sees exactly $(4, 1, 96, 96, 96)$, except possibly the
final partial batch. That is what allows a single fixed-shape ONNX/TensorRT
graph (`TRTPredictor` pads the partial batch, see [deployment.md](deployment.md)).

Cost scales linearly with the number of windows. With 50 % overlap it is
≈ $2^3 = 8\times$ the window count of non-overlapping tiling for large
volumes. `overlap = 0.25` would roughly halve latency. It is not swept here.

## 2. Post-processing

`argmax` over channels only. No connected-component filtering, no
largest-component selection, no hole filling, no test-time augmentation.
All are standard and would likely raise Dice slightly. They are omitted so the
backend comparison isolates the network.

## 3. Dice as computed here

`DiceScorer`: `AsDiscrete(argmax=True, to_onehot=2)` on predictions,
`AsDiscrete(to_onehot=2)` on labels,
`DiceMetric(include_background=False, reduction="mean")`:

```math
\mathrm{Dice}(P, G) = \frac{2\lvert P \cap G\rvert}{\lvert P\rvert + \lvert G\rvert}
```

Computed per volume on the foreground channel, then averaged over the 8
volumes (unweighted, so small and large spleens count equally). All validation
cases contain spleen, so the empty-GT NaN path (`ignore_empty=True`) is not
exercised on real data.

**Space.** Dice is evaluated on the RAS-reoriented, 1.5 × 1.5 × 2.0 mm
resampled, foreground-cropped grid. It is not computed on the native
acquisition grid. Native-space evaluation would invert `Spacingd` and
`Orientationd` (`Invertd`) and resample predictions with nearest neighbour.
For a large, compact organ the difference is usually small, but comparisons
with published native-space numbers are not like-for-like. The caveat applies
identically to every backend, so the PyTorch vs TensorRT comparison is
unaffected.

No surface metric (HD95, NSD) is reported. For surgical-guidance use, boundary
error matters at least as much as volumetric overlap. HD95 and NSD are the
natural next metrics.

## 4. Threats to validity

| Issue | Effect | Mitigation status |
|---|---|---|
| $n = 8$ validation cases | Wide uncertainty on mean Dice. Per-case scores are in `benchmark.json` (`dice_per_volume`) | Report per-case values. Cross-validation as future work |
| Selection and reporting on the same cases | Optimistic bias in best Dice | Disclosed. A held-out test split would remove it |
| Single-centre, single-phase, single-pathology cohort | No external validity claim | Disclosed |
| Resampled-space Dice | Not comparable to native-space literature numbers | Disclosed. `Invertd` as future work |
| AMP during in-training validation | Selection metric differs (slightly) from FP32 metric | Benchmark reports FP32/AMP/TRT Dice for the selected checkpoint |
| GPU non-determinism | ~1e-3 Dice jitter across runs | Treat backend Dice differences at that scale as noise |

## 5. Comparing backends

All backends share preprocessing, window geometry, blending and argmax, so
Dice differences between `pytorch_fp32`, `pytorch_amp` and `tensorrt_fp16`
reflect numerics only. A useful additional diagnostic, not implemented, is
voxelwise argmax agreement between backends on the validation volumes. The ONNX
export check reports this on random inputs only (see
[deployment.md §2](deployment.md#2-onnx-export)).
