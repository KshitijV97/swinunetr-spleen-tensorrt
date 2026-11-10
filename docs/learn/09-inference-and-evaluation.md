# 9. Inference and evaluation

> **Summary.** The network was trained on 96³ crops, but a whole scan is much
> bigger. So we slide an overlapping 96³ window across the scan, predict each
> window, and blend the overlaps with a Gaussian weight. That's overlap-add
> block processing with a window function, exactly as in an STFT. Then we
> take the argmax per voxel and score the result with **Dice**. This chapter
> also covers what a Dice number does and doesn't tell you, and why 8
> validation scans call for humility.

---

## 9.1 Why not feed the whole scan at once?

A preprocessed scan might be 250 × 250 × 130 voxels, about 8 million. Two
problems:

1. **Memory.** Activations scale with input size (chapter 8). A whole scan
   through SwinUNETR at once won't fit comfortably on a 16 GB GPU.
2. **Fixed shapes are easier to deploy.** A TensorRT engine built for one
   exact input shape is simpler and faster (chapter 10). With sliding
   windows, the network *always* sees $(4, 1, 96, 96, 96)$.

## 9.2 Sliding-window inference

`monai.inferers.sliding_window_inference` does the following:

1. **Tile** the scan with 96³ windows. With 50 % overlap the stride is 48
   voxels. Along an axis of length $L$ the number of window positions is
   $\lceil (L - 96) / 48 \rceil + 1$.
2. **Batch** the windows 4 at a time and run the network on each batch.
3. **Weight** each window's output with a 3D Gaussian bump (peak at the
   centre, falling off toward the edges, $\sigma = 0.125 \times 96 = 12$ voxels).
4. **Accumulate** weighted outputs and weights into full-size buffers, then
   divide: a weighted average at every voxel.

```math
\hat z(\mathbf{x}) = \frac{\sum_j w_j(\mathbf{x})\, z_j(\mathbf{x})}{\sum_j w_j(\mathbf{x})}
```

**Worked example.** For 250 × 250 × 130: in-plane $\lceil 154/48 \rceil + 1 = 5$
positions per axis, through-plane $\lceil 34/48 \rceil + 1 = 2$. That's
$5 \times 5 \times 2 = 50$ windows, so **13 forward passes** of batch 4 (the
last one has 2 real windows plus 2 padding). Latency is roughly proportional
to that window count, which is why bigger scans take longer (chapter 12).

**The DSP connection.** This is overlap-add, with the Gaussian as the
analysis/synthesis window. Why taper? Near a window's edge, the network's
receptive field runs off the window into zero padding, so its predictions
there are less trustworthy (less context). Tapering lets each voxel's answer
come mostly from windows where it sat near the centre.

**Cost of overlap.** 50 % overlap means roughly $2^3 = 8$ times as many
windows as non-overlapping tiling on a large volume. Overlap 0.25 would
roughly halve the work, at some cost to accuracy at seams. This project keeps
0.5, the standard choice, and lists a sweep as future work.

## 9.3 From logits to a mask

After blending, every voxel has two scores. **Argmax** picks the larger one:
0 = background, 1 = spleen. No post-processing is applied. Common tricks,
like keeping only the largest connected blob (there is one spleen), filling
holes, or averaging predictions over flipped inputs, would likely nudge the
score up. They are left out on purpose so that the PyTorch vs TensorRT
comparison measures only the network.

## 9.4 Dice, carefully

```math
\mathrm{Dice}(P, G) = \frac{2\lvert P\cap G\rvert}{\lvert P\rvert + \lvert G\rvert}
```

- Computed **per scan**, on the spleen channel only (background excluded,
  otherwise the millions of easy background voxels would make every model
  look perfect).
- Then **averaged over the 8 validation scans**. Each scan counts equally,
  whatever its spleen size.

Related metric: IoU (Jaccard) $J = \lvert P \cap G\rvert / \lvert P \cup G\rvert$.
They're monotonically related, $\mathrm{Dice} = 2J/(1+J)$, so Dice 0.95 ≈ IoU
0.90.

### What does Dice 0.93 actually mean?

Here's an instructive rough calculation. Model a 200 ml spleen as a sphere:
radius ≈ 36 mm, surface area ≈ 16,500 mm². That's about 44,000 voxels at
4.5 mm³ each. Now suppose the model's boundary is off by **one voxel layer
everywhere** (≈ 1.5–2 mm too big). The extra shell is about
16,500 mm² × 1.7 mm ≈ 28,000 mm³ ≈ 6,200 voxels:

```math
\mathrm{Dice} = \frac{2 \times 44{,}000}{44{,}000 + 50{,}200} \approx 0.93
```

So **a Dice in the low 0.9s can mean the boundary is consistently off by
about 2 mm.** A real spleen has more surface than a sphere, so it's even more
sensitive. For a big organ Dice looks forgiving. For surgical guidance, where
the boundary itself is the point, you want a **surface-distance metric** too,
such as the 95th-percentile Hausdorff distance (HD95) or normalised surface
Dice. They aren't implemented here and are listed as extensions.

### What's a "good" score?

- Well-trained models commonly reach validation Dice in the **mid-0.9s** on
  this dataset. This is context from the literature, not a claim about this
  repo.
- Two human experts outlining the same spleen don't agree perfectly either:
  boundary voxels are ambiguous (partial volume, chapter 2). Model
  improvements smaller than the human disagreement aren't clinically
  meaningful.

## 9.5 How much should you trust 8 scans?

Suppose per-scan Dice values have a standard deviation of 0.02 across
scans. The standard error of their mean is

```math
\mathrm{SE} = \frac{s}{\sqrt{n}} = \frac{0.02}{\sqrt 8} \approx 0.007
```

so a rough 95 % interval is ±0.014 around the mean. Two models whose mean
Dice differs by 0.005 on these 8 scans are statistically **indistinguishable**.
(The 0.02 is illustrative. The real per-scan values will be in
`results/benchmark.json → dice_per_volume`.)

Comparing *backends* is different: PyTorch and TensorRT are run on the
**same** 8 scans, so you can look at per-scan differences. That is a paired
comparison, and much more sensitive.

Three more caveats, all listed in the README:

1. **The checkpoint was chosen on these same 8 scans** (best of 30
   validations). Picking a maximum of noisy measurements biases it upward.
2. **Dice is measured on the resampled grid** (1.5 × 1.5 × 2 mm, cropped),
   not on each scan's original voxels. Papers that report "native space"
   Dice aren't directly comparable. It doesn't matter for comparing our
   backends, because they all share the same grid.
3. **One hospital, one contrast phase, one patient group.** No claim about
   other settings.

---

**In the code**
- Sliding windows and Dice, shared by training and benchmarking:
  [`src/evaluation.py`](../../src/evaluation.py) → `predict_volume`,
  `DiceScorer`.
- Window settings: [`configs/default.yaml`](../../configs/default.yaml),
  `inference:` block (`sw_batch_size: 4`, `overlap: 0.5`).
- Test that odd-sized volumes work:
  [`tests/test_model.py`](../../tests/test_model.py) →
  `test_sliding_window_handles_arbitrary_volume_size`.
- The compact version, including a threats-to-validity table:
  [inference and evaluation reference](../reference/inference-and-evaluation.md).

**Check yourself**
1. How many windows (and forward passes of 4) for a 200 × 220 × 96 volume?
2. Why are window edges down-weighted when blending?
3. Convert Dice 0.90 to IoU.
4. Why is a paired comparison of two backends more sensitive than comparing
   their mean Dice?

Next: [10. GPUs and deployment →](10-gpus-and-deployment.md)
