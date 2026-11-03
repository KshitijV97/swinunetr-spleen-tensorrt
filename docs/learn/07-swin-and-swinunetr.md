# 7. Swin and SwinUNETR

> **Summary.** The **Swin Transformer** makes attention affordable on large
> grids:
> - Tokens only attend within small **7×7×7 windows** (cost becomes linear
>   in the number of tokens).
> - Every other block **shifts the windows** by half a window so information
>   crosses window borders.
> - Between stages it **merges** 2×2×2 tokens into one (halve resolution,
>   double channels), building a CNN-like pyramid.
>
> **SwinUNETR** plugs that pyramid into a U-Net as its encoder and keeps a
> convolutional decoder. This chapter walks through the whole network with
> every tensor shape, for our settings.

---

## 7.1 The plan

```
96³ CT crop
  → patch embedding               48³ tokens × 48 channels
  → Swin stage 1 (2 blocks)       48³ × 48
  → merge → Swin stage 2          24³ × 96
  → merge → Swin stage 3          12³ × 192
  → merge → Swin stage 4           6³ × 384
  → merge                          3³ × 768
  → CNN decoder with skips back up to 96³
  → 2 logits per voxel
```

The rest of this chapter explains each arrow.

## 7.2 Patch embedding

A 3D convolution with kernel 2, stride 2, from 1 channel to $F = 48$
channels. Each 2×2×2 patch (8 voxels, 3 × 3 × 4 mm after resampling) becomes
one 48-number token. Output: $48 \times 48^3$. There's no separate position
embedding: position enters through the relative position bias inside
attention (chapter 6).

## 7.3 Window attention: the fix for quadratic cost

Group the 48³ token grid into non-overlapping **7×7×7 windows** (343 tokens
each) and run self-attention **only inside each window**.

48 isn't divisible by 7, so the grid is zero-padded to 49³ first. That gives
$7^3 = 343$ windows. The cost comparison per head per crop:

| | Score-matrix entries |
|---|---|
| Global attention | $110{,}592^2 \approx 1.2 \times 10^{10}$ |
| Window attention | $343 \text{ windows} \times 343^2 \approx 4.0 \times 10^{7}$ |

That's about **300× less**, and it now grows **linearly** with the number of
tokens: twice the volume means twice the windows, not four times the work.

The price is that a token can't see anything outside its own window. A
token at a window's edge is blind to its neighbour one voxel away across the
border.

## 7.4 Shifted windows: letting information cross borders

The trick: the **second** block of each stage moves the window grid by
**half a window** (3 tokens along each axis) before partitioning. Tokens
that were on opposite sides of a border in block 1 now share a window in
block 2.

A 1D picture with window size 4 and shift 2:

```
tokens:         a b c d | e f g h | i j k l
block 1 (W):   [a b c d] [e f g h] [i j k l]        d and e cannot talk
block 2 (SW):  [c d e f] [g h i j] [k l a b]*       d and e share a window
```

*The edges are handled by a **cyclic shift**: roll the tokens so the
windows line up again, compute attention, then roll back. The tokens that
wrapped around (`a b` next to `k l`) weren't actually neighbours, so an
**attention mask** adds −∞ to their scores. After softmax they get zero
weight.

Alternating regular (W-MSA) and shifted (SW-MSA) blocks, stage after stage
on ever-coarser grids, lets information spread across the whole crop. It's
like overlapping block processing with alternating block boundaries.

**At small grids the windows cover everything.** At stage 4 the grid is only
6³, smaller than a 7³ window. MONAI then shrinks the window to 6³ and turns
the shift off. So stage 4 runs true **global** attention over all 216 tokens:
cheap at that size, and it gives the deepest features a view of the whole
crop.

## 7.5 Relative position bias

Each window's attention adds a learned bias that depends only on the 3D
**offset** between two tokens. Offsets inside a 7³ window range from −6 to
+6 per axis, so there are $13^3 = 2{,}197$ possible offsets, each with one
learned number per head. The model can learn "the token directly above
matters a lot, the one six away matters little" without any absolute
coordinates.

## 7.6 Patch merging: going down a level

Between stages, every 2×2×2 group of neighbouring tokens is concatenated
($8 \times C$ numbers), LayerNorm-ed and linearly projected to $2C$:

```
stage 1 → 2: 48³ × 48  → 24³ × 96
stage 2 → 3: 24³ × 96  → 12³ × 192
stage 3 → 4: 12³ × 192 →  6³ × 384
stage 4 → bottleneck: 6³ × 384 → 3³ × 768
```

This is the transformer's equivalent of a stride-2 convolution: decimate
space, expand features (chapter 5).

## 7.7 The full encoder, as configured here

| Stage | Grid | Channels | Heads | Window / shift | Blocks |
|---|---|---|---|---|---|
| Embed | 48³ | 48 | — | — | conv |
| 1 | 48³ (padded to 49³) | 48 | 3 | 7 / 3 | 2 (W, SW) |
| 2 | 24³ (padded to 28³) | 96 | 6 | 7 / 3 | 2 |
| 3 | 12³ (padded to 14³) | 192 | 12 | 7 / 3 | 2 |
| 4 | 6³ | 384 | 24 | 6 / 0 (global) | 2 |
| Bottleneck | 3³ | 768 | — | — | merge |

Each stage's output (after merging, and LayerNorm-ed) is handed to the
decoder as a skip connection.

## 7.8 The decoder: back to full resolution

The decoder is a regular convolutional U-Net decoder (chapter 5). Before the
transformer features are used, each passes through a small residual
convolution block (MONAI calls them `encoder1`…`encoder4` and `encoder10`)
that adapts them for the CNN. One of them, `encoder1`, runs directly on the
**raw 96³ input**. That gives the decoder a full-resolution view that no
transformer stage has, which helps place the boundary to the exact voxel.

```
bottleneck 3³ × 768 → res block
  ↑ ×2, concat stage-4 output   →  6³ × 384
  ↑ ×2, concat stage-3 output   → 12³ × 192
  ↑ ×2, concat stage-2 output   → 24³ × 96
  ↑ ×2, concat stage-1 output   → 48³ × 48
  ↑ ×2, concat raw-input block  → 96³ × 48
  → 1×1×1 conv                  → 96³ × 2 logits
```

Each "↑ ×2" is a transposed convolution, followed by a residual block of
3×3×3 convolutions with InstanceNorm and LeakyReLU.

There are five halvings on the way down (embedding + four merges), so the
input size must be divisible by $2^5 = 32$. That's why the crop is 96 (3 ×
32) and why `validate_roi` rejects anything else.

## 7.9 How big is it?

| Width $F$ | Parameters | Used for |
|---|---|---|
| 12 | 4.08 M | The tiny local smoke test |
| 24 | 15.7 M | A lighter fallback |
| **48** | **62.2 M** | **The default, as in the paper** |

Doubling $F$ roughly quadruples the parameters, because most weights are in
matrices whose input and output widths both scale with $F$.

Memory during training is dominated not by those 62 M weights (≈ 250 MB in
FP32) but by **activations**: every intermediate tensor saved for backprop,
especially at the 48³ and 96³ levels. Chapter 8 does the arithmetic.

## 7.10 Why this architecture for this problem?

- **Long-range context.** The spleen is identified partly by *where* it is
  relative to other organs. Attention can relate distant regions directly,
  whereas a CNN builds that up slowly.
- **Affordable in 3D.** Windowing makes attention linear in volume size.
- **Sharp boundaries.** The CNN decoder and the full-resolution skip recover
  voxel-precise edges.
- **Proven.** SwinUNETR is a well-known, strong model for 3D medical
  segmentation, with a maintained implementation in MONAI.

To be fair to CNNs: on small single-organ tasks like this, a well-tuned 3D
U-Net (nnU-Net) is typically just as accurate. This project picked
SwinUNETR partly *because* it is harder to deploy (attention, LayerNorm,
window reshapes), which makes the ONNX and TensorRT half more interesting.
The README lists a U-Net baseline as an extension.

---

**In the code**
- [`src/model.py`](../../src/model.py) → `build_model` creates
  `monai.networks.nets.SwinUNETR(in_channels=1, out_channels=2, feature_size=48, ...)`.
  Everything in this chapter happens inside that one class.
- `python -m src.model` prints the parameter count.
- [`tests/test_model.py`](../../tests/test_model.py) checks that output shape
  equals input shape, that bad sizes are rejected, and that the default model
  has about 62 M parameters.
- The compact spec with a full shape diagram:
  [architecture reference](../reference/architecture.md).

**Check yourself**
1. How many 7³ windows does stage 2 have after padding 24³ to 28³?
2. Why does the shifted-window block need an attention mask?
3. Why does stage 4 end up with global attention?
4. What's the role of the `encoder1` block on the raw input?

Next: [8. Training this model →](08-training-this-model.md)
