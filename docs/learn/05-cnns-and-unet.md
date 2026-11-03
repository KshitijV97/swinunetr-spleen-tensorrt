# 5. CNNs and the U-Net

> **Summary.** A convolutional neural network (CNN) stacks learned filter
> banks with nonlinearities and downsampling. Each level sees a wider region
> of the input (its **receptive field**) at a coarser resolution.
> Segmentation needs both **context** ("this is a spleen") and **precision**
> ("its edge is exactly here"). The **U-Net** gets both: an encoder that
> downsamples for context, a decoder that upsamples back, and skip connections
> that carry fine detail across. If you've met multi-resolution pyramids or
> subband coding, it's the same shape. SwinUNETR is a U-Net whose encoder is
> a transformer.

---

## 5.1 From one filter to a deep CNN

A single convolution layer is a filter bank (chapter 2). A CNN is

```
conv → norm → nonlinearity → conv → norm → nonlinearity → downsample → ...
```

repeated. Because of the nonlinearities in between, the stack does more than
a single long filter could. It builds features hierarchically:

| Depth | Typical learned features |
|---|---|
| Early | Edges, intensity gradients, blobs |
| Middle | Textures, curved surfaces, boundaries between tissue types |
| Deep | Organ-scale shapes, relative positions ("bright, smooth, curved thing under the left ribs, next to the kidney") |

## 5.2 Receptive field: how much can a voxel "see"?

The **receptive field** of an output voxel is the region of the input that
can influence it. It's the support of the impulse response, if you ignore the
nonlinearities.

- One 3×3×3 conv: 3 voxels wide.
- $L$ stacked 3×3×3 convs (stride 1): $1 + 2L$ voxels wide. That grows
  **linearly**, which is slow. Seeing 100 voxels needs 50 layers.
- Add a stride-2 downsampling every couple of layers and each subsequent
  layer's 3×3×3 kernel covers twice as much of the original input. Growth
  becomes **exponential**. After 5 halvings, one voxel at the bottom of the
  network summarises a region on the order of the whole 96³ crop.

Even so, a CNN's *effective* receptive field (where the influence is
actually significant) tends to be much smaller than the theoretical one,
roughly Gaussian-shaped and concentrated in the centre. That limitation
motivates attention (chapter 6).

## 5.3 Downsampling: trading resolution for context

Downsampling (stride-2 conv or pooling) halves each spatial dimension, i.e.
it decimates by 2. In 3D the voxel count drops by 8×, so networks usually
**double the channels** at each halving. The information moves from "where"
(spatial positions) into "what" (feature channels). SwinUNETR does exactly
this: 48³ × F → 24³ × 2F → 12³ × 4F → 6³ × 8F → 3³ × 16F.

## 5.4 The segmentation dilemma

A classifier only needs the deep, coarse "what" features. **Segmentation
needs an answer at every voxel**, at full resolution. The deep features know
there's a spleen somewhere in the region but have lost precise positions:
after 5 halvings, one cell spans 32 voxels (about 5–6 cm).

## 5.5 The U-Net

Ronneberger et al. (2015) solved this with a symmetric encoder–decoder,
drawn as a "U":

```
input 96³ ─────────────────────────────── skip ──────────────────────────► + ─► output 96³
   └─► 48³ ─────────────────────── skip ──────────────────────► + ─► 48³ ─┘
         └─► 24³ ──────────── skip ──────────────► + ─► 24³ ─┘
               └─► 12³ ── skip ───────► + ─► 12³ ─┘
                     └─► 6³ ─► 3³ ─► 6³ ─┘
           encoder: downsample, more channels      decoder: upsample, fewer channels
```

- **Encoder** (left, going down): progressively coarser, more abstract
  features.
- **Decoder** (right, going up): progressively upsamples back to full
  resolution.
- **Skip connections** (horizontal): at each level, the encoder's
  feature map is **concatenated** onto the decoder's upsampled one. The
  decoder gets both "it's a spleen" from below and "the edge is exactly here"
  from the side.

**The EE analogy.** It is a multi-resolution analysis-synthesis system, like
a Laplacian pyramid or a wavelet/subband codec. The analysis bank splits the
signal into coarse approximations. Synthesis rebuilds it, adding back the
detail at each level. Without the detail subbands (skips), the reconstruction
would be blurry. The difference is that every filter here is learned, and
the synthesis is conditioned on a decision ("spleen or not") rather than
trying to reproduce the input.

## 5.6 Upsampling: transposed convolution

The decoder needs a learnable way to double resolution. A **transposed
convolution** (kernel 2, stride 2) does this: each input voxel "paints" a
learned 2×2×2 pattern into the output.

In DSP terms: zero-stuff by 2 (insert zeros between samples), then apply an
interpolation filter, except the filter is learned. With kernel = stride = 2,
there is no overlap between the patches. Each input voxel produces its own
2×2×2 output block, and the following 3×3×3 convolutions smooth the result.

## 5.7 Why U-Nets dominate medical segmentation

- They work with **little data**. Convolution's built-in assumptions
  (locality, translation equivariance) mean fewer things to learn.
- They're **fully convolutional**, so they accept any input size (modulo
  divisibility) and can be applied in sliding windows.
- The 3D variants (3D U-Net, V-Net) and the self-configuring **nnU-Net** are
  still extremely strong baselines today. Any claim that a new architecture is
  better should be checked against nnU-Net, and the README's extension list
  includes exactly this comparison.

## 5.8 What CNNs struggle with

Each layer only mixes information from a small neighbourhood. Long-range
relationships ("this bright blob is the spleen *because* it's on the
opposite side from the liver and above the left kidney") must be built up
through many layers and downsamplings. **Self-attention** (next chapter) lets
any position directly consult any other. SwinUNETR replaces the U-Net's
encoder with a transformer for exactly this reason, and keeps the
convolutional decoder for precise boundaries.

---

**In the code**
- The decoder blocks with transposed convolutions and skip concatenation
  are MONAI's `UnetrUpBlock`, used inside `SwinUNETR`
  ([`src/model.py`](../../src/model.py) → `build_model`).
- The divisibility rule that follows from 5 halvings: `validate_roi` (crop
  sizes must be divisible by $2^5 = 32$).

**Check yourself**
1. What is the receptive field of 4 stacked 3×3×3 convolutions with stride 1?
2. Why do networks double the channel count when they halve the resolution?
3. What would a U-Net's output look like without skip connections?
4. Why must the input size be divisible by 32 for a network that halves
   resolution 5 times?

Next: [6. Attention and transformers →](06-attention-and-transformers.md)
