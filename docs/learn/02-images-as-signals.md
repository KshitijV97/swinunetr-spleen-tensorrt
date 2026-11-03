# 2. Images as signals

> **Summary.** A CT volume is a **3D sampled signal**. Voxel spacing is the
> sampling period, and it differs per axis and per scan. Before learning
> anything, we put every scan on the same footing:
> - reorient the axes so they point the same way in every scan;
> - **resample** to a common sampling grid (interpolation);
> - **window** the intensities into [0, 1] (a clip followed by a gain stage);
> - crop away empty space.
>
> Then we meet the operation everything else is built on: **3D convolution**,
> which is a multi-channel FIR filter whose taps are learned.

---

## 2.1 A voxel is a sample

Think of the patient's body as a continuous 3D field $\mu(x, y, z)$. The
scanner gives us samples of it on a grid:

```math
v[i, j, k] \approx (\mu * h)(i\,s_x,\; j\,s_y,\; k\,s_z)
```

- $s_x, s_y, s_z$ are the **spacing** in mm, i.e. the sampling period per
  axis. Typical values are 0.7 × 0.7 mm in-plane and 1–5 mm between slices.
  Unlike a camera image, the grid is usually **anisotropic**: each voxel is a
  brick, not a cube.
- $h$ is the scanner's **point spread function**. Each voxel is really a
  weighted average over a small region, and thick slices average over more
  tissue along $z$.

**Partial volume effect.** A voxel that straddles the spleen boundary
averages spleen (~+120 HU) and fat (~−80 HU) and may read +20 HU, which looks
like neither tissue. That is plain low-pass filtering of a step edge. It's
why organ boundaries are fuzzy in the data and why expert labels disagree at
the edge.

**Nyquist applies.** Structures smaller than about two voxels can't be
resolved. With 5 mm slices, the thin tip of the spleen may be blurred across
slices. No amount of clever modelling recovers information the sampling
threw away.

## 2.2 Step 1: put the axes in a standard orientation

Different scanners and export tools store the axes in different orders and
directions. `Orientationd(axcodes="RAS")` permutes and flips the array axes
(and updates the affine) so that, in every scan, index +x points to the
patient's right, +y to the front, +z to the head.

The software analogy is normalising byte order before parsing a binary
protocol. It costs nothing, and the model never has to learn that "a spleen
on the left of the array" and "a spleen on the right of the array" might be
the same anatomy saved differently.

## 2.3 Step 2: resample to a common grid

`Spacingd(pixdim=(1.5, 1.5, 2.0))` re-samples every scan so that one voxel is
always 1.5 × 1.5 × 2.0 mm. Why that matters: the network will look at
96 × 96 × 96-voxel crops. After resampling, a crop **always covers the same
physical region** (144 × 144 × 192 mm), roughly a box big enough to contain a
whole spleen. Without resampling, the same crop size would cover a much
smaller region on a high-resolution scan, and the network would have to
learn every organ at every scale.

Resampling is the classic DSP pipeline: **reconstruct** a continuous signal
from samples, then **re-sample** it on the new grid. The reconstruction
kernel is the interpolation method:

| Method | DSP equivalent | Used for |
|---|---|---|
| Nearest neighbour | Zero-order hold (box kernel) | **Labels** |
| Linear (trilinear in 3D) | First-order hold (triangle kernel) | **Images** |

Why the label map **must** use nearest neighbour: labels are class IDs, not
quantities. Interpolating between label 0 and label 1 gives 0.5, which means
nothing, just as averaging two `enum` values is meaningless. Nearest
neighbour keeps every label voxel exactly 0 or 1.

A note for the DSP purist: when you **downsample**, say from 0.7 mm to
1.5 mm in-plane, you should low-pass filter first to avoid aliasing. MONAI's
resampler interpolates directly without a separate anti-aliasing filter. For
roughly 2× downsampling of CT, which the scanner's PSF has already smoothed,
the effect is minor, and it is the standard recipe. Along $z$ most scans are
**upsampled**, which interpolates but adds no information.

## 2.4 Step 3: intensity windowing

CT values span about −1000 to +2000 HU, but the spleen and its neighbours
live in a narrow band. `ScaleIntensityRanged` does:

```math
x' = \frac{\mathrm{clip}(x,\; -57,\; 164) - (-57)}{164 - (-57)}
\quad\in [0, 1]
```

In EE terms this is a **limiter followed by a gain and offset stage**, the
same thing you do when you set an ADC's full-scale range to match the signal
you care about. The 221 HU window gets the whole [0, 1] input range, while
air (−1000) and bone (+700) are clipped to 0 and 1 because they carry no
information about the spleen. Neural networks also train best when inputs
are of order 1, which comes up again in chapter 4.

Worked example:

| Tissue | HU | After windowing |
|---|---|---|
| Air | −1000 | 0.00 (clipped) |
| Fat | −80 | 0.00 (clipped) |
| Unenhanced soft tissue | +40 | 0.44 |
| Enhanced spleen | +120 | 0.80 |
| Bone | +700 | 1.00 (clipped) |

The window values (−57, 164) come from MONAI's spleen tutorial. They are
sensible defaults and were not tuned here.

## 2.5 Step 4: crop to the body

`CropForegroundd` finds the bounding box of all voxels whose windowed value
is > 0 (i.e. HU > −57, so anything denser than fat-ish) and crops to it. This
mostly removes the air around the patient and the scanner table. It saves
memory and compute in every later step and changes nothing about the
anatomy.

## 2.6 Training-only steps: random crops and augmentation

During training we don't feed whole scans (too big, chapter 8). Instead we
cut random 96³ crops. Because the spleen is tiny, purely random crops would
mostly contain no spleen at all, so `RandCropByPosNegLabeld` picks crop
centres **half on spleen voxels, half on background voxels**. Think of it as
importance sampling for the rare class.

Then come cheap **augmentations**: random flips, 90° rotations and small
intensity shifts. They create plausible variants of each crop, so 33 scans
look like many more. This is like generating extra test vectors by exploiting
symmetries of your system. The network should give the same answer for a
mirrored input (mirrored, of course).

## 2.7 Convolution: the operation everything is built on

You know 1D discrete convolution from signals and systems:

```math
y[n] = \sum_{m} h[m]\, x[n - m]
```

Deep learning uses the same thing in 3D, with three twists.

1. **It's actually cross-correlation**, with no kernel flip:
   $y[n] = \sum_m w[m]\, x[n + m]$. Since the weights are learned, the flip
   doesn't matter, and the name "convolution" stuck.
2. **Multi-channel in, multi-channel out.** The input has $C_\text{in}$
   channels (1 for CT, many inside the network). Each output channel has its
   own 3D kernel *per input channel*, and the results are summed:

   ```math
   y_{c}[\mathbf{n}] = b_c + \sum_{c'=1}^{C_\text{in}} \sum_{\mathbf{m} \in \{-1,0,1\}^3} w_{c,c'}[\mathbf{m}]\; x_{c'}[\mathbf{n} + \mathbf{m}]
   ```

   So a convolution layer is a **filter bank**: $C_\text{out}$ filters, each
   looking at all input channels.
3. **The taps are learned.** Nobody designs these filters. Training (chapter
   3) adjusts them to whatever helps find the spleen.

**Parameter count example.** A 3×3×3 conv from 1 channel to 48 channels has
$48 \times 1 \times 27$ weights $+ 48$ biases $= 1{,}344$ parameters. From 48
channels to 48 channels it has $48 \times 48 \times 27 + 48 = 62{,}256$.

**What learned filters look like.** Early filters usually end up resembling
things you'd design yourself: edge detectors (a 1D version is
$[-1, 0, +1]$, a derivative), blob detectors and smoothing kernels. Deeper
layers combine them into detectors for textures, then for shapes.

**Stride and padding.**
- **Stride 2** evaluates the filter at every second position: filter +
  decimate by 2. The output is half the size per axis.
- **Padding** adds zeros (or copies) around the border so the output keeps its
  size. The filter's view near the edge then includes made-up values. Keep
  that in mind, because it comes back in chapter 9 (why sliding-window
  borders are down-weighted).

**Two properties that make convolution a good fit for images.**
- **Weight sharing:** the same small filter slides everywhere, so a 3×3×3
  layer has 27 weights per channel pair no matter how big the image is.
- **Translation equivariance:** shift the input and the output shifts the
  same way. A spleen detector works wherever the spleen is. This is exactly
  the LTI property you know from signals and systems.

---

**In the code**
- The whole preprocessing chain, in order, with a comment per step:
  [`src/data.py`](../../src/data.py) → `_common_transforms`.
- Random crops and augmentation: `get_train_transforms`.
- Settings (`pixdim`, `hu_window`, `roi_size`, `num_samples`):
  [`configs/default.yaml`](../../configs/default.yaml), `data:` block.
- Tests that crops have the right shape, intensities are in range and labels
  stay binary: [`tests/test_transforms.py`](../../tests/test_transforms.py).

**Check yourself**
1. Why does a 96³ crop mean more after resampling than before?
2. What would go wrong if labels were resampled with trilinear interpolation?
3. A voxel reads 0.44 after windowing. What HU range does that correspond to?
4. How many parameters does a 3×3×3 convolution from 96 to 48 channels have?

Next: [3. Machine learning from zero →](03-machine-learning-from-zero.md)
