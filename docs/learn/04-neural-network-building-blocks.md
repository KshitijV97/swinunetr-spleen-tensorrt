# 4. Neural network building blocks

> **Summary.** A neural network is a stack of a few simple building blocks:
> - **linear layers** (matrix multiplies),
> - **nonlinearities** (without them the stack collapses to one matrix),
> - **softmax** to turn scores into probabilities,
> - **cross-entropy** to score those probabilities,
> - **normalisation** (automatic gain control between layers),
> - **residual connections** (bypass paths that keep deep stacks trainable).
>
> SwinUNETR uses every one of them.

---

## 4.1 Tensors and shapes

Everything is a **tensor**: an n-dimensional array. Shape conventions in
this repo:

| Tensor | Shape | Meaning |
|---|---|---|
| One training batch | `(4, 1, 96, 96, 96)` | (batch, channels, depth, height, width) |
| Network output | `(4, 2, 96, 96, 96)` | 2 scores per voxel: background, spleen |
| Labels | `(4, 1, 96, 96, 96)` | 0 or 1 per voxel |
| Inside the transformer | `(4, 48, 48, 48, 48)` | 48 channels on a 48³ grid (MONAI moves channels last internally) |

Getting shapes right is the "type checking" of deep learning. Most bugs show
up as shape mismatches.

## 4.2 The linear layer

```math
\mathbf{y} = W\mathbf{x} + \mathbf{b}, \qquad W \in \mathbb{R}^{m \times n}
```

That's $mn + m$ parameters. A convolution is a linear layer with weight
sharing and local connectivity (chapter 2). In transformers, linear layers
are applied to every token independently: the same $W$ for every position,
like a 1×1×1 convolution.

## 4.3 Why nonlinearities are mandatory

Two linear layers in a row are one linear layer:
$W_2(W_1 \mathbf{x}) = (W_2 W_1)\mathbf{x}$. Stack a hundred and you still have
one matrix. That's a linear system, and it can't represent "is this voxel
inside a curved organ?". So between linear layers we insert a simple
elementwise nonlinear function:

| Name | Formula | Where it's used here |
|---|---|---|
| ReLU | $\max(0, x)$ | The worked examples in this track |
| LeakyReLU | $x$ if $x > 0$, else $0.01x$ | CNN decoder of SwinUNETR |
| GELU | $x \cdot \Phi(x)$ ($\Phi$ = Gaussian CDF) | MLPs inside the Swin transformer blocks |

The EE intuition: a network of linear stages with nonlinear "diodes" in
between is a piecewise-linear function approximator (for ReLU-type
functions). With enough pieces it can approximate any reasonable function.
That's the **universal approximation** idea.

## 4.4 Logits and softmax

The network's last layer outputs two unbounded real numbers per voxel,
called **logits**: $z_0$ (background score) and $z_1$ (spleen score). To get
probabilities we apply **softmax**:

```math
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
```

Worked example: $z = (1.0,\; 3.0)$ gives
$p = \left(\frac{e^1}{e^1 + e^3},\, \frac{e^3}{e^1 + e^3}\right) = (0.12,\; 0.88)$.

- Softmax is a "soft argmax". As the gap between logits grows, it approaches
  a hard one-hot pick of the largest.
- Physicists will recognise the **Boltzmann distribution**, with logits as
  negative energies.
- **Numerical stability:** $e^{1000}$ overflows. Implementations compute
  $e^{z_k - \max_j z_j}$ instead, which gives the same result with no overflow.
  This matters again when we run in 16-bit floats (chapters 8 and 10).

At prediction time we skip softmax and take **argmax** of the logits (same
answer, less work). Each voxel gets the class with the higher score.

## 4.5 Cross-entropy: scoring probabilities

Given the true class $y$ and predicted probabilities $p$, the cross-entropy
loss is

```math
L_\text{CE} = -\log p_y
```

- Confident and right ($p_y = 0.99$): $L = 0.01$.
- Unsure ($p_y = 0.5$): $L = 0.69$.
- Confident and wrong ($p_y = 0.01$): $L = 4.6$.

It comes from **maximum likelihood**: minimising $-\log p_y$ over a dataset
maximises the probability the model assigns to the true labels.

Its gradient with respect to the logits is beautifully simple:

```math
\frac{\partial L_\text{CE}}{\partial z_k} = p_k - \mathbb{1}[k = y]
```

In the example above, if the voxel is truly spleen ($y = 1$), the gradient is
$(0.12 - 0,\; 0.88 - 1) = (0.12,\; -0.12)$: push the spleen score up and the
background score down, in proportion to how wrong we are. The gradient is
bounded and well-behaved, which is why cross-entropy is the default
classification loss. Our loss is cross-entropy **plus** a Dice term (chapter
8).

## 4.6 Normalisation layers: automatic gain control

As signals pass through many layers, their scale can drift. Activations
grow or shrink layer after layer, gradients follow, and training becomes
unstable. Normalisation layers re-centre and re-scale activations, then apply
a learned gain and offset:

```math
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta
```

The kinds differ in **which values $\mu$ and $\sigma$ are computed over**:

| Kind | Statistics over | Used here | Why |
|---|---|---|---|
| BatchNorm | the whole batch, per channel | no | Unreliable with tiny batches (we have 4 crops) |
| **InstanceNorm** | one sample, per channel, over all voxels | CNN decoder | Batch-independent; standard in medical segmentation |
| **LayerNorm** | one token, over its channels | every transformer block | The standard transformer choice |

The EE analogy is **automatic gain control**: hold the signal level steady so
the next stage operates in its sweet spot. LayerNorm involves a variance and a
square root, which are the operations most likely to overflow in 16-bit
floating point. Remember that for chapter 10.

## 4.7 Residual connections: a bypass path

A residual (skip) connection computes

```math
\mathbf{y} = \mathbf{x} + F(\mathbf{x})
```

so each block only learns a **correction** to its input. Two benefits:

1. **Gradients flow.** $\partial \mathbf{y}/\partial \mathbf{x} = I + \partial F/\partial \mathbf{x}$.
   The identity term carries gradient straight through, even if $F$'s
   gradient is tiny. Without it, gradients through many layers tend to
   vanish (a product of many factors < 1). That's like a long chain of
   attenuators.
2. **Easy default.** If a block isn't useful, it can learn $F \approx 0$ and
   pass the input through unchanged.

Every Swin block has two residual connections, and the CNN decoder blocks
are residual too. (Don't confuse these with U-Net "skip connections" in
chapter 5. Those carry features across the network from encoder to decoder,
not around a single block.)

## 4.8 Initialisation, dropout and other small things

- **Initialisation.** Weights start random, scaled so signals neither explode
  nor vanish in the first forward pass (e.g. variance $\propto 1/\text{fan-in}$).
  MONAI handles this.
- **Dropout / stochastic depth.** Randomly drop activations or whole blocks
  during training as regularisation. Supported by SwinUNETR, but **turned
  off here** (`dropout_path_rate: 0.0`).
- **Train vs eval mode.** `model.train()` and `model.eval()` switch layers
  that behave differently while learning (dropout, BatchNorm). Our code calls
  `eval()` before validation and inference.

---

**In the code**
- Softmax and cross-entropy happen inside MONAI's `DiceCELoss`
  ([`src/train.py`](../../src/train.py), `loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)`).
- Argmax at prediction time: [`src/evaluation.py`](../../src/evaluation.py)
  → `DiceScorer` (`AsDiscrete(argmax=True, ...)`).
- `model.eval()` before inference: `validate` in `src/train.py`, and
  `load_checkpoint` in [`src/model.py`](../../src/model.py).

**Check yourself**
1. Compute softmax of $(2, 2)$. What does the result tell you about the model?
2. Why can't you build SwinUNETR without nonlinearities?
3. Why is BatchNorm a poor choice with a batch of 4?
4. What does the residual connection's identity term do for gradients?

Next: [5. CNNs and the U-Net →](05-cnns-and-unet.md)
