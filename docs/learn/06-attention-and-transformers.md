# 6. Attention and transformers

> **Summary.** A transformer treats its input as a set of vectors called
> **tokens**. **Self-attention** lets every token look up information from
> every other token through a *soft* hash-map lookup: each token emits a
> **query**, compares it to every token's **key**, and takes a weighted
> average of their **values**. It is powerful because any position can
> directly use any other. It is expensive because the cost grows with the
> **square** of the number of tokens. For a 3D image that cost is
> prohibitive, which is the problem Swin solves in the next chapter.

---

## 6.1 Tokens

A transformer's input is a sequence of $N$ vectors, each of width $C$: a
matrix $X \in \mathbb{R}^{N \times C}$. In language models a token is a
word piece. For images (the **Vision Transformer**, ViT), you cut the image
into small patches and turn each patch into one vector with a linear layer.

For us: cut the 96³ crop into 2×2×2 patches → $48^3 = 110{,}592$ tokens,
each projected to $C = 48$ numbers. "Cut into non-overlapping patches and
project each one" is exactly a convolution with kernel size = stride = 2.
That's how it's implemented.

## 6.2 Attention as a soft dictionary lookup

A hash map does `value = table[key]`: one exact match, one value returned.

Attention is the **differentiable, fuzzy** version:

1. You present a **query** vector $\mathbf{q}$.
2. You compare it with every stored **key** $\mathbf{k}_j$ by dot product,
   which measures similarity.
3. Softmax turns the similarities into weights that sum to 1.
4. You return the **weighted average of the values** $\mathbf{v}_j$.

```math
\text{output} = \sum_j \mathrm{softmax}_j\!\left(\frac{\mathbf{q}\cdot\mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j
```

If one key matches far better than the rest, softmax puts nearly all its
weight there and you get (almost) a hard lookup. If several match, you get a
blend.

In **self**-attention, every token produces its own query, key and value,
through three learned linear maps:

```math
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
\qquad\Longrightarrow\qquad
\mathrm{Attn}(X) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
```

$QK^\top$ is an $N \times N$ matrix of every token's similarity to every
other token. Each row is softmaxed and used to average the rows of $V$. The
model *learns* what to ask for ($W_Q$), how to advertise itself ($W_K$) and
what to hand over ($W_V$).

## 6.3 A worked example

Three tokens, $d = 2$. Suppose after projection:

```
Q = [[1, 0],      K = [[1, 0],      V = [[10,  0],
     [0, 1],           [0, 1],           [ 0, 10],
     [1, 1]]           [1, 1]]           [ 5,  5]]
```

Token 1's query $(1, 0)$ dotted with each key gives $(1, 0, 1)$. Divide by
$\sqrt{2}$ to get $(0.71, 0, 0.71)$, then softmax to get $(0.40, 0.20, 0.40)$.

Token 1's output is $0.40 \cdot (10, 0) + 0.20 \cdot (0, 10) + 0.40 \cdot (5, 5) = (6, 4)$.

Token 1 "asked about" the first dimension. It pulled mostly from tokens 1 and
3, whose keys point that way, and a little from token 2. Every token does
this in parallel, and the whole thing is two matrix multiplies plus a
softmax, which is ideal for GPUs.

## 6.4 Why divide by $\sqrt{d}$?

If the entries of $\mathbf{q}$ and $\mathbf{k}$ are independent with zero
mean and unit variance, their dot product (a sum of $d$ such products) has
**variance $d$**. With $d = 64$, typical scores are around ±8, and softmax of
values that spread out saturates: nearly all weight on one token, gradients
nearly zero everywhere else. Dividing by $\sqrt{d}$ brings the variance back
to 1. That keeps the softmax in its responsive range.

The EE version: normalise the signal level going into a nonlinearity so you
don't drive it into clipping.

## 6.5 Multi-head attention

One attention operation computes one kind of relationship. **Multi-head
attention** runs $h$ smaller attentions in parallel, each with its own
$W_Q, W_K, W_V$ and width $d = C/h$. It concatenates their outputs and mixes
them with one more linear layer. Different heads learn to attend to different
things, for example nearby boundaries in one head and intensity similarity in
another.

In SwinUNETR the head width stays constant at $d = 16$: stage 1 has
$C = 48$ with 3 heads, and stage 4 has $C = 384$ with 24 heads.

## 6.6 The transformer block

Attention is wrapped in a standard block, repeated many times:

```
x = x + Attention(LayerNorm(x))     # tokens exchange information
x = x + MLP(LayerNorm(x))           # each token processes what it gathered
```

- **Attention** mixes information *across tokens*.
- The **MLP** (linear $C \to 4C$, GELU, linear $4C \to C$) processes each
  token *independently*, the same weights for every token.
- **LayerNorm** and **residual connections** (chapter 4) keep a deep stack
  trainable. Putting the norm *before* each sub-layer ("pre-norm") is the
  more stable modern arrangement.

## 6.7 Where is position?

Look at the formula again: nothing in it knows where a token is. Shuffle the
tokens and you get the same outputs, shuffled. Attention is
**permutation-equivariant**. For images, position obviously matters, so it
must be injected:

- **Absolute position embeddings:** add a learned or sinusoidal vector per
  position to each token (original ViT).
- **Relative position bias:** add a learned number $B_{ij}$ to the attention
  score depending on the **offset** between tokens $i$ and $j$:
  $\mathrm{softmax}(QK^\top/\sqrt{d} + B)$. "Tokens two to the left get +0.3"
  becomes a learned preference. **Swin uses this.**

## 6.8 The catch: quadratic cost

The score matrix $QK^\top$ is $N \times N$. For our 110,592 tokens:

```math
N^2 = 110{,}592^2 \approx 1.22 \times 10^{10} \text{ scores, per head, per crop}
```

At 2 bytes each that's **~24 GB for a single attention matrix**, larger
than the T4's entire 16 GB memory, before counting the batch of 4, the other
heads or the backward pass. Computing it would also take tens of billions of
multiply-adds per layer.

So plain global attention on a 3D medical volume is out of the question.
The options are to make tokens bigger (fewer of them, coarser detail), or to
**restrict who attends to whom**. Swin restricts attention to local windows
and shifts them so information still spreads, which brings the cost down to
linear in $N$. That's the next chapter.

## 6.9 CNNs vs transformers in one table

| | CNN | Transformer |
|---|---|---|
| Who influences an output | Fixed local neighbourhood | Any token (or any token in its window), weighted by learned similarity |
| Built-in assumptions | Locality, translation equivariance | Very few (needs position information injected) |
| Data hunger | Low | High: it has to learn what a CNN assumes |
| Long-range relationships | Built up slowly through depth | Direct |
| Cost vs number of positions | Linear | Quadratic (global) or linear (windowed) |

Fewer built-in assumptions means more flexibility *and* more data needed.
With only 33 training scans, pure transformers would struggle. SwinUNETR
hedges by using windowed attention (locality comes back) plus a
convolutional decoder. Self-supervised pre-training on unlabelled CT is the
usual further step (listed in the README's extensions).

---

**In the code**
- Attention itself lives inside MONAI's `SwinUNETR`; you don't write it
  yourself. The settings that matter (number of heads per stage, window size,
  MLP ratio) are listed in the
  [architecture reference](../reference/architecture.md#1-configuration).
- `feature_size` must be divisible by 12 ([`src/model.py`](../../src/model.py)
  → `build_model`), because the width is split across 3, then 6, 12 and 24
  heads as it doubles through the stages.

**Check yourself**
1. In the worked example, compute token 2's output.
2. Why does the dot product of two random $d$-dimensional vectors have
   variance $d$?
3. What breaks if you remove all position information from an image
   transformer?
4. Roughly how much memory would one global attention matrix need for a
   64³ crop with 2×2×2 patches, in FP16?

Next: [7. Swin and SwinUNETR →](07-swin-and-swinunetr.md)
