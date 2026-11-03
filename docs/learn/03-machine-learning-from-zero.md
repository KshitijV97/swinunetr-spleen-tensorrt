# 3. Machine learning from zero

> **Summary.** Machine learning means choosing a function from a huge
> parameterised family by **minimising a loss** on example data.
> - The minimisation is **gradient descent**, the same idea as the LMS
>   adaptive filter.
> - The gradients come from **backpropagation**, which is the chain rule
>   applied backwards through a computation graph (reverse-mode automatic
>   differentiation).
> - The hard part is not fitting the training data. It is **generalising** to
>   data you haven't seen, which is why we hold out validation data.

---

## 3.1 The core idea: learning is curve fitting

Traditional programming: you write the function. Machine learning: you write
a **family of functions** $f_\theta$ with adjustable parameters $\theta$,
plus examples $(x_i, y_i)$ of correct input→output pairs, and let an
algorithm find the $\theta$ that makes $f_\theta(x_i) \approx y_i$.

For us:
- $x$ is a 96³ crop of CT (884,736 numbers),
- $y$ is the matching 96³ crop of labels,
- $f_\theta$ is SwinUNETR, and $\theta$ is its **62.2 million** weights.

"How wrong is the model" is measured by a **loss function** $L(\theta)$, a
single number that is small when predictions match labels. Training means:

```math
\theta^\star = \arg\min_\theta \; \frac{1}{N}\sum_{i=1}^{N} L\big(f_\theta(x_i),\, y_i\big)
```

## 3.2 A tiny example you can do by hand

Fit a line $f(x) = w x + b$ to three points: $(0, 1), (1, 3), (2, 5)$. The
answer is obviously $w = 2, b = 1$, but let's *learn* it.

**Loss:** mean squared error, $L = \frac{1}{3}\sum_i (w x_i + b - y_i)^2$.

**Gradient** (how $L$ changes when you nudge each parameter):

```math
\frac{\partial L}{\partial w} = \frac{2}{3}\sum_i (w x_i + b - y_i)\,x_i,
\qquad
\frac{\partial L}{\partial b} = \frac{2}{3}\sum_i (w x_i + b - y_i)
```

**Start** at $w = 0, b = 0$. The errors are $(0-1, 0-3, 0-5) = (-1, -3, -5)$.

- $\partial L/\partial w = \frac{2}{3}(-1\cdot 0 - 3 \cdot 1 - 5 \cdot 2) = -8.67$
- $\partial L/\partial b = \frac{2}{3}(-1 - 3 - 5) = -6$

**Step** downhill with learning rate $\eta = 0.1$:
$w \leftarrow 0 - 0.1 \times (-8.67) = 0.87$, $b \leftarrow 0 - 0.1 \times (-6) = 0.6$.

The loss drops from 11.67 to about 3.2. Repeat a few hundred times and you
converge to $w = 2, b = 1$. **That loop is all of deep learning.** The rest is
making $f$ expressive, computing gradients efficiently, and keeping the
procedure stable.

## 3.3 Gradient descent is an adaptive filter

The update rule is

```math
\theta \leftarrow \theta - \eta\, \nabla_\theta L
```

If you did any adaptive signal processing, you've seen this before. The
**LMS filter** updates its taps with $\mathbf{w} \leftarrow \mathbf{w} + \mu\, e[n]\, \mathbf{x}[n]$.
That *is* gradient descent on the instantaneous squared error $e^2$. The
learning rate $\eta$ plays exactly the role of LMS's step size $\mu$:

| $\eta$ too large | $\eta$ too small | $\eta$ about right |
|---|---|---|
| Overshoots, oscillates, diverges (loss becomes `nan`) | Converges painfully slowly | Fast, stable decrease |

Two differences from textbook LMS:

1. **The loss surface is not a nice bowl.** For a linear filter with MSE it's
   a quadratic with one minimum. For a deep network it's a high-dimensional,
   non-convex landscape. Surprisingly, in very high dimensions most "bad"
   points are saddles rather than traps, and gradient descent with some noise
   finds good solutions in practice.
2. **We use smarter update rules than plain gradient descent.** Our optimiser,
   AdamW, adds momentum (a low-pass filter on the gradient) and per-parameter
   step-size normalisation (automatic gain control). Chapter 8 covers it.

## 3.4 Mini-batches, iterations and epochs

Computing the exact gradient over all data before every step is expensive.
**Stochastic gradient descent** estimates it from a small random
**mini-batch**: noisier, but much cheaper per step. The noise even helps
escape poor regions.

Vocabulary, with our numbers:

| Term | Meaning | Here |
|---|---|---|
| Batch | Examples per gradient step | 4 crops (all from 1 scan) |
| Step / iteration | One forward + backward + update | |
| Epoch | One pass over the training set | 33 steps (one per training scan) |
| Total training | | 150 epochs ≈ 4,950 steps |

## 3.5 Backpropagation: the chain rule, run backwards

Our model has 62 million parameters. We need $\partial L / \partial \theta_j$
for **every one of them, at every step**. Finite differences (nudge each
weight and re-run) would take 62 million forward passes per step. That's
impossible.

**Backpropagation** gets all of them for roughly the cost of **two** extra
forward passes. The idea: the model is a **computation graph** of simple
operations, each with a known local derivative. The chain rule composes
them.

**Worked example.** One input $x = 2$, target $y = 1$, two weights:

```
h = w1 * x         (w1 = 0.5)  -> h = 1.0
a = relu(h)                    -> a = 1.0
p = w2 * a         (w2 = 3.0)  -> p = 3.0
L = (p - y)^2                  -> L = 4.0
```

**Backward pass** (start from $\partial L/\partial L = 1$ and walk back):

| Quantity | Local rule | Value |
|---|---|---|
| $\partial L / \partial p$ | $2(p - y)$ | 4.0 |
| $\partial L / \partial w_2$ | $\partial L/\partial p \cdot a$ | 4.0 |
| $\partial L / \partial a$ | $\partial L/\partial p \cdot w_2$ | 12.0 |
| $\partial L / \partial h$ | $\partial L/\partial a \cdot \mathbb{1}[h > 0]$ | 12.0 |
| $\partial L / \partial w_1$ | $\partial L/\partial h \cdot x$ | 24.0 |

Each node only needs its own local derivative and the gradient arriving from
downstream. Note that the backward pass reused $a$, $h$ and $x$ from the
forward pass. **Training must store intermediate values (activations)**,
which is why training needs far more GPU memory than inference (chapter 8).

**Three ways to see it:**
- **Software:** reverse-mode automatic differentiation. PyTorch records the
  graph as your Python code runs, and `loss.backward()` walks it in reverse,
  filling each parameter's `.grad`.
- **EE:** it's sensitivity analysis. If you've seen the *adjoint network*
  method for computing all circuit sensitivities in one extra simulation
  (Tellegen's theorem), backprop is the same trick.
- **Complexity:** reverse mode is efficient when there are many inputs
  (parameters) and one output (the loss). That is exactly our situation.

## 3.6 What the training loop looks like in PyTorch

Stripped of extras, [`src/train.py`](../../src/train.py) does this:

```python
for epoch in range(epochs):
    for batch in train_loader:                 # 4 crops + labels
        optimizer.zero_grad()                  # clear old gradients
        logits = model(batch["image"])         # forward pass (graph recorded)
        loss = loss_fn(logits, batch["label"]) # one number
        loss.backward()                        # backprop: fill p.grad for every weight
        optimizer.step()                       # update: theta -= lr * (adjusted) grad
```

The real loop adds mixed precision (`autocast`, `GradScaler`), a
learning-rate schedule and periodic validation. Chapter 8 walks through each
addition.

## 3.7 Generalisation: the actual hard part

Fitting training data is easy. A 62-million-parameter model can memorise 33
scans. What we want is good predictions on **scans it has never seen**.

**Overfitting** is fitting the noise and quirks of the training set. The EE
picture: fit a 40th-order polynomial through 41 noisy points. It passes
through every point and swings wildly between them.

So why doesn't a 62 M-parameter model trained on 33 scans just memorise?

1. **Each scan carries a lot of supervision.** Every voxel is labelled, so
   one scan provides millions of (correlated) training signals, not one.
2. **Random crops and augmentation** mean the model rarely sees exactly the
   same input twice.
3. **Architectural priors.** Convolutions assume locality and translation
   equivariance. Windowed attention assumes locality first. These priors
   shrink the space of functions the model can easily represent.
4. **Regularisation.** Weight decay (chapter 8) gently pulls weights toward
   zero, favouring simpler functions.

Still, the risk is real, and the only way to know is to **measure on held-out
data**.

## 3.8 Train, validation and test splits

| Split | Used for | Here |
|---|---|---|
| **Training** | Computing gradients | 33 scans |
| **Validation** | Deciding things: which checkpoint to keep, which hyperparameters work | 8 scans |
| **Test** | One final, untouched estimate of real-world performance | **none** |

Every time you use the validation set to *choose* something, you leak a
little information about it into your model, and the validation score
becomes slightly optimistic. A separate test set fixes that. With only 41
labelled scans this project doesn't have one, and the README says so. The
proper fix is **k-fold cross-validation**: rotate which scans are held out
and average the results.

The split is **deterministic**: a fixed random seed shuffles the sorted file
list, so the same 8 scans are held out on every machine. Reproducibility is a
software property you already value. In ML it's also a scientific
requirement.

## 3.9 Parameters vs hyperparameters

- **Parameters** ($\theta$): learned by gradient descent. These are the 62.2 M
  weights.
- **Hyperparameters:** chosen by you before training. Examples are the
  learning rate, number of epochs, crop size and network width. They live in
  [`configs/default.yaml`](../../configs/default.yaml). Choosing them well is
  partly experience and partly experiment, and every experiment you judge on
  the validation set nudges it toward being optimistic (§3.8).

---

**In the code**
- The training loop: [`src/train.py`](../../src/train.py) → `train`.
- Deterministic split: [`src/data.py`](../../src/data.py) → `split_datalist`,
  tested in [`tests/test_transforms.py`](../../tests/test_transforms.py).
- Hyperparameters: [`configs/default.yaml`](../../configs/default.yaml).

**Check yourself**
1. In the line-fitting example, what happens with $\eta = 1.0$? (Try one step.)
2. Why does backprop need to keep the forward pass's intermediate values?
3. Why is "the best validation score during training" slightly optimistic?
4. Map each piece of the LMS update $\mathbf{w} \leftarrow \mathbf{w} + \mu e \mathbf{x}$
   onto the gradient-descent update.

Next: [4. Neural network building blocks →](04-neural-network-building-blocks.md)
