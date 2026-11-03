# 13. Where to go next

A short, opinionated list, ordered within each section from "start here" to
"go deeper". Most items are free.

---

## Machine learning foundations

1. **3Blue1Brown, *Neural Networks*** (YouTube series). The best visual
   intuition for gradient descent and backprop. Watch it in an evening.
2. **Andrej Karpathy, *Neural Networks: Zero to Hero*** (YouTube). Builds
   backprop and then a transformer from scratch in Python. Ideal for a
   software engineer, because you write every line.
3. **Simon Prince, *Understanding Deep Learning*** (MIT Press, 2023; free
   PDF at udlbook.github.io). A modern, clear textbook with excellent
   figures. Chapters on CNNs, transformers and training map directly onto
   chapters 3–8 here.
4. **Zhang et al., *Dive into Deep Learning*** (d2l.ai). Runnable notebooks
   for everything.
5. **Goodfellow, Bengio & Courville, *Deep Learning*** (deeplearningbook.org).
   The classic reference. Denser and more mathematical.

## Transformers and Swin

1. **Jay Alammar, *The Illustrated Transformer*** (blog post). Attention in
   pictures.
2. **Vaswani et al., *Attention Is All You Need*** (2017). The original
   transformer paper.
3. **Dosovitskiy et al., *An Image is Worth 16x16 Words*** (ViT, 2021).
   Transformers for images.
4. **Liu et al., *Swin Transformer*** (ICCV 2021). Windows and shifts (our
   encoder).
5. **Hatamizadeh et al., *Swin UNETR*** (2022), and **Tang et al.,
   *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image
   Analysis*** (CVPR 2022). Our model, and how to pre-train it.

## Medical imaging

1. **Radiopaedia.org.** Free, peer-reviewed articles and cases. Search
   "spleen", "Hounsfield unit", "CT windowing", "contrast phases".
2. **Kak & Slaney, *Principles of Computerized Tomographic Imaging*** (free
   online). CT reconstruction written for engineers: Radon transform,
   Fourier slice theorem, filtered back-projection.
3. **3D Slicer** (slicer.org). A free viewer. Open a `.nii.gz` from the
   dataset alongside its label and scroll through axial, coronal and
   sagittal views. Ten minutes of this teaches more anatomy than an hour of
   reading.
4. **Antonelli et al., *The Medical Segmentation Decathlon*** (Nature
   Communications, 2022). Our dataset and the benchmark's design.
5. **Maier-Hein et al., *Metrics Reloaded*** (Nature Methods, 2024). How to
   choose segmentation metrics properly, and why Dice alone is not enough.

## Medical segmentation in practice

1. **MONAI tutorials** (github.com/Project-MONAI/tutorials). Start with the
   spleen segmentation notebook (the source of our preprocessing defaults),
   then the BTCV SwinUNETR tutorial.
2. **Ronneberger et al., *U-Net*** (2015) and **Çiçek et al., *3D U-Net***
   (2016).
3. **Isensee et al., *nnU-Net*** (Nature Methods, 2021). The self-configuring
   baseline any new method should be compared against. Its paper is also a
   masterclass in what actually matters in practice.

## GPUs, performance and deployment

1. **Horace He, *Making Deep Learning Go Brrrr From First Principles*** (blog
   post). Compute-bound vs memory-bound vs overhead-bound, explained for
   practitioners.
2. **Williams, Waterman & Patterson, *Roofline*** (CACM, 2009). The original
   roofline-model paper. Short and readable.
3. **Micikevicius et al., *Mixed Precision Training*** (ICLR 2018). Loss
   scaling and FP32 master weights.
4. **Kirk, Hwu & El Hajj, *Programming Massively Parallel Processors***. CUDA
   and GPU architecture properly.
5. **NVIDIA TensorRT Developer Guide** and **Polygraphy** (for comparing
   layer outputs between frameworks when FP16 goes wrong).

## Optimisation

1. **Kingma & Ba, *Adam*** (ICLR 2015) and **Loshchilov & Hutter, *Decoupled
   Weight Decay Regularization*** (AdamW, ICLR 2019).
2. **Loshchilov & Hutter, *SGDR*** (ICLR 2017). Cosine learning-rate schedules.

## A learning path through this repo

1. Read chapters 1–3, then run `pytest -q` and the synthetic smoke test.
2. Open a real scan and label in 3D Slicer (after `python -m src.data --download`).
3. Read chapters 4–7 with [`src/model.py`](../../src/model.py) and MONAI's
   `swin_unetr.py` side by side. Print the shapes of `hidden_states_out` for
   a 96³ input.
4. Read chapter 8 while the Colab notebook trains. Watch the log against
   §8.10.
5. Read chapters 9–11 when the benchmark finishes, and check the real numbers
   against the back-of-envelope bound in §10.3.
6. Do the exercises at the end of [chapter 12](12-software-engineering-of-this-repo.md).

Back to the [learning track index](README.md) · [Glossary](glossary.md)
