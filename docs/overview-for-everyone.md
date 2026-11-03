# The project explained for everyone

This page explains what this repository does and why, for readers with **no
background** in medicine, engineering, programming or artificial intelligence.
It takes about 15 minutes to read. Every technical word is explained the first
time it appears, and there is an FAQ at the end.

If you want the 2-minute version, the
[plain-English section of the README](../README.md#what-this-project-does-in-plain-english)
covers it.

---

## Contents

1. [The story in one minute](#1-the-story-in-one-minute)
2. [What a CT scan is](#2-what-a-ct-scan-is)
3. [What the spleen is, and why surgeons care](#3-what-the-spleen-is-and-why-surgeons-care)
4. [What "segmentation" means](#4-what-segmentation-means)
5. [How you teach a computer to do it](#5-how-you-teach-a-computer-to-do-it)
6. [The model, without the maths](#6-the-model-without-the-maths)
7. [Making it fast](#7-making-it-fast)
8. [How the results are judged](#8-how-the-results-are-judged)
9. [How to read the results tables](#9-how-to-read-the-results-tables)
10. [Honest limits](#10-honest-limits)
11. [What was built here, and what was reused](#11-what-was-built-here-and-what-was-reused)
12. [FAQ](#12-faq)

---

## 1. The story in one minute

Doctors take 3D X-ray pictures of the body called CT scans. Surgeons, and
increasingly surgical robots, need to know exactly where each organ is in
those pictures. Outlining organs by hand is slow, so we want a computer to do
it.

This project:

1. **teaches** a computer program to outline the spleen in CT scans, using 41
   scans that experts have already outlined;
2. **tests** it on scans it has never seen;
3. **speeds it up** with NVIDIA's optimisation software, so it could
   realistically run inside medical or robotic equipment;
4. **measures** how much faster it became, and whether it stayed accurate.

## 2. What a CT scan is

A **CT (computed tomography) scan** is made by an X-ray machine that rotates
around the patient. A computer combines the X-rays into a 3D picture.

- Think of the body as a **loaf of bread**. A CT scan is that loaf cut into
  thin **slices**, with a black-and-white photo of each slice. Stack the
  photos back up and you have the whole 3D body.
- Each photo is a grid of **pixels**. Because the photos are stacked in 3D,
  each pixel is really a tiny box, called a **voxel** ("volume pixel").
- **Brightness means density.** Air is black, fat is dark grey, organs and
  muscle are mid-grey, and bone is white. The number stored in each voxel is
  called a **Hounsfield unit** (water is 0, air is −1000).
- The scans used here were taken after injecting a **contrast agent**,
  which makes blood-rich organs (like the spleen) show up more brightly.

In this dataset, each scan is **512 × 512 pixels per slice**, with **31 to
168 slices**, so a single scan holds somewhere between about 8 and 44
million voxels.

## 3. What the spleen is, and why surgeons care

The **spleen** is an organ about the size of a fist. It sits on the upper
left side of the belly, tucked under the ribs (in the figure in the README,
it appears on the right side of the image, because doctors view scans as if
facing the patient). It filters blood and helps fight infection.

Why it matters for surgery:

- It is **soft and full of blood**, so damage can cause serious bleeding.
- During some abdominal operations the spleen is close to where the surgeon
  is working, and accidental injury to it is a recognised complication.
- Software that knows the spleen's exact 3D shape can help plan an
  operation, show the surgeon where it is even when it is hidden, or let a
  surgical robot keep its instruments away from it (a "virtual no-go zone").

The spleen is also a good **learning problem**: it is a well-studied public
benchmark, so effort can go into doing the engineering properly rather than
collecting data.

## 4. What "segmentation" means

**Segmentation** means colouring in every voxel that belongs to something,
here the spleen. It is like a **colouring book**: the computer must colour
exactly the spleen's area on every slice, and nothing else.

The result is a second 3D grid the same size as the scan, where each voxel
holds **1 (spleen)** or **0 (not spleen)**. Stack the coloured slices and you
get a 3D model of the organ.

It is harder than it sounds:

- The spleen is **tiny compared with the scan**. On average it is only about
  **0.5% of the voxels** (between 0.08% and 1.3% across the 41 scans).
- It sits next to organs with a **similar grey level** (stomach, kidney,
  liver), so the computer must use shape and position, not just brightness.
- Its **size and shape vary a lot** between people.

## 5. How you teach a computer to do it

Nobody writes rules like "the spleen is the grey blob on the left". Instead,
the computer **learns from examples**. This is **machine learning**.

1. **Examples with answers.** The dataset has 41 scans where experts
   outlined the spleen by hand. These outlines are the **labels**, the
   answer key.
2. **Split into practice and exam.** 33 scans are used for learning
   (**training**). 8 are kept aside and never learned from (**validation**).
   Testing on scans the computer has not seen is the only fair test, the same
   way a real exam uses new questions.
3. **Practice, compare, adjust.** The program (the **model**) starts out
   making random guesses. For each training scan it guesses an outline,
   measures how wrong it was, and slightly adjusts its millions of internal
   settings (called **parameters** or **weights**) to be a bit less wrong
   next time. One pass over all 33 scans is an **epoch**. This project runs
   up to 150 epochs.
4. **Keep the best version.** Every few epochs, the model is tested on the 8
   validation scans, and the best-scoring version is saved.

A few practical tricks are explained in the technical docs. In short:

- the model learns from small **3D cubes** cut out of each scan, because a
  whole scan is too big to process in one go;
- half the cubes are picked to contain spleen, so it gets enough examples of
  what it is looking for;
- cubes are randomly flipped and rotated so the model does not memorise
  details that do not matter.

## 6. The model, without the maths

The model is called **SwinUNETR**. Its name is a combination of two ideas:

- **"Swin" (a Swin Transformer).** It looks at the scan in small 3D
  **windows** and works out how each part relates to the others nearby. Then
  the windows **shift**, so information spreads across the whole scan. This
  is how it learns things like "the spleen is next to the left kidney and
  under the ribs". Transformers are the same family of technology behind
  modern AI chatbots, adapted here to 3D images.
- **"UNETR" (a U-Net-style decoder).** After understanding the big picture
  at low detail, it builds the outline back up step by step to full
  detail, re-using fine detail it saw earlier so the edges come out sharp.
  Drawn as a diagram, that "zoom out, then zoom back in" shape looks like a
  **U**.

It has about **62 million** adjustable parameters. To segment a whole scan,
the model slides a 3D window across the scan piece by piece and stitches the
results together (**sliding-window inference**).

## 7. Making it fast

A trained model is just a large amount of arithmetic. How fast it runs
depends on the hardware and how well the arithmetic is organised.

- **GPU.** A graphics chip (GPU) does thousands of simple calculations at
  once, which is exactly what AI needs. The measurements use an **NVIDIA
  T4**, a common data-centre GPU that Google Colab provides for free.
- **ONNX: a standard file.** The trained model is saved in **ONNX** format,
  which is to AI models roughly what PDF is to documents. The project
  **checks the saved copy gives the same answers as the original** before
  trusting it.
- **TensorRT: an optimiser.** NVIDIA's TensorRT takes the ONNX file and
  rewrites the model specifically for one GPU. It merges steps that can be
  done together, tries several ways of doing each calculation and keeps the
  fastest. A good analogy is a **translator who rewrites a recipe for one
  particular kitchen**: same dish, fewer wasted moves.
- **FP16: fewer digits.** Computers normally do this arithmetic with about 7
  significant digits (**FP32**). TensorRT can use about 3 digits (**FP16**)
  where that is safe, which is much faster on this GPU. Fewer digits could
  in principle change the answer slightly, which is why the project
  **re-checks accuracy after optimising**.
- **AMP: the middle option.** PyTorch (the software the model was built in)
  can also use FP16 in places, without TensorRT. That is called **AMP**. It
  is included in the comparison so readers can see how much of the gain
  comes from fewer digits and how much from TensorRT's other optimisations.

## 8. How the results are judged

### Accuracy: the Dice score

Put the computer's outline and the expert's outline on top of each other.
The **Dice score** measures how much they overlap:

```
Dice = 2 × (area both outlines agree on) ÷ (area of computer's outline + area of expert's outline)
```

- **1.0** means a perfect match. **0** means no overlap at all.
- It ignores the huge "not spleen" background, so a lazy program that says
  "no spleen anywhere" scores 0, not 99%.
- The score reported is the **average over the 8 validation scans**.

As a rough guide only: well-trained systems on this public dataset typically
score in the mid-0.9s. That is context, not a claim about this project. Its
own numbers will appear once the full run is done.

### Speed: latency

**Latency** is how long it takes to outline the spleen in **one whole scan**.
To be fair and precise, the timing:

- starts after the scan is already loaded and prepared (so it measures the AI
  part only, not reading files);
- skips the first couple of runs, which are always slower while the GPU
  "warms up";
- waits until the GPU has really finished before stopping the clock;
- is repeated 3 times over all 8 validation scans.

## 9. How to read the results tables

Once the full run is done, the main table in the README will look like this,
with real numbers in place of *TBD*:

| Column | What it tells you |
|---|---|
| **Backend** | Which version of the model: the original (**PyTorch FP32**), the original with fewer digits (**PyTorch AMP**), or the optimised one (**TensorRT FP16**) |
| **Mean latency / volume (ms)** | Average time per scan, in milliseconds (1,000 ms = 1 second). Lower is better |
| **p95 (ms)** | The "slow end": 95% of scans took this long or less. Bigger scans take longer, so this shows the worst typical case |
| **Speed-up vs FP32** | How many times faster than the original. For example, 3.00× would mean three times as fast |
| **Mean Dice** | Accuracy of the outlines, from 0 to 1. Ideally the same for all three rows |

The ideal outcome is: **TensorRT row much faster, Dice essentially
unchanged.** If the Dice drops, that is reported, not hidden.

## 10. Honest limits

Good science says clearly what the results do *not* show:

- **Small dataset.** 41 scans from one hospital and one group of patients
  (people being treated for cancer that had spread to the liver). It might
  not work as well on scans from other hospitals or machines.
- **Only 8 test scans.** That is enough to see whether it works, but the
  average could move noticeably with a different 8.
- **One GPU.** Speeds are measured on one GPU type on Google Colab. Other
  hardware will give different numbers.
- **AI part only.** The timing excludes loading and preparing the scan.
- **Not for patients.** This is a research and learning project, not a
  medical device.

The [technical README section](../README.md#limitations) lists these in
precise terms.

## 11. What was built here, and what was reused

Being clear about credit matters:

**Reused (open-source and public):**
- the **SwinUNETR model design** (published research, implemented in the
  open-source **MONAI** library);
- the **dataset** (the public Medical Segmentation Decathlon);
- **PyTorch**, **ONNX** and **NVIDIA TensorRT** as tools.

**Built in this project:**
- the complete pipeline that connects these: data download, checks and
  splitting; preprocessing; training with validation and checkpointing;
- ONNX export with an automatic check that the exported model matches the
  original;
- the TensorRT integration, including making the optimised model fit into the
  same inference code as the original;
- a careful, documented benchmark that measures speed and accuracy the same
  way for every version;
- automated tests, a laptop "smoke test", a one-click Google Colab notebook,
  and this documentation.

## 12. FAQ

**Is this used on real patients?**
No. It is a research and portfolio project and has not been clinically
tested.

**Where do the scans come from? Is patient privacy protected?**
From the **Medical Segmentation Decathlon**, a public research dataset of
anonymised scans released for exactly this kind of work. The scans are not
stored in this repository; the code downloads them from the official source.

**Why the spleen and not something harder?**
It is a well-known benchmark, so the focus can be on doing the whole
engineering pipeline correctly and measuring honestly. The same code would
work for other organs with a different dataset.

**Why aren't there any results yet?**
Training the full-size model needs a powerful NVIDIA GPU and can take hours. The code
has been fully tested on a laptop with a tiny version of the model, and the
full run is done on Google Colab. Until that run happens, the tables say
*TBD* rather than showing guessed numbers.

**What is the "smoke test"?**
A deliberately tiny run (a very small model, 2 scans, 2 rounds of training)
that only checks every step of the pipeline works from start to finish. Its
accuracy is meaningless by design and is never reported.

**Can I run it myself?**
Yes. The easiest way is the **"Open in Colab"** button at the top of the
README, which runs everything in your browser on a free Google GPU. The
[Colab guide](reference/colab.md) walks through it.

**Do I need an NVIDIA GPU?**
Only for full training and the TensorRT part. The tests and the smoke test
run on an ordinary laptop, including a Mac.

**Why would a robot need this to be fast?**
Planning before surgery can wait minutes. But anything that updates while
the surgeon or robot is working, such as re-aligning the organ map as things
move, needs answers quickly and predictably. Fast, optimised inference on the
robot's own GPU is a prerequisite for that.

**What does "TBD: run notebook" mean?**
"To be determined": the number will be filled in after the Colab notebook is
run. Nothing is estimated in the meantime.

---

**Want to go deeper?** The [learning track](learn/README.md) explains all of
this from first principles for engineers, and the
[technical reference](reference/README.md) has the exact details. Unfamiliar
words are in the [glossary](learn/glossary.md).
