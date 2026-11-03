# 1. The medical problem

> **Summary.** The spleen is a fist-sized, blood-filled organ tucked under the
> left ribs. Surgeons working nearby must avoid it, and doctors often want to
> measure its size. A CT scanner measures how strongly each small cube of the
> body absorbs X-rays and stores the result as a 3D array of numbers in
> **Hounsfield units**. Our task is to label every element of that array as
> "spleen" or "not spleen". That task is called **segmentation**.

---

## 1.1 The spleen in two minutes

**Where it is.** In the upper left of the abdomen, behind the stomach, under
the diaphragm, protected by the lower left ribs. Its neighbours are the
stomach, the left kidney, the tail of the pancreas, the diaphragm and a bend
of the large intestine called the **splenic flexure**. This crowding matters
for surgery: operations on any of those neighbours happen right next to it.

**What it looks like.** In an adult it is roughly 11–12 cm long and weighs
on the order of 150 g, with wide variation between people. Its shape is a
curved wedge, like a coffee bean pressed against the inside of the ribs.

**What it does.**
- **Filters blood.** It removes old or damaged red blood cells.
- **Immune function.** It holds lymphocytes and helps fight blood-borne
  infections, especially bacteria with a sugar capsule.
- **Reservoir.** It holds some blood and platelets.

**Why it is delicate.** It has a very rich blood supply (the splenic artery
branches off one of the main arteries leaving the aorta) and a thin capsule.
A small tear can bleed a lot. Sometimes the only fix is to remove the organ
(**splenectomy**), which leaves the patient with a lifelong higher risk of
serious infections and the need for vaccinations.

**Why doctors measure it.** An enlarged spleen (**splenomegaly**) is a sign
of several conditions: liver disease with high pressure in the portal vein,
blood cancers such as lymphoma and leukaemia, and some infections. Spleen
volume measured on CT is a common quantitative marker. Measuring it by hand
means outlining the organ on dozens of slices, which is slow. Automating that
is one direct use of a segmentation model like ours.

## 1.2 Why surgical robotics cares

In robot-assisted and image-guided surgery, the surgeon (and the robot) only
see what the camera sees: the surface of tissue, through a small field of
view. A pre-operative CT shows everything underneath. Segmentation turns that
CT into a **3D model of named structures**, which can be used for:

| Use | What segmentation provides |
|---|---|
| **Planning** | Patient-specific 3D model: where to put the ports, which path the instruments take, how far to stay from at-risk organs |
| **Registration and navigation** | Matching the CT model to the live camera view, so hidden structures can be overlaid ("augmented reality") |
| **Safety constraints** | "Virtual fixtures": software boundaries the robot arm is not allowed to cross near delicate organs |
| **Measurement** | Organ volumes, distances, change over time |

**A reality check.** The body during surgery is not the body in the scanner.
The abdomen is inflated with CO₂ for keyhole surgery, the patient lies
differently and breathes, and organs shift and deform. A pre-operative
segmentation is a starting point that registration then has to warp. The
model's accuracy at the **boundary** of the organ matters a lot for this use,
which is why chapter 9 discusses metrics beyond plain overlap.

## 1.3 How a CT scanner works (the EE view)

You can think of CT as **a signal-processing problem that the scanner
solves before we ever see the data.**

1. **Measure line integrals.** An X-ray tube and a curved detector array
   rotate around the patient. Each detector element measures the intensity
   $I$ of a thin beam that crossed the body. By the **Beer–Lambert law**, each
   bit of tissue attenuates the beam exponentially:

   ```math
   I = I_0 \exp\!\left(-\int_\text{ray} \mu(x, y, z)\, d\ell\right)
   \quad\Longrightarrow\quad
   -\ln\frac{I}{I_0} = \int_\text{ray} \mu\, d\ell
   ```

   Here $\mu$ is the **linear attenuation coefficient**, which is what we
   want to image. Taking the log turns each measurement into a line integral
   of $\mu$.

2. **The set of all line integrals is the Radon transform** of $\mu$. Collect
   them over all angles and you have a **sinogram**.

3. **Invert it.** The **Fourier slice theorem** says that the 1D Fourier
   transform of one projection equals a line through the 2D Fourier
   transform of the slice. Inverting this directly gives **filtered
   back-projection**: filter each projection with a ramp filter $\lvert\omega\rvert$
   (a high-pass filter that undoes the blurring of back-projection), then
   smear it back across the image. Modern scanners use iterative
   reconstruction, which models noise better, but the principle is the same.

4. **Helical scanning** moves the table continuously while the gantry spins,
   producing a stack of 2D slices. Together they form a 3D volume.

Typical numbers: each slice is 512 × 512 pixels, pixels are about 0.6–0.9 mm
wide, and slices are 0.5–5 mm apart. A single abdominal scan is a few hundred
slices, about 100–200 million voxels' worth of 16-bit integers.

> If you want the full maths with an EE flavour, *Principles of
> Computerized Tomographic Imaging* by Kak & Slaney is free online (see
> [chapter 13](13-where-to-go-next.md)).

## 1.4 Hounsfield units: the numbers inside the array

Raw $\mu$ values depend on the X-ray energy, so CT rescales them relative to
water and air:

```math
\mathrm{HU} = 1000 \times \frac{\mu - \mu_\text{water}}{\mu_\text{water} - \mu_\text{air}}
```

By construction, water = 0 HU and air = −1000 HU. Everything else falls on
this scale:

| Material | Approximate HU |
|---|---|
| Air (lungs, bowel gas, outside the body) | −1000 |
| Fat | −100 to −50 |
| Water, simple fluid | 0 |
| Soft tissue, unenhanced spleen, blood | +30 to +60 |
| Organs after IV contrast | roughly +100 to +200 |
| Bone | +300 to +1000 and above |
| Metal implants | very high, and they cause streak artefacts |

So a CT voxel is not a "colour". It is a **calibrated physical measurement**.
That makes CT unusually friendly for machine learning: the same tissue gives
roughly the same number on every scanner, which is not true of MRI or of
camera images.

### Contrast phases

For abdominal CT, patients usually get an intravenous **iodine contrast
agent**, which makes blood (and blood-rich organs) much brighter. When the
scan happens after injection matters:

- **Arterial phase** (about 25–35 s): the spleen often looks mottled or
  striped, because different parts fill at different rates. That pattern is
  normal, but a model can mistake it for disease or for a boundary.
- **Portal-venous phase** (about 60–70 s): the spleen looks uniformly
  bright. **Our dataset is portal-venous**, which makes the task easier and
  is also a limitation: a model trained here may struggle on arterial-phase
  scans.

## 1.5 How radiologists look at the data

A 3D volume is viewed as slices in three standard planes:

| Plane | Cuts the body into | Viewer looks from |
|---|---|---|
| **Axial** (transverse) | top and bottom pieces | the feet, by convention |
| **Coronal** | front and back pieces | the front |
| **Sagittal** | left and right pieces | the side |

Two conventions trip up engineers:

1. **Patient left is screen right.** In radiology viewers you look at the
   patient as if they were facing you, so their left appears on your right.
   That's why the spleen (patient left) appears on the right of an axial
   image.
2. **Display windowing.** A screen shows ~256 grey levels, but CT spans
   ~4000 HU. Viewers map a chosen **window** (centre/level and width) to
   black→white. A typical soft-tissue window is level 40, width 400, i.e.
   −160 to +240 HU. Our preprocessing uses a similar idea: it clips to −57
   to +164 HU (chapter 2).

## 1.6 The files: DICOM and NIfTI

- **DICOM** is what scanners produce: one file per slice, full of metadata
  (patient, scanner, timing, geometry). Hospitals use it.
- **NIfTI** (`.nii.gz`) is the research format: the whole 3D volume in one
  file, plus a 4×4 **affine matrix** in the header.

The affine maps array indices to millimetres in the scanner's coordinate
system:

```math
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}_\text{mm}
= A \begin{bmatrix} i \\ j \\ k \\ 1 \end{bmatrix}_\text{voxel},
\qquad
A = \begin{bmatrix} R \cdot \mathrm{diag}(s) & t \\ 0 & 1 \end{bmatrix}
```

Here $s$ is the voxel **spacing** in mm, $R$ is a rotation/flip that says
which way each array axis points in the body, and $t$ is the position of
voxel (0, 0, 0). Two files with identical arrays but different affines
describe **different anatomy layouts**. Ignoring the affine is the classic
bug in medical imaging code, comparable to ignoring timezone or endianness.

Axis direction conventions have names made of three letters, one per axis:
**RAS** means +x points to the patient's **R**ight, +y to **A**nterior (front),
+z to **S**uperior (head). DICOM uses LPS (Left, Posterior, Superior). Our
pipeline converts everything to RAS (chapter 2).

## 1.7 Segmentation: the task itself

**Segmentation = classify every voxel.** The output is an array the same
size as the input, with 0 (background) or 1 (spleen) in each cell. The
ground-truth arrays in our dataset were drawn by people, slice by slice.

Three facts about these labels shape everything else:

1. **Extreme class imbalance.** The spleen is a tiny fraction of an abdominal
   CT. Even after cropping away the air around the body, it is typically
   well under 1 % of voxels. A model that always says "background" would be
   99 % accurate and completely useless. Chapter 8 explains how the loss
   function and sampling deal with this.
2. **Labels are imperfect at the edges.** Voxels on the boundary contain a
   mixture of spleen and fat or neighbouring organs (the **partial volume
   effect**, chapter 2). Two experts outlining the same spleen disagree by a
   voxel or two along the surface. That sets a ceiling on how "accurate" a
   model can meaningfully be.
3. **Labelling is expensive.** Outlining one organ carefully takes an expert
   tens of minutes. That's why public datasets are small. **Ours has 41
   labelled scans**, which is tiny by machine-learning standards and shapes
   many design decisions.

## 1.8 Our dataset in one table

The **Medical Segmentation Decathlon** (MSD) is a public benchmark of ten
3D segmentation tasks. We use **Task09 Spleen**:

| | |
|---|---|
| Scans | 41 labelled (plus 20 without public labels, unused) |
| Patients | Adults having chemotherapy for cancer that spread to the liver, from one hospital (Memorial Sloan Kettering) |
| Phase | Portal-venous contrast CT |
| Labels | 0 = background, 1 = spleen |
| Format | NIfTI |
| Size | ~1.5 GB download |

Because all scans come from one hospital and one patient group, a model
trained here says nothing about how it would do on other scanners, other
hospitals or other diseases. The README lists this as a limitation, and it is
the first question any clinician will ask.

---

**In the code**
- Downloading and listing the scans: [`src/data.py`](../../src/data.py),
  functions `download_spleen` and `load_datalist`.
- Splitting 41 scans into 33 for training and 8 for validation:
  `split_datalist` (chapter 3 explains why we split).
- Fake "scans" for tests (a bright ellipsoid inside a body-shaped
  ellipsoid): `make_synthetic_dataset`.

**Check yourself**
1. Why does the log of the detector reading give you a line integral of $\mu$?
2. A voxel reads −1000 HU. What is probably there? What about +130 HU in a
   portal-venous scan?
3. Two NIfTI files hold identical arrays. Can they show different anatomy?
4. Why would "99 % voxel accuracy" be a meaningless score for this task?

Next: [2. Images as signals →](02-images-as-signals.md)
