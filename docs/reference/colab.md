# Running the full pipeline on Google Colab (T4 GPU)

The notebook [`notebooks/colab_train_and_benchmark.ipynb`](../../notebooks/colab_train_and_benchmark.ipynb)
does everything: install, test, download, train, export, benchmark and save
results. This page walks through it and lists what can go wrong.

---

## 1. Open the notebook

**Option A: direct link (easiest).** Open this URL in a browser while logged
into your Google account:

```
https://colab.research.google.com/github/kshitijv97/swinunetr-spleen-tensorrt/blob/main/notebooks/colab_train_and_benchmark.ipynb
```

(The README's "Open in Colab" badge points to the same URL.)

**Option B: from inside Colab.** Go to <https://colab.research.google.com>,
then **File → Open notebook → GitHub**. Type `kshitijv97/swinunetr-spleen-tensorrt`,
press Enter, and pick `notebooks/colab_train_and_benchmark.ipynb`.

## 2. Select the GPU

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
The first cell (`!nvidia-smi`) must show "Tesla T4". If it errors, there is
no GPU attached.

Free-tier GPU availability varies. If Colab says no GPU is available, try
again later.

## 3. Run the cells

Use **Runtime → Run all**, or run the cells one by one to watch each step.

| Section | What happens | Rough duration |
|---|---|---|
| 1. GPU check | `nvidia-smi` | seconds |
| 2. Google Drive (optional) | Mount Drive so checkpoints survive a disconnect. Click through Google's permission prompt | seconds |
| 3. Clone + install | `git clone`, `pip install -r requirements-colab.txt`, then `tensorrt-cu12` or `tensorrt-cu13` (10.16.1.11) matching Colab's PyTorch CUDA version | a few minutes |
| 4. Tests | `pytest` | under a minute |
| 5. Download | MSD Spleen, ~1.5 GB, MD5-verified | a few minutes |
| 6. Train | 150 epochs by default. The log prints elapsed seconds after every epoch | **the long step**; estimate from the first few epochs |
| 7. Export | ONNX export + verification | a few minutes |
| 8. Benchmark | Builds the TensorRT engine (first time), then times 3 backends | several minutes |
| 9. Plot | Saves `results/training_curve.png` | seconds |
| 10. Numbers | Prints the table to paste into the README | seconds |
| 11. Download | Downloads `results.zip` to your computer | seconds |

The durations are rough guides, not measurements. The epoch timer is the
source of truth.

### If your session is short

Free Colab sessions can disconnect when idle and have a maximum lifetime.
Two safeguards:

1. Keep `USE_DRIVE = True` in section 2. Checkpoints and results then live in
   `MyDrive/swinunetr-spleen-tensorrt/` and survive a disconnect.
2. Lower `EPOCHS` in section 6. If you do, state the epoch count next to your
   results in the README. A 60-epoch result is still a real result, as long
   as it is labelled.

If a disconnect happens after training finished, re-run sections 1–5 (to
reinstall and re-download), skip section 6, and continue from section 7:
`checkpoints/best.pt` is still on your Drive.

## 4. Get the results back into the repository

1. Section 11 downloads `results.zip` with `benchmark.json`, `benchmark.md`,
   `onnx_export.json`, `training_history.json` and `training_curve.png`.
   These are small text/image files with no data and no weights.
2. Unzip it into `results/` in your local clone.
3. Paste the table from `results/benchmark.md` into the README's **Results**
   section and fill in the other placeholders (see
   [author-notes/reporting-results.md](../author-notes/reporting-results.md)).
4. Commit and push:

   ```bash
   git add results/ README.md
   git commit -m "Add T4 benchmark results"
   git push
   ```

## 4b. Publish `best.pt` as a GitHub Release (optional)

Checkpoints are never committed (`.gitignore` excludes `*.pt`, and the repo
does not use Git LFS). If you want others to download the trained weights
without retraining, attach `best.pt` to a **GitHub Release**. A release is
tied to a tag, and the tag is cut on `main`, so the weights are linked to the
exact code that produced them.

Before you start:

- Merge the code that produced the checkpoint into `main`. The tag points at
  `main` on GitHub, not at your Colab working copy.
- Use one tag per training run (`v1`, `v1-run2`, ...). Don't overwrite the
  weights of an old release.
- Upload `best.pt` only. `last.pt` is for resuming training, is larger
  (it holds the AdamW state), and nobody else needs it.

### Option A: from your laptop (safest, recommended)

No credentials ever enter Colab. Download `best.pt` from Drive
(`MyDrive/swinunetr-spleen-tensorrt/checkpoints/best.pt`) into your local
clone's `checkpoints/`, then:

```bash
gh release create v1 checkpoints/best.pt \
  --repo KshitijV97/swinunetr-spleen-tensorrt \
  --target main --title "v1" \
  --notes "SwinUNETR spleen checkpoint (<epochs> epochs, val Dice <x>)"
```

To add the file to a release that already exists, use
`gh release upload v1 checkpoints/best.pt --clobber` instead.

### Option B: from the Colab notebook (last cell, after training)

Convenient, but it puts a write-capable token inside Colab.

1. Create a **fine-grained** GitHub token limited to this one repository, with
   **Contents: Read and write** and a short expiry.
2. Add it in Colab's 🔑 **Secrets** panel as `GH_TOKEN` and enable notebook
   access. Never paste it into a cell.
3. Run a cell like this:

   ```python
   import os
   from google.colab import userdata

   os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")
   REPO, TAG = "KshitijV97/swinunetr-spleen-tensorrt", "v1"   # new TAG per run

   !(type -p gh >/dev/null) || apt-get -qq install -y gh
   !gh release view {TAG} -R {REPO} >/dev/null 2>&1 || \
     gh release create {TAG} -R {REPO} --target main --title "{TAG}" \
       --notes "SwinUNETR spleen checkpoint"
   !gh release upload {TAG} -R {REPO} --clobber checkpoints/best.pt
   ```

   `--target main` only matters when the tag does not exist yet. `--clobber`
   makes re-running the cell safe.

Colab runs untrusted notebook code and shared sessions can expose secrets, so
revoke the token once you have uploaded.

### Downloading it later

```bash
gh release download v1 -R KshitijV97/swinunetr-spleen-tensorrt \
  -p best.pt -D checkpoints/
```

Load it only if you trust the source: `.pt` files are pickles and can run
code when loaded (see the security note on `load_checkpoint()`).

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi: command not found` | The runtime has no GPU. Change the runtime type to T4 |
| `CUDA out of memory` during training | Training stops with a message. In section 6 set `EXTRA_FLAGS = '--use-checkpoint'` and re-run that cell. If it still does not fit, add `--feature-size 24` (and say so in the README) |
| Session crashes with "used all available RAM" while loading data | Edit `data.cache_rate` to `0.5`, or add `--max-volumes 30` |
| `pip` dependency-resolver warnings about Colab's preinstalled packages | Usually harmless. Check the `import torch, monai, tensorrt` cell prints versions |
| `ModuleNotFoundError: tensorrt` | Re-run the install cell in section 3. It prints which `tensorrt-cu…` wheel it chose; that must match `torch.version.cuda` |
| A step fails | The notebook stops at that step and shows the error. The full output is in `results/logs/<step>.log` (on Drive if mounted), so it survives a disconnect |
| TensorRT engine build fails | Read the error printed by the parser. See [deployment.md](deployment.md#7-troubleshooting) |
| `git clone` fails | The repository must be public, or you need to authenticate inside Colab |
