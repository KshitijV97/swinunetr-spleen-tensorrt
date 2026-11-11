# Training the full 150 epochs for free

The published 20-epoch run reached a validation Dice of only 0.587 and was
still improving (see the README's *Why the Dice is low*). This page is the plan
for a full 150-epoch run **without paying for compute**, and for finishing the
benchmark afterwards.

> **Status: planned, not yet run.** The notebook
> [`notebooks/kaggle_train_150.ipynb`](../../notebooks/kaggle_train_150.ipynb)
> has been written and its JSON validated, but it has **not** been executed on
> Kaggle. The timings below are extrapolations from the 20-epoch Colab run, and
> the platform limits are from memory, so check them on the day. Expect to fix
> small environment issues on the first attempt. Nothing here predicts what
> Dice 150 epochs will reach.

---

## 1. What has to be true

| Fact | Source |
|---|---|
| 20 epochs took 42.6 min on a T4 with gradient checkpointing, i.e. about 2.1 min/epoch including validation | `results/training_history.json → train_minutes` |
| So 150 epochs is about **5.3 hours** (plan for 5 to 6) | extrapolation, not measured |
| `last.pt` (full training state) is written atomically after every epoch, so a stop loses at most one epoch | `src/train.py` |
| `--resume` needs the same `--epochs` and flags, except `--use-checkpoint`, `num_workers`, `cache_rate` and paths | `RESUME_ALLOWED_DIFFERENCES` in `src/train.py` |
| The 20-epoch run cannot be extended: `--resume` refuses a different epoch count, and its cosine schedule already decayed to zero | `src/train.py` |

## 2. Where to run it for free

| Option | Runs in background? | Limits (verify before relying on them) | Verdict |
|---|---|---|---|
| **Kaggle Notebooks** (T4 x2) | **Yes.** *Save & Run All (Commit)* keeps running with the browser closed | About 30 GPU-hours/week, 12 h per session. Needs a phone-verified account for Internet access | **Recommended** |
| Colab free (T4) | No. Needs an open tab, and idles out or disconnects | Roughly 12 h at best, often less; GPU availability varies | Fallback, with Drive + `--resume` |
| GitHub Actions | Yes | Hosted runners have **no GPU** | Not usable for training |
| Your Mac (MPS) | Yes | Very slow for a 62 M-parameter 3D transformer; also no TensorRT | Not realistic for 150 epochs |

Why Kaggle: 150 epochs fits in one 12-hour session, the run survives a closed
laptop, and the output files (checkpoints) are stored with the saved version.
The weekly quota is more than one full run needs, so there is room for one
retry.

## 3. Steps (Kaggle)

1. **Push the branch** so Kaggle can clone it:
   `git push -u origin train/150-epoch-free-run`. (The repository must be
   public for the unauthenticated clone to work.)
2. On kaggle.com: create an account and verify your phone number (needed for
   Internet access in notebooks).
3. **Either push it from the command line** (recommended, starts the
   background run directly):
   `pip install kaggle`, create a token at kaggle.com → Settings → API →
   *Create New Token*, save it as `~/.kaggle/kaggle.json` (`chmod 600`), then
   run `scripts/kaggle_push.sh`. It uploads the notebook as a private kernel
   with GPU T4 and Internet on and starts *Save & Run All*. Skip steps 4–5.
   Track it with `kaggle kernels status <user>/swinunetr-spleen-150-epochs`
   and download with `kaggle kernels output <user>/swinunetr-spleen-150-epochs -p out/`.

   **Or in the browser:** **New Notebook → File → Import Notebook**, upload
   `notebooks/kaggle_train_150.ipynb`.
4. In the session options set **Accelerator = GPU T4 x2** and **Internet = On**.
5. Click **Save Version → Save & Run All (Commit)**. Close the tab. Check
   progress from the notebook's *Versions* list, which has live logs.
6. When it completes, open the version's **Output** tab and download
   `checkpoints/best.pt`, `results/training_history.json` and
   `results/training_curve.png`.

**If the session ends before epoch 150** (time budget reached or a crash): open
a new session of the same notebook, **Add Input → Notebook Output Files**, pick
the previous version, and Save & Run All again. Section 4 of the notebook
copies `last.pt` in and training continues. Do not change `EPOCHS`.

**Fallback on Colab free:** use `notebooks/colab_train_and_benchmark.ipynb` with
`USE_DRIVE = True`, `EPOCHS = 150`, `EXTRA_FLAGS = '--use-checkpoint'`. After
each disconnect set `RESUME = True` and run again. This needs you to babysit it
and may take several sessions.

## 4. Choices and trade-offs

- **Training only on Kaggle, benchmark on Colab.** TensorRT's wheel must match
  the CUDA version of the preinstalled PyTorch, which is the part most likely to
  break on a new platform. Keeping it off Kaggle keeps the long run simple, and
  the benchmark takes about 30 minutes on Colab with the downloaded `best.pt`.
- **Same flags as the 20-epoch run** (`--use-checkpoint`), so the memory
  behaviour is already known to fit. Dropping it is about 30% faster but may
  run out of GPU memory; it is also allowed to change between resumed
  sessions.
- **`timeout` at 11 hours.** Lets training stop cleanly before Kaggle's hard
  kill instead of losing the session.
- **Dataset in `/kaggle/temp`.** Keeps the 1.5 GB download out of the saved
  output.
- **Benchmark numbers stay comparable.** Same T4 family and the same
  protocol. (Kaggle has retired the P100, and current PyTorch builds no
  longer support its compute capability 6.0; the notebook stops early on it.)
- **Minimal installs on Kaggle.** Only MONAI, nibabel and einops are
  installed (same pins as `requirements-colab.txt`). The Colab file also pins
  ONNX, ONNX Runtime, pytest and requests, which training does not need and
  which would replace Kaggle's preinstalled copies.
- **Failures still save output.** A failed training step is caught so the
  summary and cleanup cells run, then the last cell re-raises, so the version
  shows as failed but `last.pt` and the logs are kept.

## 5. After training: export and benchmark on Colab

1. Open `notebooks/colab_train_and_benchmark.ipynb`, run sections 1 to 5
   (GPU, Drive, clone/install, tests, dataset).
2. Put the downloaded `best.pt` into `checkpoints/` (on Drive if mounted).
3. **Skip section 6 (train)**, then run sections 7 (export), 8 (benchmark) and
   9 to 10 (curve and numbers). Copy `training_history.json` and
   `training_curve.png` from Kaggle into `results/` first, so section 9 and
   the README use the 150-epoch history.
4. Update the README tables (epochs, training time, Dice, latency) and remove
   the "under-trained" warnings only if the new Dice supports it.
5. Optional: publish `best.pt` as a GitHub Release rather than committing it
   (see the checkpoint-release notes on the `docs/release-checkpoints` branch).

## 6. Risks, honestly

- **Untested notebook.** First run on a new platform. Likely failure points:
  package pins (`requirements-colab.txt` was pinned for Colab's torch), the
  repo not being public, Internet not enabled.
- **Quota and availability.** Kaggle's weekly GPU hours and queue times change.
  A queued GPU session can wait before it starts.
- **Time estimate is an extrapolation.** Kaggle's CPU count and disk may make
  data loading slower than Colab. Check the epoch time printed in the first
  few minutes of the log; if it is above about 3 min/epoch, 150 epochs will
  not fit in one session and you should plan on resuming.
- **Dice is not guaranteed.** Training from scratch on 33 volumes may plateau
  below 0.9. If so, report the real number and discuss it (small data, no
  pretrained weights) rather than tuning until it looks good.
- **Free means no guarantees.** Never leave the only copy of a result on the
  VM: download `best.pt` and `last.pt` from the version's Output tab as soon as
  the run finishes.
