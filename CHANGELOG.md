# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project has no
tagged releases yet, so entries are grouped under **Unreleased** until the
first tag.

## [Unreleased]

### Resumable training

Long training runs no longer have to finish in one sitting. If a Colab
session disconnects, the laptop goes to sleep or the process is killed,
training continues from the last finished epoch instead of epoch 1.

#### Highlights

- **`python -m src.train --resume`** continues an interrupted run from
  `<checkpoint_dir>/last.pt`. `--resume PATH` resumes from a specific file.
- **`last.pt` is now written after every epoch** (previously only once, at the
  very end). It holds the complete training state: weights, AdamW moments,
  cosine-schedule position, AMP loss scale, Python/NumPy/torch/CUDA random
  generators, the history so far and the best Dice. An interruption loses at
  most the epoch in progress.
- **Colab notebook:** section 6 has a new `RESUME` switch. After a disconnect,
  re-run sections 1–5, set `RESUME = True`, keep `EPOCHS` and `EXTRA_FLAGS`
  unchanged, and run the cell. Requires `USE_DRIVE = True` so `last.pt`
  survives on Google Drive. A resumed run logs to
  `results/logs/train_resume.log`, keeping the original `train.log`.

#### Safety checks

- **Atomic save.** `last.pt` is written to `last.pt.tmp` and renamed into
  place, so a kill during the save never leaves a truncated checkpoint.
- **Config must match.** Resuming compares the saved config with the current
  one over `seed`, `data`, `model` and `train`, and stops with a list of every
  differing key. This prevents, for example, resuming a 20-epoch run with
  `--epochs 150`, which would stretch the learning-rate schedule mid-run.
  Settings that do not change the computation may differ: `data.root_dir`,
  `data.num_workers`, `data.cache_rate`, `model.use_checkpoint`,
  `train.checkpoint_dir`.
- **Fails fast.** The resume file and config are checked *before* the
  dataset is cached, so a wrong path or flag fails in seconds rather than
  after several minutes of loading.
- **`best.pt` is protected.** The best Dice so far is restored, so a resumed
  run only replaces `best.pt` with a genuine improvement.

#### Provenance

- `results/training_history.json` covers the whole run across sessions and
  gains a `resumed_from_epochs` field (e.g. `[11]`), so a reader can see that
  a run was resumed. `train_minutes` accumulates over all sessions.

#### Compatibility and upgrade notes

- **No change for uninterrupted runs.** Without `--resume`, training behaves
  and computes exactly as before; the only difference is that `last.pt` is
  saved every epoch.
- **`last.pt` is larger.** It now also contains the optimiser state (about
  twice the weight size for AdamW), so roughly 3× the size of `best.pt`.
  For the default 62 M-parameter model that is on the order of 0.75 GB on
  your Drive, rewritten every epoch. If Drive keeps the replaced copies
  in Trash or as revisions, they can fill your quota on a long run; check
  Drive's storage after the first few epochs.
- **Old `last.pt` files cannot be resumed.** Files written before this change
  hold weights only; `--resume` refuses them with an explanation. Their
  `best.pt` remains fully usable for export, parity and benchmarking.
  A run already in progress on the previous code cannot be resumed after a
  disconnect: its `best.pt` on Drive is still usable, or start a new run with
  the updated notebook.
- **`last.pt` still works with `load_checkpoint`.** It keeps the
  `model_state` and `config` keys, so tools that read checkpoints are
  unaffected.
- **Not bit-identical.** Epochs completed before the interruption are
  unchanged. After resuming, MONAI's per-transform augmentation generators
  restart, so later crops are different random draws from the same
  distribution. Expect differences of the same order as GPU
  non-determinism.
- **Extending a run is not supported.** Changing `--epochs` on resume is
  refused because it would alter the cosine schedule. Start a new run
  instead.

#### Tests

- New `tests/test_train_resume.py` (4 tests, about 1 minute on a laptop CPU)
  simulates a disconnect after epoch 2 by injecting a fault into the save
  step, resumes, and checks that the pre-interruption losses match an
  uninterrupted run, that the epoch list is complete, that the LR schedule
  continued, and that a changed `train.epochs`, a missing `last.pt` and the
  allowed config differences are each handled correctly.

#### Documentation

- [Training reference](docs/reference/training.md): new §10 *Interruption and
  resume* (state contents, atomic write, compatibility rule, equivalence
  caveat) and new failure-mode rows.
- [Colab guide](docs/reference/colab.md): new *Resuming after a disconnect*
  walkthrough and troubleshooting rows.
- Learning track: new [§8.13 *Surviving interruptions*](docs/learn/08-training-this-model.md#813-surviving-interruptions-checkpoints-and-resuming)
  explaining why weights alone are not enough to resume, plus a new
  check-yourself question; [chapter 13](docs/learn/13-software-engineering-of-this-repo.md)
  covers designing for interruption and fault-injection testing; new glossary
  entries *Resume* and *Atomic write*.
- README: resume mentioned under *Reproduce*; test-suite time updated.
