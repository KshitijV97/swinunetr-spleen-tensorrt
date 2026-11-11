"""--resume: an interrupted run continues where it stopped.

Runs a deliberately tiny training (64^3 crops, feature_size 12, two small
synthetic volumes) on the CPU, so the whole file takes about a minute.
"""

import pytest
import torch

import src.train as train_mod
from src.data import SYNTHETIC_TASK, make_synthetic_dataset


class Interrupted(Exception):
    """Stands in for a Colab disconnect."""


@pytest.fixture
def tiny_cfg(smoke_cfg, tmp_path):
    make_synthetic_dataset(tmp_path / "data", n_volumes=2, shape=(72, 72, 64))
    cfg = smoke_cfg
    cfg["data"].update(task=SYNTHETIC_TASK, root_dir=str(tmp_path / "data"),
                       roi_size=[64, 64, 64], num_samples=1, num_workers=0)
    cfg["train"].update(epochs=3, val_interval=1, checkpoint_dir=str(tmp_path / "ckpt"))
    cfg["inference"]["sw_batch_size"] = 1
    cfg["results_dir"] = str(tmp_path / "results")
    return cfg


def interrupt_after(monkeypatch, stop_epoch):
    """Make training die right after last.pt for ``stop_epoch`` is written."""
    real_save = train_mod.atomic_save

    def save_then_maybe_die(obj, path):
        real_save(obj, path)
        if obj["epoch"] == stop_epoch:
            raise Interrupted

    monkeypatch.setattr(train_mod, "atomic_save", save_then_maybe_die)


def test_resume_continues_from_the_saved_epoch(tiny_cfg, monkeypatch):
    cpu = torch.device("cpu")
    reference = train_mod.train(tiny_cfg, cpu)  # uninterrupted run, for comparison

    interrupt_after(monkeypatch, stop_epoch=2)
    with pytest.raises(Interrupted):
        train_mod.train(tiny_cfg, cpu)
    last = torch.load(tiny_cfg["train"]["checkpoint_dir"] + "/last.pt", weights_only=False)
    assert last["epoch"] == 2
    monkeypatch.undo()

    resumed = train_mod.train(tiny_cfg, cpu, resume="auto")
    assert resumed["epoch"] == [1, 2, 3]
    assert resumed["val_epoch"] == [1, 2, 3]
    assert resumed["resumed_from_epochs"] == [2]
    # Epochs before the interruption are the same run, so identical numbers.
    assert resumed["train_loss"][:2] == pytest.approx(reference["train_loss"][:2])
    assert resumed["best_val_dice"] == max(resumed["val_dice"])
    final = torch.load(tiny_cfg["train"]["checkpoint_dir"] + "/last.pt", weights_only=False)
    assert final["epoch"] == 3
    # The LR schedule continued instead of restarting: 3 cosine steps in total.
    assert final["scheduler_state"]["last_epoch"] == 3


def test_resume_refuses_a_different_config(tiny_cfg, monkeypatch):
    interrupt_after(monkeypatch, stop_epoch=1)
    with pytest.raises(Interrupted):
        train_mod.train(tiny_cfg, torch.device("cpu"))
    monkeypatch.undo()

    tiny_cfg["train"]["epochs"] = 5
    with pytest.raises(SystemExit, match=r"train\.epochs: 3 -> 5"):
        train_mod.train(tiny_cfg, torch.device("cpu"), resume="auto")


def test_resume_without_checkpoint_explains_itself(tiny_cfg):
    with pytest.raises(SystemExit, match="no checkpoint"):
        train_mod.train(tiny_cfg, torch.device("cpu"), resume="auto")


def test_paths_and_memory_settings_may_change_on_resume(smoke_cfg):
    import copy

    changed = copy.deepcopy(smoke_cfg)
    changed["data"]["root_dir"] = "/elsewhere"
    changed["data"]["num_workers"] = 8
    changed["model"]["use_checkpoint"] = True
    changed["train"]["checkpoint_dir"] = "/drive/ckpt"
    assert train_mod.config_differences(smoke_cfg, changed) == []

    changed["train"]["lr"] = 1.0
    assert train_mod.config_differences(smoke_cfg, changed) == [
        f"train.lr: {smoke_cfg['train']['lr']!r} -> 1.0"
    ]
