"""Shared pytest fixtures.

Tests never touch the real dataset: they write a couple of tiny synthetic
volumes into a temporary folder, so `pytest` runs in seconds on any laptop.
"""

import pytest

from src.data import make_synthetic_dataset, load_datalist
from src.utils import load_config


@pytest.fixture
def smoke_cfg():
    """The smoke-test configuration (tiny model, 64^3 crops)."""
    return load_config(smoke_test=True)


@pytest.fixture
def synthetic_items(tmp_path):
    """Two synthetic volumes in MSD layout; returns the [{image, label}] list."""
    task_dir = make_synthetic_dataset(tmp_path, n_volumes=2, shape=(80, 80, 64))
    return load_datalist(task_dir)
