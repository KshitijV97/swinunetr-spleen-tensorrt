"""Tests for model construction, output shapes and the TensorRT import guard."""

import importlib

import pytest
import torch

from src.evaluation import predict_volume
from src.model import build_model, count_parameters, validate_roi


@pytest.mark.parametrize("roi", [(64, 64, 64), (96, 64, 64)])
def test_output_shape_matches_input(smoke_cfg, roi):
    model = build_model(smoke_cfg["model"]).eval()
    x = torch.rand(2, 1, *roi)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, smoke_cfg["model"]["out_channels"], *roi)


def test_sliding_window_handles_arbitrary_volume_size(smoke_cfg):
    model = build_model(smoke_cfg["model"]).eval()
    volume = torch.rand(1, 1, 70, 90, 50)  # not a multiple of 32, smaller than roi in z
    with torch.no_grad():
        y = predict_volume(model, volume, (64, 64, 64), sw_batch_size=2, overlap=0.25)
    assert y.shape == (1, 2, 70, 90, 50)


def test_invalid_sizes_are_rejected(smoke_cfg):
    with pytest.raises(ValueError):
        validate_roi((96, 96, 80))
    with pytest.raises(ValueError):
        build_model({**smoke_cfg["model"], "feature_size": 16})


def test_default_model_size(smoke_cfg):
    # feature_size=48 is the paper configuration (~62M parameters).
    n = count_parameters(build_model({**smoke_cfg["model"], "feature_size": 48}))
    assert 55e6 < n < 70e6


def test_tensorrt_module_imports_without_cuda():
    """src.trt_utils must be importable on machines without CUDA/TensorRT."""
    trt_utils = importlib.import_module("src.trt_utils")
    if not torch.cuda.is_available():
        assert trt_utils.tensorrt_available() is False
        with pytest.raises(RuntimeError):
            trt_utils.TRTPredictor("missing.engine")
