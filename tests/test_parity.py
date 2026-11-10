"""Tests for the numerical parity gate (src/parity.py).

Three kinds of test:
* metric maths on hand-made arrays (instant);
* the gate's decisions: identical outputs pass, precision noise passes the
  matching tier, and a genuinely broken model fails;
* a real ONNX export of a tiny model, checked through ``ORTPredictor``.
Everything runs on CPU in well under a minute.
"""

import copy

import numpy as np
import pytest
import torch

from src.export_onnx import export
from src.model import build_model
from src.parity import (
    ORTPredictor,
    compare,
    fp32_reference,
    golden_inputs,
    judge,
    mask_dice,
    run_golden,
    top2_margin,
)

ROI = (64, 64, 64)  # the smallest size SwinUNETR's deepest layer accepts


# -----------------------------------------------------------------------------
# Metric maths
# -----------------------------------------------------------------------------
def test_top2_margin_three_classes():
    logits = np.array([[[0.0], [3.0], [1.0]]])  # (N=1, C=3, voxels=1)
    assert top2_margin(logits).item() == pytest.approx(2.0)


def test_mask_dice_both_empty_counts_as_agreement():
    empty = np.zeros((4, 4), dtype=bool)
    assert mask_dice(empty, empty) == 1.0
    full = np.ones((4, 4), dtype=bool)
    assert mask_dice(full, empty) == 0.0


def test_compare_identical_is_perfect():
    ref = np.random.default_rng(0).normal(size=(2, 2, 8, 8, 8)).astype(np.float32)
    m = compare(ref, ref.copy(), atol=1e-3, rtol=1e-3, margin_threshold=1.0)
    assert m["max_abs_diff"] == 0
    assert m["fraction_within_tolerance"] == 1.0
    assert m["cosine_similarity"] == pytest.approx(1.0)
    assert m["flip_rate"] == 0 and m["confident_flips"] == 0


def test_compare_separates_unsure_from_confident_flips():
    # Three voxels, two classes. Voxel 0: margin 0.01 (unsure). Voxel 1: margin
    # 5 (confident). Voxel 2: margin 5, untouched.
    ref = np.array([[[0.0, 0.0, 0.0], [0.01, 5.0, 5.0]]])
    out = ref.copy()
    out[0, 1, 0] = -0.01  # unsure voxel flips: expected precision noise
    out[0, 1, 1] = -1.0   # confident voxel flips: a real defect
    m = compare(ref, out, atol=1e-3, rtol=1e-3, margin_threshold=1.0)
    assert m["flip_rate"] == pytest.approx(2 / 3)
    assert m["confident_voxels"] == 2
    assert m["confident_flips"] == 1
    lowest = m["flips_by_margin"][0]
    assert lowest["voxels"] == 1 and lowest["flips"] == 1


def test_compare_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare(np.zeros((1, 2, 4)), np.zeros((1, 2, 5)), 1e-3, 1e-3, 1.0)


def test_non_finite_output_fails_the_gate():
    ref = np.zeros((1, 2, 4))
    out = ref.copy()
    out[0, 0, 0] = np.nan  # e.g. an FP16 overflow somewhere in the engine
    m = compare(ref, out, 1e-3, 1e-3, 1.0)
    checks = judge(m, {"max_flip_rate": 1.0})
    assert len(checks) == 1 and checks[0]["check"] == "finite_outputs"
    assert not checks[0]["passed"]


def test_judge_directions_and_optional_thresholds():
    metrics = {"finite": True, "cosine_similarity": 0.98, "flip_rate": 0.01, "confident_flips": 0}
    tier = {"min_cosine": 0.99, "max_flip_rate": 0.05}  # no max_confident_flips: not checked
    by_name = {c["check"]: c["passed"] for c in judge(metrics, tier)}
    assert by_name == {"cosine_similarity": False, "flip_rate": True}


def test_fp32_reference_disables_and_restores_tf32():
    before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    with fp32_reference():
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
    assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == before


def test_golden_inputs_are_reproducible():
    a = golden_inputs((1, 1, 4, 4, 4), 2, seed=7)
    b = golden_inputs((1, 1, 4, 4, 4), 2, seed=7)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


# -----------------------------------------------------------------------------
# Gate decisions on a real (tiny, untrained) SwinUNETR
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model():
    from src.utils import load_config

    torch.manual_seed(0)
    cfg = load_config(smoke_test=True)
    return build_model(cfg["model"]).eval(), cfg["parity"]


def _fp16_rounded(model):
    """Weights, input and output rounded through float16: a CPU stand-in for FP16 noise."""
    m16 = copy.deepcopy(model)
    with torch.no_grad():
        for p in m16.parameters():
            p.copy_(p.half().float())
    return lambda x: m16(x.half().float()).half().float()


def _broken(model, scale=1.01):
    """The same model with its output layer scaled by 1%: a subtle real defect."""
    bad = copy.deepcopy(model)
    with torch.no_grad():
        dict(bad.named_parameters())["out.conv.conv.weight"].mul_(scale)
    return bad


def _gate(model, predictor, pcfg, tier_name):
    inputs = golden_inputs((1, 1, *ROI), 1, seed=0)
    cpu = torch.device("cpu")
    result = run_golden(model, predictor, inputs, cpu, cpu, pcfg["tiers"][tier_name],
                        pcfg["margin_threshold"])
    return all(c["passed"] for c in result["checks"]), result


def test_identical_model_passes_strictest_tier(tiny_model):
    model, pcfg = tiny_model
    passed, _ = _gate(model, model, pcfg, "fp32")
    assert passed


def test_fp16_noise_passes_fp16_tier_but_not_fp32_tier(tiny_model):
    """The reason tiers exist: honest FP16 noise is fine for FP16, a defect for FP32."""
    model, pcfg = tiny_model
    predictor = _fp16_rounded(model)
    assert _gate(model, predictor, pcfg, "fp16")[0]
    assert not _gate(model, predictor, pcfg, "fp32")[0]


@pytest.mark.parametrize("tier", ["fp32", "fp16"])
def test_broken_model_fails_even_the_loose_tier(tiny_model, tier):
    model, pcfg = tiny_model
    passed, result = _gate(model, _broken(model), pcfg, tier)
    assert not passed
    assert "flip_rate" in [c["check"] for c in result["checks"] if not c["passed"]]


# -----------------------------------------------------------------------------
# Real ONNX export through ONNX Runtime
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def exported(tiny_model, tmp_path_factory):
    model, _ = tiny_model
    path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    export(model, path, (2, 1, *ROI), opset=18)  # fixed batch of 2, like sw_batch_size
    return path


@pytest.mark.parametrize("batch", [1, 3])  # smaller than, and not a multiple of, the fixed batch
def test_ort_predictor_pads_and_chunks(tiny_model, exported, batch):
    model, _ = tiny_model
    predictor = ORTPredictor(exported)
    x = torch.rand(batch, 1, *ROI)
    with torch.no_grad():
        ref = model(x)
    out = predictor(x)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_onnx_export_passes_fp32_gate(tiny_model, exported):
    model, pcfg = tiny_model
    passed, result = _gate(model, ORTPredictor(exported), pcfg, "fp32")
    assert passed, result["checks"]


def test_ort_predictor_rejects_wrong_spatial_shape(exported):
    with pytest.raises(ValueError):
        ORTPredictor(exported)(torch.rand(1, 1, 32, 64, 64))


# -----------------------------------------------------------------------------
# Stale-engine protection (pure file logic, so it runs without a GPU)
# -----------------------------------------------------------------------------
def _fake_engine(tmp_path, recorded):
    """An engine file plus the build record build_engine would have written."""
    import json

    from src.trt_utils import engine_meta_path

    engine = tmp_path / "model.engine"
    engine.write_bytes(b"not a real engine")
    if recorded is not None:
        engine_meta_path(engine).write_text(json.dumps(recorded))
    return engine


EXPECTED = {"onnx_sha256": "aaa", "tensorrt": "10.16.1", "gpu": "Tesla T4", "precision": "fp16"}


def test_engine_matching_its_build_record_is_reused(tmp_path):
    from src.trt_utils import engine_stale_reason

    assert engine_stale_reason(_fake_engine(tmp_path, EXPECTED), EXPECTED) is None


@pytest.mark.parametrize("key, value", [
    ("onnx_sha256", "bbb"),     # the bug this guards against: ONNX re-exported, engine not rebuilt
    ("tensorrt", "10.17.0"),    # engines do not survive a TensorRT upgrade
    ("gpu", "NVIDIA L4"),       # or a move to another GPU model
])
def test_engine_is_stale_when_any_input_changed(tmp_path, key, value):
    from src.trt_utils import engine_stale_reason

    reason = engine_stale_reason(_fake_engine(tmp_path, {**EXPECTED, key: value}), EXPECTED)
    assert reason is not None and key in reason


def test_engine_without_build_record_is_stale(tmp_path):
    """An engine built before build records existed has unknown provenance: rebuild it."""
    from src.trt_utils import engine_stale_reason

    assert "no build record" in engine_stale_reason(_fake_engine(tmp_path, None), EXPECTED)


def test_missing_engine_is_stale(tmp_path):
    from src.trt_utils import engine_stale_reason

    assert engine_stale_reason(tmp_path / "absent.engine", EXPECTED) == "no engine on disk"


def test_file_sha256_changes_with_content(tmp_path):
    from src.utils import file_sha256

    f = tmp_path / "x.onnx"
    f.write_bytes(b"graph v1")
    first = file_sha256(f)
    f.write_bytes(b"graph v2")
    assert file_sha256(f) != first
