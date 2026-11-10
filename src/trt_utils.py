"""TensorRT engine building and execution (NVIDIA GPUs only).

IMPORTANT: this module imports ``tensorrt`` lazily, inside functions, and only
after checking ``torch.cuda.is_available()``. Importing ``src.trt_utils`` on a
Mac or a CPU-only machine is therefore safe; *calling* it there raises a clear
``RuntimeError`` instead of an obscure import failure.

What TensorRT does
------------------
TensorRT takes the ONNX graph and compiles it into an "engine" tuned for one
specific GPU model and TensorRT version:
  * fuses layers (e.g. conv + bias + activation become one kernel);
  * with the FP16 flag, runs layers in half precision on Tensor Cores where
    that does not hurt accuracy, keeping sensitive layers in fp32;
  * benchmarks several kernel implementations per layer and keeps the fastest.

An engine built on a T4 will not load on a different GPU or TensorRT version,
which is why ``.engine`` files are git-ignored and rebuilt on demand.

Stale engines
-------------
Building takes minutes, so engines are cached on disk. The danger of a cache:
re-export the ONNX file, forget to rebuild, and every later test silently runs
the *old* engine. To prevent that, ``build_engine`` writes a small sidecar
file next to the engine (``<engine>.json``) recording the SHA-256 of the ONNX
file it was built from, the TensorRT version and the GPU name.
``ensure_engine`` compares that record with the current situation and
rebuilds on any mismatch.

This module uses the TensorRT 10 Python API (pinned in requirements-colab.txt) (``execute_async_v3`` with named
tensors) and shares GPU memory directly with PyTorch tensors, so no extra
CUDA bindings package (pycuda / cuda-python) is needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from src.utils import file_sha256


def _require_tensorrt():
    """Import TensorRT, or explain why it is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT needs an NVIDIA GPU with CUDA; none is available here.")
    try:
        import tensorrt as trt
    except ImportError as e:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "TensorRT is not installed. Run: pip install tensorrt-cu<12 or 13>==10.16.1.11, "
            f"matching PyTorch's CUDA {torch.version.cuda}"
        ) from e
    return trt


def tensorrt_available() -> bool:
    """True if CUDA is present and the ``tensorrt`` package imports."""
    try:
        _require_tensorrt()
        return True
    except RuntimeError:
        return False


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True, workspace_gb: float = 4) -> dict:
    """Compile an ONNX file into a serialized TensorRT engine on disk.

    Returns a small dict with build time and engine size for the results file.
    Building takes a few minutes on a T4 because TensorRT times many kernels.
    """
    trt = _require_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 10: explicit batch is the only mode
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 2**30))
    if fp16:
        if not getattr(builder, "platform_has_fast_fp16", True):  # deprecated in newer TRT
            print("Warning: this GPU has no fast FP16 path; building anyway.")
        config.set_flag(trt.BuilderFlag.FP16)

    start = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see log above).")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    meta = {
        "onnx_sha256": file_sha256(onnx_path),
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "precision": "fp16" if fp16 else "fp32",
    }
    engine_meta_path(engine_path).write_text(json.dumps(meta, indent=2) + "\n")
    return {
        **meta,
        "build_seconds": time.time() - start,
        "engine_size_mb": engine_path.stat().st_size / 2**20,
    }


def engine_meta_path(engine_path: Path) -> Path:
    """Where the sidecar record for ``engine_path`` lives: ``<engine>.json``."""
    return engine_path.with_name(engine_path.name + ".json")


def engine_stale_reason(engine_path: Path, expected: dict[str, Any]) -> str | None:
    """Why the cached engine cannot be trusted, or ``None`` if it can.

    ``expected`` holds what the engine *should* have been built from, with the
    same keys ``build_engine`` records (``onnx_sha256``, ``tensorrt``, ``gpu``,
    ``precision``). Pure file and dict logic, so it is unit-tested on CPU.
    """
    if not engine_path.exists():
        return "no engine on disk"
    meta_path = engine_meta_path(engine_path)
    if not meta_path.exists():
        return f"no build record ({meta_path.name}); provenance unknown"
    try:
        recorded = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return f"unreadable build record ({meta_path.name})"
    for key, want in expected.items():
        if recorded.get(key) != want:
            return f"{key} changed: engine built from {recorded.get(key)!r}, now {want!r}"
    return None


def ensure_engine(onnx_path: Path, engine_path: Path, workspace_gb: float,
                  force: bool = False) -> dict[str, Any]:
    """Return a TensorRT FP16 engine for ``onnx_path``, rebuilding only if needed.

    Reuses the cached engine when its build record matches the current ONNX
    file, TensorRT version and GPU; otherwise (or with ``force``) rebuilds it.
    The returned dict says which happened and why, for the results file.
    """
    trt = _require_tensorrt()
    expected = {
        "onnx_sha256": file_sha256(onnx_path),
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "precision": "fp16",
    }
    reason = "--rebuild-engine requested" if force else engine_stale_reason(engine_path, expected)
    if reason is None:
        return {"engine_path": str(engine_path), "rebuilt": False, **expected}
    print(f"Building TensorRT FP16 engine from {onnx_path} ({reason}); takes a few minutes...")
    info = build_engine(onnx_path, engine_path, fp16=True, workspace_gb=workspace_gb)
    return {"engine_path": str(engine_path), "rebuilt": True, "rebuild_reason": reason, **info}


class TRTPredictor:
    """Callable wrapper that makes a TensorRT engine look like a PyTorch model.

    ``predictor(x)`` takes a CUDA float tensor of shape ``(n, 1, *roi)`` and
    returns logits ``(n, C, *roi)``. The engine was built for exactly one batch
    size ``B`` (= ``sw_batch_size``); sliding-window inference can hand us a
    smaller final batch, so we zero-pad up to ``B`` and slice the result.
    """

    def __init__(self, engine_path: Path):
        trt = _require_tensorrt()
        self.trt = trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load {engine_path} (built for another GPU/TensorRT version?)")
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        is_input = lambda n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT  # noqa: E731
        self.input_name = next(n for n in names if is_input(n))
        self.output_name = next(n for n in names if not is_input(n))
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.batch = self.input_shape[0]
        # Pre-allocate input/output buffers once and reuse them for every call.
        self.input_buf = torch.zeros(self.input_shape, dtype=torch.float32, device="cuda")
        self.output_buf = torch.empty(self.output_shape, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.input_name, self.input_buf.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buf.data_ptr())

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        self.input_buf[:n].copy_(x)
        if n < self.batch:
            self.input_buf[n:].zero_()
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        # Clone so the next call cannot overwrite a result still in use.
        return self.output_buf[:n].clone()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().contiguous()
        if x.shape[1:] != self.input_shape[1:]:
            raise ValueError(f"engine expects (*, {self.input_shape[1:]}), got {tuple(x.shape)}")
        chunks = [self._run(x[i:i + self.batch]) for i in range(0, x.shape[0], self.batch)]
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]
