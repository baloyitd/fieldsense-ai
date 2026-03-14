"""
ncaa_models.export
==================
ONNX export, verification, and inference benchmarking for NCAAPredictor.

Requires: torch, onnx, onnxruntime

Functions
---------
export_to_onnx          : Export NCAAPredictor to ONNX format.
verify_onnx_equivalence : Compare PyTorch and ONNX outputs on random samples.
benchmark_onnx_latency  : Measure mean CPU inference time per matchup.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .neural import TORCH_AVAILABLE, NCAAPredictor

logger = logging.getLogger(__name__)

_ONNX_OPSET = 14  # requires torch >= 1.12


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch (torch) is required for ONNX export. "
            "Install with: pip install torch"
        )


def _require_onnx() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX export. "
            "Install with: pip install onnx"
        )


def _require_ort() -> None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX verification/benchmarking. "
            "Install with: pip install onnxruntime"
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: "NCAAPredictor",
    feature_dim: int,
    save_path: str | Path,
    opset: int = _ONNX_OPSET,
) -> Path:
    """
    Export a fitted NCAAPredictor to ONNX format.

    The ONNX model accepts two named float32 inputs ``x1`` and ``x2``
    (each shape [batch_size, feature_dim]) and produces one named output
    ``prob`` (shape [batch_size, 1]).  The batch dimension is dynamic.

    Parameters
    ----------
    model       : NCAAPredictor — must be in eval mode or will be set to eval.
    feature_dim : int           — per-team feature dimension.
    save_path   : str or Path   — destination for the .onnx file.
    opset       : int           — ONNX opset version. Default 14.

    Returns
    -------
    Path to the exported file.
    """
    _require_torch()
    _require_onnx()

    import torch

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_x1 = torch.zeros(1, feature_dim, dtype=torch.float32)
    dummy_x2 = torch.zeros(1, feature_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_x1, dummy_x2),
        str(save_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["x1", "x2"],
        output_names=["prob"],
        dynamic_axes={
            "x1":   {0: "batch_size"},
            "x2":   {0: "batch_size"},
            "prob": {0: "batch_size"},
        },
        verbose=False,
    )
    logger.info("ONNX model exported to %s (opset %d).", save_path, opset)
    return save_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_onnx_equivalence(
    pt_model: "NCAAPredictor",
    onnx_path: str | Path,
    feature_dim: int,
    n_samples: int = 100,
    tol: float = 1e-5,
    random_seed: int = 0,
) -> dict:
    """
    Compare PyTorch and ONNX inference outputs on random samples.

    Parameters
    ----------
    pt_model    : fitted NCAAPredictor in eval mode.
    onnx_path   : path to the exported .onnx file.
    feature_dim : per-team feature dimension.
    n_samples   : number of random matchup samples. Default 100.
    tol         : maximum allowed absolute difference. Default 1e-5.
    random_seed : numpy RNG seed. Default 0.

    Returns
    -------
    dict with keys:
        passed        : bool — True iff max_diff < tol for all samples.
        max_abs_diff  : float — worst-case absolute difference.
        mean_abs_diff : float — average absolute difference.
        n_samples     : int
        tol           : float
    """
    _require_torch()
    _require_ort()

    import torch
    import onnxruntime as ort

    rng = np.random.default_rng(random_seed)
    X1 = rng.standard_normal((n_samples, feature_dim)).astype(np.float32)
    X2 = rng.standard_normal((n_samples, feature_dim)).astype(np.float32)

    # PyTorch outputs
    pt_model.eval()
    with torch.no_grad():
        t1 = torch.tensor(X1, dtype=torch.float32)
        t2 = torch.tensor(X2, dtype=torch.float32)
        pt_probs = pt_model(t1, t2).squeeze(-1).numpy()

    # ONNX Runtime outputs
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_probs = sess.run(["prob"], {"x1": X1, "x2": X2})[0].squeeze(-1)

    abs_diff = np.abs(pt_probs - ort_probs)
    result = {
        "passed": bool(abs_diff.max() < tol),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "n_samples": n_samples,
        "tol": tol,
    }
    logger.info(
        "ONNX verification: passed=%s  max_diff=%.2e  mean_diff=%.2e",
        result["passed"], result["max_abs_diff"], result["mean_abs_diff"],
    )
    return result


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_onnx_latency(
    onnx_path: str | Path,
    feature_dim: int,
    n_warmup: int = 20,
    n_iters: int = 200,
    random_seed: int = 0,
) -> dict:
    """
    Measure ONNX Runtime CPU inference latency for a single matchup.

    Parameters
    ----------
    onnx_path   : path to the .onnx file.
    feature_dim : per-team feature dimension.
    n_warmup    : warm-up iterations (excluded from timing). Default 20.
    n_iters     : timed iterations. Default 200.
    random_seed : numpy RNG seed. Default 0.

    Returns
    -------
    dict with keys:
        mean_ms   : float — mean latency in milliseconds.
        p50_ms    : float — median latency in milliseconds.
        p99_ms    : float — 99th-percentile latency in milliseconds.
        n_iters   : int
    """
    _require_ort()

    import onnxruntime as ort

    rng = np.random.default_rng(random_seed)
    x1 = rng.standard_normal((1, feature_dim)).astype(np.float32)
    x2 = rng.standard_normal((1, feature_dim)).astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {"x1": x1, "x2": x2}

    # Warm-up
    for _ in range(n_warmup):
        sess.run(["prob"], inputs)

    # Timed runs
    times_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        sess.run(["prob"], inputs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_arr = np.array(times_ms)
    result = {
        "mean_ms": float(times_arr.mean()),
        "p50_ms": float(np.percentile(times_arr, 50)),
        "p99_ms": float(np.percentile(times_arr, 99)),
        "n_iters": n_iters,
    }
    logger.info(
        "ONNX latency benchmark: mean=%.3fms  p50=%.3fms  p99=%.3fms",
        result["mean_ms"], result["p50_ms"], result["p99_ms"],
    )
    return result
