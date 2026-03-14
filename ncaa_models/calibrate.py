"""
ncaa_models.calibrate
=====================
Stage 04 — CalibrationPipeline: create LoRA adapters in <42 seconds.

Adapter types
-------------
``'tournament'``    Fine-tune on recent tournament games; captures
                    tournament-specific dynamics (pressure, seeding effects).
``'conference'``    Fine-tune on games from a specific conference; captures
                    conference-specific play styles (SEC, Big Ten, …).
``'recalibration'`` General-purpose fast adapter for between-round updates
                    with minimal data; uses higher LR and fewer epochs.

Usage::

    from ncaa_models.neural import NCAAPredictor
    from ncaa_models.lora import apply_lora_to_model
    from ncaa_models.adapter import AdapterRegistry
    from ncaa_models.calibrate import CalibrationPipeline

    backbone = NCAAPredictor(feature_dim=28, hidden_dim=64)
    apply_lora_to_model(backbone, rank=4)

    registry = AdapterRegistry(backbone)
    pipeline = CalibrationPipeline(backbone, registry=registry, lr=2e-4)

    path = pipeline.create_adapter(
        "tournament_2025",
        adapter_type="tournament",
        X1_train=X1, X2_train=X2, y_train=y,
        time_budget=42.0,
    )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .lora import (
    LoRALinear,
    zero_lora_weights,
    TORCH_AVAILABLE,
)
from .adapter import AdapterRegistry

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Adapter type presets
# ---------------------------------------------------------------------------

#: Hyper-parameter presets for each adapter type.
_ADAPTER_PRESETS: Dict[str, Dict] = {
    "tournament": {
        "lr": 2e-4,
        "n_epochs": 5,
        "batch_size": 32,
        "description": "Fine-tune on tournament games (seeding, pressure effects).",
    },
    "conference": {
        "lr": 1e-4,
        "n_epochs": 5,
        "batch_size": 32,
        "description": "Fine-tune on conference-specific game data.",
    },
    "recalibration": {
        "lr": 5e-4,
        "n_epochs": 3,
        "batch_size": 16,
        "description": "Fast general-purpose recalibration with minimal data.",
    },
}

VALID_ADAPTER_TYPES = tuple(_ADAPTER_PRESETS.keys())


# ---------------------------------------------------------------------------
# CalibrationPipeline
# ---------------------------------------------------------------------------

class CalibrationPipeline:
    """
    End-to-end pipeline for creating LoRA adapters on an NCAAPredictor.

    The pipeline:

    1. Resets LoRA B to zero (unload any active adapter).
    2. Freezes all parameters except LoRA A and B.
    3. Trains with Adam + BCELoss for up to *n_epochs* epochs or
       until *time_budget* seconds elapse.
    4. Saves adapter weights to disk via :class:`~ncaa_models.adapter.AdapterRegistry`.

    Parameters
    ----------
    base_model : NCAAPredictor (or any nn.Module with LoRALinear layers)
        Backbone with LoRA applied.  Must already have been processed by
        :func:`~ncaa_models.lora.apply_lora_to_model`.
    registry : AdapterRegistry, optional
        Registry for managing adapter files.  Created automatically if not
        provided.
    lr : float
        Default learning rate.  Overridden per adapter type. Default 2e-4.
    weight_decay : float
        Adam weight decay.  Default 1e-5.
    device : str
        Torch device string.  Default ``'cpu'``.
    random_state : int
        Seed for DataLoader shuffling.  Default 42.
    """

    def __init__(
        self,
        base_model: object,
        registry: Optional[AdapterRegistry] = None,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for CalibrationPipeline. "
                "Install with: pip install torch"
            )
        self.base_model = base_model
        self.registry = registry if registry is not None else AdapterRegistry(base_model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_adapter(
        self,
        name: str,
        X1_train: np.ndarray,
        X2_train: np.ndarray,
        y_train: np.ndarray,
        adapter_type: str = "tournament",
        time_budget: float = 42.0,
        save_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> Path:
        """
        Train a LoRA adapter and save it to disk.

        Parameters
        ----------
        name : str
            Logical name for the adapter (e.g. ``'tournament_2025'``).
        X1_train, X2_train : np.ndarray, shape (n, feature_dim)
            Team feature arrays.
        y_train : np.ndarray, shape (n,)
            Labels: 1 if team1 wins, 0 otherwise.
        adapter_type : str
            One of ``'tournament'``, ``'conference'``, ``'recalibration'``.
        time_budget : float
            Maximum wall-clock seconds for training.  Default 42.0.
        save_dir : Path, optional
            Directory to save the adapter file.  Defaults to
            ``ncaa_models/adapters/``.
        verbose : bool
            Print per-epoch loss.  Default False.

        Returns
        -------
        Path
            Path to the saved adapter file.

        Raises
        ------
        ValueError
            If *adapter_type* is not recognised.
        """
        if adapter_type not in _ADAPTER_PRESETS:
            raise ValueError(
                f"adapter_type must be one of {VALID_ADAPTER_TYPES}, "
                f"got '{adapter_type}'."
            )

        preset = _ADAPTER_PRESETS[adapter_type]
        lr = preset["lr"]
        n_epochs = preset["n_epochs"]
        batch_size = preset["batch_size"]

        # Reset any active adapter; start from base model
        zero_lora_weights(self.base_model)

        # Move model to device
        model = self.base_model
        model.to(self.device)  # type: ignore[attr-defined]
        model.train()          # type: ignore[attr-defined]

        # Build optimiser over LoRA params only
        lora_params = [
            p for p in model.parameters()  # type: ignore[attr-defined]
            if p.requires_grad
        ]
        if not lora_params:
            raise RuntimeError(
                "No trainable parameters found. "
                "Did you call apply_lora_to_model() before creating the pipeline?"
            )

        optimizer = torch.optim.Adam(
            lora_params,
            lr=lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        # Build DataLoader
        t1 = torch.tensor(X1_train, dtype=torch.float32)
        t2 = torch.tensor(X2_train, dtype=torch.float32)
        ty = torch.tensor(y_train, dtype=torch.float32)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        dataset = TensorDataset(t1, t2, ty)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(y_train)),
            shuffle=True,
            generator=generator,
        )

        # Training loop with time budget
        start = time.perf_counter()
        history = []

        for epoch in range(n_epochs):
            elapsed = time.perf_counter() - start
            if elapsed >= time_budget:
                logger.warning(
                    "Time budget %.1fs reached after epoch %d/%d "
                    "(%.1fs elapsed). Stopping early.",
                    time_budget, epoch, n_epochs, elapsed,
                )
                break

            epoch_loss = self._train_epoch(model, loader, optimizer, criterion)
            history.append(epoch_loss)

            if verbose:
                elapsed = time.perf_counter() - start
                logger.info(
                    "Adapter '%s' [%s] epoch %d/%d  loss=%.4f  %.1fs",
                    name, adapter_type, epoch + 1, n_epochs, epoch_loss, elapsed,
                )

        total_time = time.perf_counter() - start
        model.eval()  # type: ignore[attr-defined]
        logger.info(
            "Adapter '%s' (%s) trained: %d epochs, %.2fs, final loss=%.4f",
            name, adapter_type, len(history), total_time,
            history[-1] if history else float("nan"),
        )

        # Save to registry
        if save_dir is not None:
            save_path = Path(save_dir) / f"{name}.bin"
        else:
            save_path = None

        path = self.registry.save(name, path=save_path)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: "nn.Module",
        loader: "DataLoader",
        optimizer: "torch.optim.Optimizer",
        criterion: "nn.Module",
    ) -> float:
        """Run one training epoch; return mean batch loss."""
        total_loss = 0.0
        n_batches = 0

        for x1_b, x2_b, y_b in loader:
            x1_b = x1_b.to(self.device)
            x2_b = x2_b.to(self.device)
            y_b = y_b.to(self.device)

            optimizer.zero_grad()
            pred = model(x1_b, x2_b).squeeze(-1)  # (B,)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate_brier(
        self,
        model: "nn.Module",
        X1_val: np.ndarray,
        X2_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Compute Brier score on validation data (no gradients)."""
        from .evaluate import compute_brier_score

        model.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            t1 = torch.tensor(X1_val, dtype=torch.float32, device=self.device)
            t2 = torch.tensor(X2_val, dtype=torch.float32, device=self.device)
            probs = model(t1, t2).squeeze(-1).cpu().numpy()  # type: ignore[attr-defined]

        return compute_brier_score(probs.astype(float), y_val)
