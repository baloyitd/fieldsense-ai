"""
ncaa_models.training
====================
Training infrastructure for NCAAPredictor (torch backend).

Components
----------
EarlyStopping       : Monitor validation Brier; stop and restore best weights.
MatchupDataset      : PyTorch Dataset with 50% team-swap augmentation.
train_epoch         : Single training epoch (BCELoss + Adam).
validate_brier      : Compute validation Brier score (no gradients).
train_model         : Full training loop with LR scheduling + early stopping.

This module requires PyTorch. The sklearn fallback in NCAANeuralModel
does not use these utilities.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Optional, Tuple

import numpy as np

from .evaluate import compute_brier_score
from .neural import TORCH_AVAILABLE

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor a scalar metric; stop training when no improvement for ``patience``
    consecutive checks.  Restores the best model weights on ``restore_best()``.

    Parameters
    ----------
    patience  : int   Number of non-improving epochs before stopping. Default 10.
    min_delta : float Minimum absolute improvement to count as improvement. Default 1e-4.
    mode      : str   'min' (lower is better) or 'max' (higher is better).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, metric: float, model: "nn.Module") -> None:
        """
        Update state based on the new metric value.

        Sets ``should_stop = True`` after ``patience`` non-improving epochs.
        Saves the model state whenever a new best is reached.
        """
        if self.best is None:
            self.best = metric
            self.best_state = copy.deepcopy(model.state_dict())
            return

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: "nn.Module") -> None:
        """Load the best saved weights back into ``model``."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# PyTorch Dataset with augmentation
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class MatchupDataset(Dataset):
        """
        PyTorch Dataset for matchup data.

        Parameters
        ----------
        X1, X2 : np.ndarray, shape (N, feature_dim)
        y       : np.ndarray, shape (N,)  binary labels
        augment : bool  if True, randomly swap team order 50% of the time.
        seed    : int   base RNG seed for augmentation.
        """

        def __init__(
            self,
            X1: np.ndarray,
            X2: np.ndarray,
            y: np.ndarray,
            augment: bool = True,
            seed: int = 42,
        ) -> None:
            self.X1 = torch.tensor(X1, dtype=torch.float32)
            self.X2 = torch.tensor(X2, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.augment = augment
            self._rng_seed = seed

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int):
            x1, x2, label = self.X1[idx], self.X2[idx], self.y[idx]
            if self.augment and torch.rand(1).item() < 0.5:
                # Swap teams and flip label to enforce order-agnostic learning
                x1, x2, label = x2, x1, 1.0 - label
            return x1, x2, label


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(
    model: "nn.Module",
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    criterion: "nn.Module",
    device: str,
) -> float:
    """
    Run one training epoch.

    Returns
    -------
    float : mean BCELoss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x1_batch, x2_batch, y_batch in loader:
        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x1_batch, x2_batch).squeeze(-1)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_brier(
    model: "nn.Module",
    X1_val: np.ndarray,
    X2_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
) -> float:
    """
    Compute validation Brier score (no gradient updates).

    Returns
    -------
    float : Brier score on the validation set.
    """
    model.eval()
    with torch.no_grad():
        t1 = torch.tensor(X1_val, dtype=torch.float32, device=device)
        t2 = torch.tensor(X2_val, dtype=torch.float32, device=device)
        probs = model(t1, t2).squeeze(-1).cpu().numpy()
    return compute_brier_score(probs, y_val)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    model: "nn.Module",
    X1_train: np.ndarray,
    X2_train: np.ndarray,
    y_train: np.ndarray,
    val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 10,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    device: str = "cpu",
    random_state: int = 42,
    augment: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Train NCAAPredictor with Adam, ReduceLROnPlateau, and early stopping.

    Parameters
    ----------
    model        : NCAAPredictor (nn.Module)
    X1_train, X2_train, y_train : training arrays
    val_data     : optional (X1_val, X2_val, y_val) for early stopping.
                   If None, no early stopping is applied.
    lr           : initial Adam learning rate. Default 1e-3.
    weight_decay : Adam L2 regularisation. Default 1e-4.
    batch_size   : mini-batch size. Default 64.
    max_epochs   : maximum number of epochs. Default 200.
    patience     : early stopping patience on val Brier. Default 10.
    lr_patience  : ReduceLROnPlateau patience. Default 5.
    lr_factor    : LR reduction factor. Default 0.5.
    device       : 'cpu' or 'cuda'. Default 'cpu'.
    random_state : random seed for augmentation and DataLoader. Default 42.
    augment      : enable team-swap augmentation. Default True.
    verbose      : print per-epoch progress. Default False.

    Returns
    -------
    dict with keys:
        train_losses  : list[float] — per-epoch mean BCELoss
        val_briers    : list[float] — per-epoch validation Brier (if val provided)
        best_val_brier: float       — best validation Brier achieved
        stopped_epoch : int         — epoch at which training stopped
        total_time_s  : float       — wall-clock training time in seconds
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_model.")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = model.to(device)
    effective_bs = min(batch_size, len(y_train))

    dataset = MatchupDataset(X1_train, X2_train, y_train, augment=augment, seed=random_state)
    loader = DataLoader(dataset, batch_size=effective_bs, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=lr_patience, factor=lr_factor, verbose=False
    )

    early_stop = EarlyStopping(patience=patience, mode="min") if val_data is not None else None

    train_losses: list = []
    val_briers: list = []
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        if val_data is not None:
            X1_v, X2_v, y_v = val_data
            val_brier = validate_brier(model, X1_v, X2_v, y_v, device)
            val_briers.append(val_brier)
            scheduler.step(val_brier)

            if early_stop is not None:
                early_stop.step(val_brier, model)

            if verbose:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.debug(
                    "Epoch %d/%d  loss=%.4f  val_brier=%.4f  lr=%.2e",
                    epoch, max_epochs, train_loss, val_brier, current_lr,
                )

            if early_stop is not None and early_stop.should_stop:
                if verbose:
                    logger.info("Early stopping at epoch %d.", epoch)
                break
        elif verbose:
            logger.debug("Epoch %d/%d  loss=%.4f", epoch, max_epochs, train_loss)

    # Restore best weights if early stopping was used
    if early_stop is not None:
        early_stop.restore_best(model)

    elapsed = time.perf_counter() - start
    best_val = early_stop.best if early_stop is not None else (val_briers[-1] if val_briers else float("nan"))

    history = {
        "train_losses": train_losses,
        "val_briers": val_briers,
        "best_val_brier": best_val,
        "stopped_epoch": len(train_losses),
        "total_time_s": elapsed,
    }
    logger.info(
        "Training complete: %d epochs in %.1fs. Best val Brier: %.4f",
        len(train_losses), elapsed, best_val,
    )
    return history
