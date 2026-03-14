"""
ncaa_models.lora
================
Stage 04 — LoRA (Low-Rank Adaptation) for the NCAAPredictor backbone.

Architecture
------------
LoRALinear wraps an existing nn.Linear with a low-rank adapter:

    output = original_linear(x) + scale * B(A(x))

where:
    A     : (rank, in_features)   — Kaiming uniform init
    B     : (out_features, rank)  — zeros init (zero effect at start)
    scale : alpha / rank

Only A and B are trainable; original weights are frozen.  This enables rapid
calibration (<42 s) with <1MB adapter files and minimal catastrophic forgetting.

Public API
----------
LoRALinear           : nn.Module that wraps one nn.Linear with LoRA.
apply_lora_to_model  : Replace target Linear layers in NCAAPredictor with LoRALinear.
remove_lora          : Revert LoRALinear back to standard Linear (fuse optional).
get_lora_state_dict  : Extract only LoRA (A, B) weights from a model.
load_lora_state_dict : Inject LoRA weights into a model's LoRALinear layers.
count_trainable_params : Count parameters with requires_grad=True.
count_total_params     : Count all parameters.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set

from .neural import TORCH_AVAILABLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target modules (layer names in NCAAPredictor)
# By default we adapt the matchup head only — gives the best calibration
# signal while keeping LoRA params << model params.
# ---------------------------------------------------------------------------

#: Linear layer names in NCAAPredictor's team_encoder Sequential.
ENCODER_TARGET_MODULES: Set[str] = {
    "team_encoder.0",  # Linear(feature_dim, hidden_dim)
    "team_encoder.3",  # Linear(hidden_dim, hidden_dim)
}

#: Linear layer names in NCAAPredictor's matchup_net Sequential.
MATCHUP_TARGET_MODULES: Set[str] = {
    "matchup_net.0",  # Linear(hidden_dim*2, hidden_dim)
    "matchup_net.2",  # Linear(hidden_dim, 1)
}

#: Default: adapt only the matchup head (avoids touching shared encoder).
DEFAULT_TARGET_MODULES: Set[str] = MATCHUP_TARGET_MODULES

#: Full set of all adaptable layers.
ALL_TARGET_MODULES: Set[str] = ENCODER_TARGET_MODULES | MATCHUP_TARGET_MODULES

# ---------------------------------------------------------------------------
# Conditional base class (so the module imports cleanly without torch)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    _LoRABase = nn.Module
else:
    _LoRABase = object  # type: ignore[assignment,misc]
    torch = None       # type: ignore[assignment]
    nn = None          # type: ignore[assignment]


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class LoRALinear(_LoRABase):  # type: ignore[misc]
    """
    Wraps an existing ``nn.Linear`` with a low-rank adapter.

    Forward pass::

        output = original_linear(x) + scale * B(A(x))

    where ``scale = alpha / rank``.

    Parameters
    ----------
    original_linear : nn.Linear
        The frozen base layer to wrap.
    rank : int
        LoRA rank (number of low-rank columns/rows).  Default 4.
    alpha : float
        LoRA scaling factor.  ``scale = alpha / rank``.  Default 1.0.

    Notes
    -----
    * ``A`` is initialised with Kaiming uniform; ``B`` is initialised to
      **zeros**, so at construction the adapter has zero effect.
    * Only ``lora_A`` and ``lora_B`` have ``requires_grad=True``.
    * The original weight/bias tensors are frozen (``requires_grad=False``).
    """

    def __init__(
        self,
        original_linear: "nn.Linear",
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LoRALinear. "
                "Install with: pip install torch"
            )
        super().__init__()

        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        self.rank = rank
        self.alpha = float(alpha)
        self.scale = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Preserve original layer — freeze its parameters
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad_(False)

        # Low-rank adapter matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # A: Kaiming uniform; B: zeros (zero adapter effect at init)
        nn.init.kaiming_uniform_(self.lora_A, nonlinearity="linear")

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Compute ``original_linear(x) + scale * B @ A @ x``.

        Parameters
        ----------
        x : Tensor, shape (..., in_features)

        Returns
        -------
        Tensor, shape (..., out_features)
        """
        base = self.original_linear(x)
        # LoRA path: x -> A (rank) -> B (out_features), then scale
        lora = (x @ self.lora_A.T) @ self.lora_B.T
        return base + self.scale * lora

    def extra_repr(self) -> str:
        lin = self.original_linear
        return (
            f"in_features={lin.in_features}, "
            f"out_features={lin.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scale={self.scale:.4f}"
        )


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _navigate_to_parent(model: "nn.Module", dotted_name: str):
    """
    Given a dotted name like ``'team_encoder.0'``, return
    ``(parent_module, child_name_str)`` where *child_name_str* is the last
    component (``'0'``).
    """
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        try:
            parent = parent[int(part)]
        except (TypeError, ValueError):
            parent = getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: "nn.Module", child_name: str, new_module: "nn.Module") -> None:
    """Set a child module on *parent* using integer (Sequential) or attr access."""
    try:
        parent[int(child_name)] = new_module  # type: ignore[index]
    except (TypeError, ValueError):
        setattr(parent, child_name, new_module)


# ---------------------------------------------------------------------------
# apply_lora_to_model
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: "nn.Module",
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[Set[str]] = None,
) -> Dict[str, "LoRALinear"]:
    """
    Replace target ``nn.Linear`` layers in *model* with :class:`LoRALinear`.

    Parameters
    ----------
    model : nn.Module
        The model to adapt in-place (typically an :class:`NCAAPredictor`).
    rank : int
        LoRA rank.  Default 4.
    alpha : float
        LoRA scaling factor.  Default 1.0.
    target_modules : set of str, optional
        Dotted module names to wrap.  Defaults to
        :data:`DEFAULT_TARGET_MODULES` (matchup head only).

    Returns
    -------
    dict
        Mapping of ``{module_name: LoRALinear}`` for every replaced layer.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for apply_lora_to_model.")

    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    # Snapshot (name, module) pairs BEFORE modifying the model
    all_named = list(model.named_modules())

    replaced: Dict[str, LoRALinear] = {}
    for name, module in all_named:
        if name in target_modules and isinstance(module, nn.Linear):
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            parent, child_name = _navigate_to_parent(model, name)
            _set_child(parent, child_name, lora_layer)
            replaced[name] = lora_layer
            logger.debug("Wrapped %s → LoRALinear(rank=%d, alpha=%.1f)", name, rank, alpha)

    logger.info(
        "Applied LoRA to %d/%d target layers: %s",
        len(replaced),
        len(target_modules),
        sorted(replaced.keys()),
    )
    return replaced


# ---------------------------------------------------------------------------
# remove_lora
# ---------------------------------------------------------------------------

def remove_lora(model: "nn.Module", fuse: bool = False) -> None:
    """
    Remove all :class:`LoRALinear` wrappers from *model* in-place.

    Parameters
    ----------
    model : nn.Module
        Model containing LoRALinear layers.
    fuse : bool
        If ``True``, fold adapter weights into the original layer before
        removal::

            W_fused = W_original + scale * B @ A

        If ``False`` (default), discard adapter weights and restore the
        original frozen weights as trainable.  Predictions revert exactly
        to the pre-adaptation state.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for remove_lora.")

    all_named = list(model.named_modules())
    for name, module in all_named:
        if not isinstance(module, LoRALinear):
            continue

        lin = module.original_linear

        if fuse:
            with torch.no_grad():
                lin.weight.data += module.scale * (module.lora_B @ module.lora_A)
            lin.weight.requires_grad_(True)
        else:
            lin.weight.requires_grad_(True)
            if lin.bias is not None:
                lin.bias.requires_grad_(True)

        parent, child_name = _navigate_to_parent(model, name)
        _set_child(parent, child_name, lin)
        logger.debug("Restored %s → nn.Linear (fuse=%s)", name, fuse)

    logger.info("LoRA removed from model (fuse=%s)", fuse)


# ---------------------------------------------------------------------------
# State-dict helpers
# ---------------------------------------------------------------------------

def get_lora_state_dict(model: "nn.Module") -> Dict[str, "torch.Tensor"]:
    """
    Return a dict containing only the LoRA adapter tensors (A and B).

    The keys follow the pattern ``'{module_name}.lora_A'`` and
    ``'{module_name}.lora_B'``.

    Parameters
    ----------
    model : nn.Module
        Model with LoRALinear layers.

    Returns
    -------
    dict
        ``{str: Tensor}`` — adapter weight tensors only.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for get_lora_state_dict.")

    state: Dict[str, "torch.Tensor"] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.data.clone()
            state[f"{name}.lora_B"] = module.lora_B.data.clone()
    return state


def load_lora_state_dict(
    model: "nn.Module",
    state_dict: Dict[str, "torch.Tensor"],
) -> None:
    """
    Inject LoRA adapter weights from *state_dict* into the model's
    :class:`LoRALinear` layers.

    Parameters
    ----------
    model : nn.Module
        Model with LoRALinear layers.
    state_dict : dict
        Mapping from ``'{name}.lora_A'`` / ``'{name}.lora_B'`` to tensors,
        as produced by :func:`get_lora_state_dict`.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    KeyError
        If a LoRALinear layer is found but its keys are absent from
        *state_dict*.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for load_lora_state_dict.")

    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        key_A = f"{name}.lora_A"
        key_B = f"{name}.lora_B"
        if key_A in state_dict:
            module.lora_A.data.copy_(state_dict[key_A])
        if key_B in state_dict:
            module.lora_B.data.copy_(state_dict[key_B])


def zero_lora_weights(model: "nn.Module") -> None:
    """
    Reset all lora_B matrices to zero, restoring base model behaviour.

    This is the fast "unload" operation: setting B=0 makes the LoRA
    contribution zero without removing the LoRALinear wrappers.

    Parameters
    ----------
    model : nn.Module
        Model with LoRALinear layers.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for zero_lora_weights.")

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_B.data.zero_()


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_trainable_params(model: "nn.Module") -> int:
    """Return the number of trainable parameters (``requires_grad=True``)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for count_trainable_params.")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: "nn.Module") -> int:
    """Return the total number of parameters (trainable + frozen)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for count_total_params.")
    return sum(p.numel() for p in model.parameters())


def count_lora_params(model: "nn.Module") -> int:
    """Return the number of LoRA adapter parameters (A and B tensors only)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for count_lora_params.")
    total = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            total += module.lora_A.numel() + module.lora_B.numel()
    return total
