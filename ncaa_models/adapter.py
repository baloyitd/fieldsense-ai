"""
ncaa_models.adapter
===================
Stage 04 — AdapterRegistry: store, load, and hot-swap LoRA adapters.

The registry wraps an ``NCAAPredictor`` that has already had LoRA layers
applied via :func:`~ncaa_models.lora.apply_lora_to_model`.  It manages a
catalogue of saved adapter weight files and supports:

* **Hot-swap**: load a named adapter into the live model in O(1) time.
* **Unload**: reset adapter to zero (B=0) → base model behaviour restored.
* **Introspection**: list registered adapters, query file sizes.

Usage example::

    from ncaa_models.neural import NCAAPredictor
    from ncaa_models.lora import apply_lora_to_model
    from ncaa_models.adapter import AdapterRegistry

    backbone = NCAAPredictor(feature_dim=28, hidden_dim=64)
    apply_lora_to_model(backbone, rank=4)

    registry = AdapterRegistry(backbone)
    registry.save("tournament_2025", path="ncaa_models/adapters/tournament_2025.bin")
    registry.load("tournament_2025")   # hot-swap
    registry.unload()                  # revert to base
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .lora import (
    LoRALinear,
    get_lora_state_dict,
    load_lora_state_dict,
    zero_lora_weights,
    TORCH_AVAILABLE,
)

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    import torch


# Default directory for adapter files
_DEFAULT_ADAPTER_DIR = Path(__file__).parent / "adapters"


class AdapterRegistry:
    """
    Manages a catalogue of LoRA adapter weight files for one backbone model.

    Parameters
    ----------
    base_model : NCAAPredictor (or any nn.Module with LoRALinear layers)
        The model whose LoRA weights are managed.  Must already have LoRA
        applied via :func:`~ncaa_models.lora.apply_lora_to_model`.

    Notes
    -----
    * ``load(name)`` copies adapter tensors into the live model → instant
      hot-swap without creating a new model object.
    * ``unload()`` resets lora_B to zeros → predictions identical to
      base model (no LoRA contribution since scale * B @ A = 0).
    * The registry does **not** apply LoRA itself; call
      :func:`~ncaa_models.lora.apply_lora_to_model` first.
    """

    def __init__(self, base_model: object) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for AdapterRegistry. "
                "Install with: pip install torch"
            )
        self.base_model = base_model
        self._adapters: Dict[str, Path] = {}
        self._active: Optional[str] = None

    # ------------------------------------------------------------------
    # Catalogue management
    # ------------------------------------------------------------------

    def register(self, name: str, path: "str | Path") -> None:
        """
        Register an existing adapter file without loading it.

        Parameters
        ----------
        name : str
            Logical adapter name (e.g. ``'tournament_2025'``).
        path : str or Path
            Path to the saved ``.bin`` adapter file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter file not found: {path}")
        self._adapters[name] = path
        logger.debug("Registered adapter '%s' from %s", name, path)

    def save(
        self,
        name: str,
        path: "Optional[str | Path]" = None,
    ) -> Path:
        """
        Save the current LoRA weights to a file and register the adapter.

        Parameters
        ----------
        name : str
            Logical name for this adapter.
        path : str or Path, optional
            Destination file path.  Defaults to
            ``ncaa_models/adapters/{name}.bin``.

        Returns
        -------
        Path
            The path where the adapter was saved.
        """
        if path is None:
            path = _DEFAULT_ADAPTER_DIR / f"{name}.bin"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = get_lora_state_dict(self.base_model)
        # Include metadata for provenance
        payload = {
            "name": name,
            "state_dict": state,
        }
        torch.save(payload, path)
        self._adapters[name] = path
        logger.info(
            "Saved adapter '%s' → %s (%d tensors, %.1f KB)",
            name,
            path,
            len(state),
            path.stat().st_size / 1024,
        )
        return path

    # ------------------------------------------------------------------
    # Hot-swap
    # ------------------------------------------------------------------

    def load(self, name: str) -> None:
        """
        Load a registered adapter into the live model.

        Replaces the current LoRA weights with those from the saved file.
        No model copy is made; the operation is O(num_lora_params).

        Parameters
        ----------
        name : str
            Adapter name previously registered with :meth:`register` or
            :meth:`save`.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in self._adapters:
            raise KeyError(
                f"Adapter '{name}' not found. "
                f"Registered: {list(self._adapters.keys())}"
            )
        path = self._adapters[name]
        payload = torch.load(path, map_location="cpu")
        state = payload["state_dict"] if "state_dict" in payload else payload
        load_lora_state_dict(self.base_model, state)
        self._active = name
        logger.info("Loaded adapter '%s' from %s", name, path)

    def unload(self) -> None:
        """
        Reset all lora_B matrices to zero, restoring base model behaviour.

        Because ``output = base(x) + scale * B @ A @ x`` and B is zeroed,
        the adapter contribution is exactly zero — predictions are identical
        to the original frozen backbone.

        This is faster than :func:`~ncaa_models.lora.remove_lora` because
        LoRALinear wrappers stay in place (ready for the next :meth:`load`).
        """
        zero_lora_weights(self.base_model)
        prev = self._active
        self._active = None
        logger.info("Unloaded adapter '%s' (B reset to zero)", prev)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_adapter(self) -> Optional[str]:
        """Name of the currently loaded adapter, or ``None``."""
        return self._active

    def list_adapters(self) -> List[str]:
        """Return a sorted list of registered adapter names."""
        return sorted(self._adapters.keys())

    def get_size_bytes(self, name: str) -> int:
        """
        Return the on-disk size (bytes) of a registered adapter file.

        Parameters
        ----------
        name : str
            Adapter name.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not in registry.")
        return self._adapters[name].stat().st_size

    def __repr__(self) -> str:
        return (
            f"AdapterRegistry("
            f"n_adapters={len(self._adapters)}, "
            f"active={self._active!r})"
        )
