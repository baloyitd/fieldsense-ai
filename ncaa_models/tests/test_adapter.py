"""
test_adapter.py
===============
Unit tests for ncaa_models.adapter — AdapterRegistry.

All tests that require PyTorch are skipped when torch is not installed.
"""

from __future__ import annotations

import pytest

from ncaa_models.lora import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_with_lora():
    """NCAAPredictor with LoRA applied to the matchup head."""
    pytest.importorskip("torch")
    from ncaa_models.neural import NCAAPredictor
    from ncaa_models.lora import apply_lora_to_model
    model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
    apply_lora_to_model(model, rank=4)
    model.eval()
    return model


@pytest.fixture
def registry(model_with_lora):
    from ncaa_models.adapter import AdapterRegistry
    return AdapterRegistry(model_with_lora)


@pytest.fixture
def saved_adapter(registry, tmp_path):
    """Save a (zero) adapter and return (name, path)."""
    import torch
    from ncaa_models.lora import LoRALinear
    # Make B non-zero so the adapter has a distinguishable effect
    for mod in registry.base_model.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.lora_B.normal_(std=0.5)

    name = "test_adapter"
    path = registry.save(name, path=tmp_path / f"{name}.bin")
    return name, path


# ---------------------------------------------------------------------------
# TestAdapterRegistry
# ---------------------------------------------------------------------------

class TestAdapterRegistry:

    def test_init_no_active_adapter(self, registry):
        assert registry.active_adapter is None

    def test_list_adapters_empty(self, registry):
        assert registry.list_adapters() == []

    def test_register_valid_path(self, registry, saved_adapter):
        name, path = saved_adapter
        # Already registered by save(); confirm it appears in list
        assert name in registry.list_adapters()

    def test_register_missing_file_raises(self, registry):
        with pytest.raises(FileNotFoundError):
            registry.register("ghost", "/nonexistent/adapter.bin")

    def test_save_creates_file(self, registry, tmp_path):
        path = registry.save("zero_adapter", path=tmp_path / "zero.bin")
        assert path.exists()

    def test_saved_file_below_1mb(self, registry, tmp_path):
        path = registry.save("size_test", path=tmp_path / "size.bin")
        size_bytes = path.stat().st_size
        assert size_bytes < 1_048_576, (
            f"Adapter file {size_bytes} bytes ≥ 1 MB"
        )

    def test_get_size_bytes(self, registry, saved_adapter):
        name, path = saved_adapter
        size = registry.get_size_bytes(name)
        assert isinstance(size, int)
        assert size > 0
        assert size == path.stat().st_size

    def test_get_size_bytes_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_size_bytes("nonexistent")

    def test_load_changes_predictions(self, model_with_lora, saved_adapter, registry):
        """After load, predictions differ from the zero-adapter baseline."""
        import torch
        from ncaa_models.lora import zero_lora_weights

        name, _ = saved_adapter

        x1 = torch.randn(6, 10)
        x2 = torch.randn(6, 10)

        # Baseline: adapter zeroed out
        zero_lora_weights(model_with_lora)
        model_with_lora.eval()
        with torch.no_grad():
            base_preds = model_with_lora(x1, x2).clone()

        # Load adapter
        registry.load(name)
        model_with_lora.eval()
        with torch.no_grad():
            loaded_preds = model_with_lora(x1, x2)

        # Predictions should differ (B was set non-zero when saving)
        assert not torch.allclose(base_preds, loaded_preds), (
            "Loaded adapter should change predictions"
        )

    def test_unload_restores_base_predictions(self, model_with_lora, saved_adapter, registry):
        """After load then unload, predictions match baseline (B=0)."""
        import torch
        from ncaa_models.lora import zero_lora_weights

        name, _ = saved_adapter

        x1 = torch.randn(6, 10)
        x2 = torch.randn(6, 10)

        # Record predictions with zeroed adapter
        zero_lora_weights(model_with_lora)
        model_with_lora.eval()
        with torch.no_grad():
            base_preds = model_with_lora(x1, x2).clone()

        # Load and then unload
        registry.load(name)
        registry.unload()

        model_with_lora.eval()
        with torch.no_grad():
            after_unload = model_with_lora(x1, x2)

        torch.testing.assert_close(base_preds, after_unload)

    def test_active_adapter_set_on_load(self, registry, saved_adapter):
        name, _ = saved_adapter
        registry.load(name)
        assert registry.active_adapter == name

    def test_active_adapter_cleared_on_unload(self, registry, saved_adapter):
        name, _ = saved_adapter
        registry.load(name)
        registry.unload()
        assert registry.active_adapter is None

    def test_load_unknown_adapter_raises(self, registry):
        with pytest.raises(KeyError, match="not found"):
            registry.load("nonexistent_adapter")

    def test_list_adapters_sorted(self, registry, tmp_path):
        registry.save("zebra", path=tmp_path / "zebra.bin")
        registry.save("apple", path=tmp_path / "apple.bin")
        names = registry.list_adapters()
        assert names == sorted(names)

    def test_multiple_adapters_hot_swap(self, model_with_lora, registry, tmp_path):
        """Load adapter A, load adapter B → predictions differ between the two."""
        import torch
        from ncaa_models.lora import LoRALinear

        x1 = torch.randn(6, 10)
        x2 = torch.randn(6, 10)

        # Adapter A: B filled with +1
        for mod in model_with_lora.modules():
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    mod.lora_B.fill_(1.0)
        registry.save("adapter_A", path=tmp_path / "A.bin")

        # Adapter B: B filled with -1
        for mod in model_with_lora.modules():
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    mod.lora_B.fill_(-1.0)
        registry.save("adapter_B", path=tmp_path / "B.bin")

        registry.load("adapter_A")
        model_with_lora.eval()
        with torch.no_grad():
            preds_A = model_with_lora(x1, x2).clone()

        registry.load("adapter_B")
        model_with_lora.eval()
        with torch.no_grad():
            preds_B = model_with_lora(x1, x2).clone()

        assert not torch.allclose(preds_A, preds_B), (
            "Hot-swap between adapter_A and adapter_B should change predictions"
        )

    def test_repr(self, registry):
        r = repr(registry)
        assert "AdapterRegistry" in r
