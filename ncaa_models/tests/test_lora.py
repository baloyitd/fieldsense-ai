"""
test_lora.py
============
Unit tests for ncaa_models.lora — LoRALinear and model-level utilities.

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
def small_model():
    """Minimal NCAAPredictor for fast testing."""
    torch = pytest.importorskip("torch")
    from ncaa_models.neural import NCAAPredictor
    model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
    model.eval()
    return model


@pytest.fixture
def lora_linear():
    """A LoRALinear wrapping a (8→4) Linear."""
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from ncaa_models.lora import LoRALinear
    base = nn.Linear(8, 4)
    return LoRALinear(base, rank=4, alpha=1.0)


# ---------------------------------------------------------------------------
# TestLoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:

    def test_output_shape(self, lora_linear):
        import torch
        x = torch.randn(5, 8)
        out = lora_linear(x)
        assert out.shape == (5, 4)

    def test_init_zero_effect(self):
        """B=0 at init → LoRALinear output == original Linear output."""
        import torch
        import torch.nn as nn
        from ncaa_models.lora import LoRALinear
        base = nn.Linear(8, 4, bias=True)
        lora = LoRALinear(base, rank=2, alpha=1.0)

        x = torch.randn(10, 8)
        with torch.no_grad():
            out_lora = lora(x)
            out_base = base(x)
        # B is all zeros → lora contribution = 0
        torch.testing.assert_close(out_lora, out_base)

    def test_lora_B_zeros_at_init(self, lora_linear):
        import torch
        assert lora_linear.lora_B.data.abs().max().item() == 0.0

    def test_lora_A_nonzero_at_init(self, lora_linear):
        """A is initialised with Kaiming uniform — should be nonzero."""
        assert lora_linear.lora_A.data.abs().sum().item() > 0.0

    def test_only_lora_params_trainable(self, lora_linear):
        trainable = [n for n, p in lora_linear.named_parameters() if p.requires_grad]
        assert set(trainable) == {"lora_A", "lora_B"}

    def test_original_weights_frozen(self, lora_linear):
        frozen = [
            n for n, p in lora_linear.original_linear.named_parameters()
            if not p.requires_grad
        ]
        assert "weight" in frozen

    def test_scale_computed_correctly(self):
        import torch.nn as nn
        from ncaa_models.lora import LoRALinear
        base = nn.Linear(4, 4)
        lora = LoRALinear(base, rank=4, alpha=2.0)
        assert lora.scale == pytest.approx(2.0 / 4)

    def test_forward_changes_with_nonzero_B(self):
        """When B is non-zero the output differs from the original."""
        import torch
        import torch.nn as nn
        from ncaa_models.lora import LoRALinear
        base = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(base, rank=2, alpha=1.0)

        # Manually set B to a non-zero value
        with torch.no_grad():
            lora.lora_B.fill_(1.0)

        x = torch.randn(5, 8)
        with torch.no_grad():
            out_lora = lora(x)
            out_base = base(x)
        assert not torch.allclose(out_lora, out_base)

    def test_gradient_flows_through_lora(self, lora_linear):
        """Backward pass computes gradients for lora_A and lora_B."""
        import torch
        x = torch.randn(4, 8)
        out = lora_linear(x).sum()
        out.backward()
        assert lora_linear.lora_A.grad is not None
        assert lora_linear.lora_B.grad is not None

    def test_rank_and_shape(self):
        import torch.nn as nn
        from ncaa_models.lora import LoRALinear
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=3, alpha=1.0)
        assert lora.lora_A.shape == (3, 16)
        assert lora.lora_B.shape == (8, 3)

    def test_invalid_rank_raises(self):
        import torch.nn as nn
        from ncaa_models.lora import LoRALinear
        base = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="rank"):
            LoRALinear(base, rank=0)

    def test_invalid_alpha_raises(self):
        import torch.nn as nn
        from ncaa_models.lora import LoRALinear
        base = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="alpha"):
            LoRALinear(base, rank=2, alpha=-1.0)


# ---------------------------------------------------------------------------
# TestApplyLoRA
# ---------------------------------------------------------------------------

class TestApplyLoRA:

    def test_target_layers_replaced(self, small_model):
        from ncaa_models.lora import apply_lora_to_model, LoRALinear, DEFAULT_TARGET_MODULES
        replaced = apply_lora_to_model(small_model, rank=4)
        for name in DEFAULT_TARGET_MODULES:
            # Navigate to the module
            parts = name.split(".")
            mod = small_model
            for part in parts:
                try:
                    mod = mod[int(part)]
                except (TypeError, ValueError):
                    mod = getattr(mod, part)
            assert isinstance(mod, LoRALinear), f"{name} not replaced with LoRALinear"

    def test_returns_replaced_dict(self, small_model):
        from ncaa_models.lora import apply_lora_to_model, DEFAULT_TARGET_MODULES
        replaced = apply_lora_to_model(small_model, rank=4)
        assert isinstance(replaced, dict)
        assert len(replaced) == len(DEFAULT_TARGET_MODULES)

    def test_only_lora_trainable_after_apply(self, small_model):
        from ncaa_models.lora import apply_lora_to_model, count_trainable_params
        apply_lora_to_model(small_model, rank=4)
        trainable = count_trainable_params(small_model)
        # Only lora_A and lora_B parameters should be trainable
        from ncaa_models.lora import count_lora_params
        assert trainable == count_lora_params(small_model)

    def test_trainable_params_small_fraction(self, small_model):
        """LoRA params << full model params (spirit of <1% for larger models)."""
        from ncaa_models.lora import apply_lora_to_model, count_trainable_params, count_total_params
        apply_lora_to_model(small_model, rank=4)
        trainable = count_trainable_params(small_model)
        total = count_total_params(small_model)
        pct = trainable / total
        # For production-scale models (hidden_dim≥512) rank-4 LoRA is <1%.
        # This small test model (hidden_dim=16) gives a higher fraction;
        # the key invariant is that LoRA adds far fewer params than full retraining.
        assert pct < 0.50, (
            f"LoRA trainable fraction {pct:.1%} seems too high for any model size"
        )

    def test_trainable_below_1pct_large_model(self):
        """rank-4 LoRA on matchup_net.2 only is <1% for a large backbone."""
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.lora import (
            apply_lora_to_model,
            count_trainable_params,
            count_total_params,
        )
        # Use large hidden_dim so base model has >> 1M params >> LoRA params
        model = NCAAPredictor(feature_dim=28, hidden_dim=256, dropout=0.0)
        apply_lora_to_model(model, rank=4, target_modules={"matchup_net.2"})
        trainable = count_trainable_params(model)
        total = count_total_params(model)
        assert trainable / total < 0.01, (
            f"LoRA on matchup_net.2 should be <1% for hidden_dim=256: "
            f"{trainable}/{total}={trainable/total:.2%}"
        )

    def test_custom_target_modules(self, small_model):
        from ncaa_models.lora import apply_lora_to_model, LoRALinear
        replaced = apply_lora_to_model(small_model, rank=2, target_modules={"matchup_net.2"})
        assert "matchup_net.2" in replaced
        assert len(replaced) == 1

    def test_model_output_unchanged_at_init(self, small_model):
        """B=0 at init → model output unchanged after apply_lora."""
        import torch
        from ncaa_models.lora import apply_lora_to_model

        x1 = torch.randn(4, 10)
        x2 = torch.randn(4, 10)

        small_model.eval()
        with torch.no_grad():
            before = small_model(x1, x2).clone()

        apply_lora_to_model(small_model, rank=4)

        small_model.eval()
        with torch.no_grad():
            after = small_model(x1, x2)

        torch.testing.assert_close(before, after)


# ---------------------------------------------------------------------------
# TestRemoveLoRA
# ---------------------------------------------------------------------------

class TestRemoveLoRA:

    def test_remove_restores_predictions(self):
        """remove_lora (fuse=False) restores exact base model predictions."""
        import torch
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.lora import apply_lora_to_model, remove_lora, LoRALinear

        model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
        model.eval()

        x1 = torch.randn(6, 10)
        x2 = torch.randn(6, 10)

        with torch.no_grad():
            base_out = model(x1, x2).clone()

        apply_lora_to_model(model, rank=4)

        # Perturb lora_B so adapter has effect
        for mod in model.modules():
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    mod.lora_B.fill_(0.1)

        remove_lora(model, fuse=False)
        model.eval()

        with torch.no_grad():
            after = model(x1, x2)

        torch.testing.assert_close(base_out, after)

    def test_remove_no_lora_layers_remain(self):
        import torch.nn as nn
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.lora import apply_lora_to_model, remove_lora, LoRALinear

        model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
        apply_lora_to_model(model, rank=4)
        remove_lora(model, fuse=False)

        for mod in model.modules():
            assert not isinstance(mod, LoRALinear)

    def test_fuse_changes_weights(self):
        """remove_lora(fuse=True) changes the Linear weight (non-trivially)."""
        import torch
        import torch.nn as nn
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.lora import apply_lora_to_model, remove_lora, LoRALinear

        model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)

        # Record original weight of matchup_net.2
        orig_w = model.matchup_net[2].weight.data.clone()

        apply_lora_to_model(model, rank=4, target_modules={"matchup_net.2"})

        # Set B non-zero so fusion changes the weight
        for mod in model.modules():
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    mod.lora_B.fill_(1.0)

        remove_lora(model, fuse=True)

        fused_w = model.matchup_net[2].weight.data
        assert not torch.allclose(orig_w, fused_w), "Fuse should change the weight"


# ---------------------------------------------------------------------------
# TestStateDictHelpers
# ---------------------------------------------------------------------------

class TestStateDictHelpers:

    def test_get_lora_state_dict_keys(self):
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.lora import apply_lora_to_model, get_lora_state_dict, DEFAULT_TARGET_MODULES

        model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
        apply_lora_to_model(model, rank=4)
        state = get_lora_state_dict(model)

        for name in DEFAULT_TARGET_MODULES:
            assert f"{name}.lora_A" in state
            assert f"{name}.lora_B" in state

    def test_load_lora_state_dict_roundtrip(self):
        """Save then reload → identical tensors."""
        import torch
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.lora import (
            apply_lora_to_model,
            get_lora_state_dict,
            load_lora_state_dict,
            zero_lora_weights,
            LoRALinear,
        )

        model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
        apply_lora_to_model(model, rank=4)

        # Randomise B
        for mod in model.modules():
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    mod.lora_B.normal_()

        saved = get_lora_state_dict(model)

        # Zero out then reload
        zero_lora_weights(model)
        load_lora_state_dict(model, saved)

        reloaded = get_lora_state_dict(model)
        for key in saved:
            torch.testing.assert_close(saved[key], reloaded[key])

    def test_zero_lora_restores_base(self):
        """zero_lora_weights → model output = base model output."""
        import torch
        from ncaa_models.neural import NCAAPredictor
        from ncaa_models.lora import apply_lora_to_model, zero_lora_weights, LoRALinear

        model = NCAAPredictor(feature_dim=10, hidden_dim=16, dropout=0.0)
        model.eval()

        x1 = torch.randn(5, 10)
        x2 = torch.randn(5, 10)

        with torch.no_grad():
            base_out = model(x1, x2).clone()

        apply_lora_to_model(model, rank=4)

        # Perturb B so adapter has effect
        for mod in model.modules():
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    mod.lora_B.normal_()

        # Now zero B → should revert to base
        zero_lora_weights(model)
        model.eval()
        with torch.no_grad():
            zeroed_out = model(x1, x2)

        torch.testing.assert_close(base_out, zeroed_out)
