"""
LoRA adapter module for FieldSense AI v3.0
Enables fast team-specific calibration (<42s) with minimal parameters
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import json

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft library not available. Install with: pip install peft")

from .pytorch_backbone import BackboneConvNet


class BackboneWithAdapter(nn.Module):
    """
    Backbone model with LoRA adapter for team-specific calibration.
    """

    def __init__(
        self,
        backbone: Optional[BackboneConvNet] = None,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1
    ):
        """
        Initialize backbone with LoRA adapter.

        Args:
            backbone: Pre-trained backbone model (creates new if None)
            lora_rank: LoRA rank (default: 4 for ~0.8MB)
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout rate
        """
        super().__init__()

        self.backbone = backbone if backbone is not None else BackboneConvNet()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.peft_model = None
        self.use_peft = PEFT_AVAILABLE

        # Initialize LoRA if PEFT is available
        if self.use_peft:
            self._init_lora()
        else:
            print("PEFT not available, using backbone without LoRA")

    def _init_lora(self):
        """Initialize LoRA configuration and wrap model."""
        # Configure LoRA for Conv2d layers
        # Note: PEFT primarily supports Linear layers, so we'll use a custom approach
        # for Conv2d or convert to a compatible architecture

        # For this implementation, we'll add LoRA-like adapters manually
        # to the conv layers to maintain compatibility
        self._add_conv_lora_layers()

    def _add_conv_lora_layers(self):
        """Add LoRA-like adaptation layers to convolutions."""
        # Add LoRA to conv1
        self.conv1_lora_A = nn.Parameter(
            torch.randn(self.lora_rank, 8, 3, 3) * 0.01
        )
        self.conv1_lora_B = nn.Parameter(
            torch.zeros(16, self.lora_rank, 1, 1)
        )

        # Add LoRA to conv2
        self.conv2_lora_A = nn.Parameter(
            torch.randn(self.lora_rank, 16, 1, 1) * 0.01
        )
        self.conv2_lora_B = nn.Parameter(
            torch.zeros(1, self.lora_rank, 1, 1)
        )

        # Scaling factor
        self.lora_scale = self.lora_alpha / self.lora_rank

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (batch, 8, 105, 68)

        Returns:
            xT grid of shape (batch, 105, 68)
        """
        if not self.use_peft or not hasattr(self, 'conv1_lora_A'):
            # Use backbone without LoRA
            return self.backbone(x)

        # Conv1 with LoRA
        # Original conv1
        out = self.backbone.conv1(x)

        # Add LoRA contribution
        # LoRA: W' = W + (B @ A) * scale
        lora_weight = torch.einsum('oihw,irhw->orhw',
                                   self.conv1_lora_B,
                                   self.conv1_lora_A)
        lora_weight = lora_weight.sum(dim=1, keepdim=True).squeeze(1)
        lora_out = torch.nn.functional.conv2d(x, lora_weight, padding=1)
        out = out + lora_out * self.lora_scale

        # ReLU
        out = self.backbone.relu1(out)

        # Conv2 with LoRA
        # Original conv2
        out2 = self.backbone.conv2(out)

        # Add LoRA contribution
        lora_weight2 = torch.einsum('oihw,irhw->orhw',
                                    self.conv2_lora_B,
                                    self.conv2_lora_A)
        lora_weight2 = lora_weight2.sum(dim=1, keepdim=True).squeeze(1)
        lora_out2 = torch.nn.functional.conv2d(out, lora_weight2)
        out2 = out2 + lora_out2 * self.lora_scale

        # Sigmoid
        out2 = self.backbone.sigmoid(out2)

        # Remove channel dimension
        out2 = out2.squeeze(1)

        return out2

    def save_adapter(self, path: str):
        """
        Save only LoRA adapter weights (lightweight).

        Args:
            path: Path to save adapter weights
        """
        if not hasattr(self, 'conv1_lora_A'):
            print("No LoRA adapter to save")
            return

        adapter_state = {
            'conv1_lora_A': self.conv1_lora_A.data.cpu(),
            'conv1_lora_B': self.conv1_lora_B.data.cpu(),
            'conv2_lora_A': self.conv2_lora_A.data.cpu(),
            'conv2_lora_B': self.conv2_lora_B.data.cpu(),
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_scale': self.lora_scale
        }

        torch.save(adapter_state, path)

        # Print size
        import os
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Adapter saved to {path} ({size_mb:.2f} MB)")

    def load_adapter(self, path: str):
        """
        Load LoRA adapter weights.

        Args:
            path: Path to adapter weights
        """
        if not Path(path).exists():
            print(f"Adapter not found: {path}")
            return False

        adapter_state = torch.load(path, map_location='cpu')

        # Initialize LoRA layers if not present
        if not hasattr(self, 'conv1_lora_A'):
            self._add_conv_lora_layers()

        # Load weights
        self.conv1_lora_A.data = adapter_state['conv1_lora_A']
        self.conv1_lora_B.data = adapter_state['conv1_lora_B']
        self.conv2_lora_A.data = adapter_state['conv2_lora_A']
        self.conv2_lora_B.data = adapter_state['conv2_lora_B']
        self.lora_scale = adapter_state.get('lora_scale', self.lora_alpha / self.lora_rank)

        print(f"Adapter loaded from {path}")
        return True

    def get_trainable_params(self):
        """Get number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'trainable_pct': 100 * trainable / total if total > 0 else 0
        }


def create_adapter_model(
    onnx_path: Optional[str] = None,
    lora_rank: int = 4
) -> BackboneWithAdapter:
    """
    Create backbone model with LoRA adapter.

    Args:
        onnx_path: Optional path to ONNX model to load weights from
        lora_rank: LoRA rank (default: 4)

    Returns:
        BackboneWithAdapter model
    """
    # Create backbone
    backbone = BackboneConvNet()

    # Load ONNX weights if provided
    if onnx_path and Path(onnx_path).exists():
        backbone.load_from_onnx_weights(onnx_path)

    # Wrap with adapter
    model = BackboneWithAdapter(
        backbone=backbone,
        lora_rank=lora_rank,
        lora_alpha=8,
        lora_dropout=0.1
    )

    # Print parameter info
    params = model.get_trainable_params()
    print(f"Model parameters: {params['total']:,} total, "
          f"{params['trainable']:,} trainable ({params['trainable_pct']:.2f}%)")

    return model
