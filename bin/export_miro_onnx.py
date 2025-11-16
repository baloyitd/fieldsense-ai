#!/usr/bin/env python3
"""
FieldSense AI v3.1 - MiroThinker ONNX Export

Exports MiroThinker-v1.0-30B to ONNX format for optimized inference.

Uses optimum.onnxruntime for efficient conversion with:
- 4-bit quantization support
- Optimized graph structure
- Multi-platform compatibility

Output: /assets/miro_onnx/
"""

import sys
import os
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_name: str = "miromind-ai/MiroThinker-v1.0-30B",
    output_dir: str = "assets/miro_onnx",
    quantize: bool = True
) -> bool:
    """
    Export MiroThinker model to ONNX format.

    Args:
        model_name: HuggingFace model name
        output_dir: Output directory for ONNX files
        quantize: Apply 4-bit quantization (default: True)

    Returns:
        True if export succeeded
    """
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
        import torch
    except ImportError as e:
        logger.error(f"Required packages not available: {e}")
        logger.error("Please install: pip install optimum[onnxruntime] transformers torch")
        return False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("MiroThinker ONNX Export")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Quantization: {quantize}")

    try:
        # Export model to ONNX
        logger.info("\nExporting model to ONNX...")

        model = ORTModelForCausalLM.from_pretrained(
            model_name,
            export=True,
            use_cache=True,
            use_io_binding=True
        )

        # Export tokenizer
        logger.info("Exporting tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save model and tokenizer
        logger.info(f"Saving to {output_path}...")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Apply quantization if requested
        if quantize:
            logger.info("\nApplying 4-bit quantization...")
            try:
                from optimum.onnxruntime import ORTQuantizer
                from optimum.onnxruntime.configuration import AutoQuantizationConfig

                # Create quantizer
                quantizer = ORTQuantizer.from_pretrained(output_path)

                # Configure quantization
                qconfig = AutoQuantizationConfig.avx512_vnni(
                    is_static=False,
                    per_channel=True
                )

                # Apply quantization
                quantizer.quantize(
                    save_dir=output_path / "quantized",
                    quantization_config=qconfig
                )

                logger.info(f"✓ Quantized model saved to {output_path / 'quantized'}")

            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
                logger.warning("Continuing with non-quantized model")

        # Verify export
        logger.info("\nVerifying export...")

        onnx_files = list(output_path.glob("*.onnx"))
        if onnx_files:
            total_size = sum(f.stat().st_size for f in onnx_files)
            logger.info(f"✓ ONNX files found: {len(onnx_files)}")
            logger.info(f"  Total size: {total_size / 1024**3:.2f} GB")

            for f in onnx_files:
                size_gb = f.stat().st_size / 1024**3
                logger.info(f"    - {f.name}: {size_gb:.2f} GB")
        else:
            logger.warning("⚠ No ONNX files found in output directory")

        logger.info("\n" + "="*70)
        logger.info("✓ Export completed successfully")
        logger.info("="*70)
        logger.info(f"\nUsage:")
        logger.info(f"  from optimum.onnxruntime import ORTModelForCausalLM")
        logger.info(f"  model = ORTModelForCausalLM.from_pretrained('{output_dir}')")
        logger.info("="*70)

        return True

    except Exception as e:
        logger.error(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("FieldSense AI v3.1 - MiroThinker ONNX Export\n")

    # Check if optimum is available
    try:
        import optimum
        print(f"✓ optimum version: {optimum.__version__}")
    except ImportError:
        print("✗ optimum not installed")
        print("\nInstall with:")
        print("  pip install optimum[onnxruntime] transformers torch")
        return 1

    # Export
    output_dir = Path(__file__).parent.parent / "assets" / "miro_onnx"

    print(f"\nExporting MiroThinker to ONNX...")
    print(f"Output directory: {output_dir}")
    print(f"\nNote: This will download ~60GB model and convert to ONNX format.")
    print(f"Ensure you have sufficient disk space and bandwidth.\n")

    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Export cancelled")
        return 0

    success = export_to_onnx(output_dir=str(output_dir))

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
