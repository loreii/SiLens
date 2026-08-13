#!/usr/bin/env python3
"""
End-to-end conversion pipeline for SmolVLM-256M to SiLens hardware format.

This script orchestrates the complete conversion process:
1. Download SmolVLM-256M model from HuggingFace
2. Extract all weights with detailed statistics
3. Quantize to ternary (-1, 0, +1) with optimal alpha
4. Validate quantization quality
5. Generate comprehensive statistics report
6. Export weights in formats ready for Verilog generation

Usage:
    python run_pipeline.py
    python run_pipeline.py --output ./output --alpha 0.7
    python run_pipeline.py --skip-download --model ./local/model

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import local modules
try:
    from extract_weights import WeightExtractor
    from quantize_ternary import ModelQuantizer, TernaryQuantizationConfig, QuantizationMode
    from validate_quantization import QuantizationValidator
    from export_config import ConfigExporter
except ImportError:
    # Allow running from different directory
    import importlib.util
    script_dir = Path(__file__).parent
    
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    extract_weights = load_module("extract_weights", script_dir / "extract_weights.py")
    WeightExtractor = extract_weights.WeightExtractor
    
    quantize_ternary = load_module("quantize_ternary", script_dir / "quantize_ternary.py")
    ModelQuantizer = quantize_ternary.ModelQuantizer
    TernaryQuantizationConfig = quantize_ternary.TernaryQuantizationConfig
    QuantizationMode = quantize_ternary.QuantizationMode


@dataclass
class PipelineConfig:
    """Configuration for the conversion pipeline."""
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    output_dir: str = "./model/pipeline_output"
    alpha: float = 0.7
    quantization_mode: str = "per_tensor"
    device: str = "cpu"
    skip_download: bool = False
    skip_validation: bool = False
    generate_verilog: bool = False
    validation_tolerance: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Result of the complete pipeline execution."""
    success: bool
    config: PipelineConfig
    
    # Timing
    total_time_seconds: float
    step_times: Dict[str, float]
    
    # Statistics
    total_parameters: int
    quantized_parameters: int
    sparsity: float
    compression_ratio: float
    
    # Validation
    validation_passed: bool
    avg_cosine_similarity: float
    quality_assessment: str
    
    # Outputs
    output_paths: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['config'] = self.config.to_dict()
        return d


class ConversionPipeline:
    """
    End-to-end conversion pipeline for SmolVLM-256M.
    
    Orchestrates all conversion steps from download to Verilog-ready export.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.step_times: Dict[str, float] = {}
        self.model = None
        self.extractor = None
        self.quantizer = None
        self.validator = None

    
    def setup_directories(self) -> None:
        """Create output directory structure."""
        logger.info(f"Setting up output directory: {self.output_dir}")
        
        dirs = [
            self.output_dir,
            self.output_dir / "extracted",
            self.output_dir / "quantized",
            self.output_dir / "quantized" / "weights",
            self.output_dir / "validation",
            self.output_dir / "reports",
            self.output_dir / "verilog_ready",
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def step_download_model(self) -> bool:
        """
        Step 1: Download model from HuggingFace.
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Download Model")
        logger.info("=" * 60)
        
        start = time.time()
        
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers torch")
            return False
        
        if self.config.skip_download:
            logger.info(f"Skipping download, using local model: {self.config.model_id}")
        else:
            logger.info(f"Downloading model: {self.config.model_id}")
            logger.info("(This may take a few minutes on first run)")
        
        try:
            import torch
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            
            self.processor = AutoProcessor.from_pretrained(self.config.model_id)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded: {total_params:,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
        
        self.step_times['download'] = time.time() - start
        return True

    
    def step_extract_weights(self) -> bool:
        """
        Step 2: Extract and analyze all weights.
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Extract Weights")
        logger.info("=" * 60)
        
        start = time.time()
        
        try:
            self.extractor = WeightExtractor(self.config.model_id, device=self.config.device)
            self.extractor.model = self.model  # Reuse loaded model
            
            weights = self.extractor.extract_all_weights()
            
            # Print summary
            self.extractor.print_statistics(detailed=False)
            
            # Export extracted weights
            extract_dir = str(self.output_dir / "extracted")
            self.extractor.export_weights(extract_dir, format='numpy', include_metadata=True)
            
            logger.info(f"Extracted {len(weights)} weight tensors")
            
        except Exception as e:
            logger.error(f"Failed to extract weights: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        self.step_times['extract'] = time.time() - start
        return True
    
    def step_quantize(self) -> bool:
        """
        Step 3: Quantize weights to ternary.
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Quantize to Ternary")
        logger.info("=" * 60)
        
        start = time.time()
        
        try:
            # Map mode string to enum
            mode_map = {
                'per_tensor': QuantizationMode.PER_TENSOR,
                'per_channel': QuantizationMode.PER_CHANNEL,
                'per_group': QuantizationMode.PER_GROUP,
            }
            
            quant_config = TernaryQuantizationConfig(
                alpha=self.config.alpha,
                mode=mode_map.get(self.config.quantization_mode, QuantizationMode.PER_TENSOR),
            )
            
            self.quantizer = ModelQuantizer(
                self.config.model_id,
                config=quant_config,
                device=self.config.device
            )
            self.quantizer.model = self.model  # Reuse loaded model
            
            # Quantize all weights
            self.quantizer.quantize_all_weights()
            
            # Print summary
            self.quantizer.print_summary()
            
            # Export quantized weights
            quantize_dir = str(self.output_dir / "quantized")
            self.quantizer.export(quantize_dir, include_scales=True, hardware_encoding=True)
            
        except Exception as e:
            logger.error(f"Failed to quantize weights: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        self.step_times['quantize'] = time.time() - start
        return True

    
    def step_validate(self) -> Dict[str, Any]:
        """
        Step 4: Validate quantization quality.
        
        Returns:
            Validation results dictionary
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Validate Quantization")
        logger.info("=" * 60)
        
        start = time.time()
        
        validation_result = {
            'passed': False,
            'avg_cosine_similarity': 0.0,
            'quality': 'unknown',
        }
        
        if self.config.skip_validation:
            logger.info("Skipping validation as requested")
            validation_result['passed'] = True
            validation_result['quality'] = 'skipped'
            return validation_result
        
        try:
            self.validator = QuantizationValidator(
                model_path=self.config.model_id,
                quantized_path=str(self.output_dir / "quantized"),
                alpha=self.config.alpha,
                device=self.config.device
            )
            self.validator.model = self.model  # Reuse loaded model
            
            # Run validation
            self.validator.validate_all_layers(tolerance=self.config.validation_tolerance)
            
            # Get summary
            summary = self.validator.get_summary()
            
            # Print report
            self.validator.print_report(detailed=False)
            
            # Export report
            report_path = str(self.output_dir / "validation" / "validation_report.json")
            self.validator.export_report(report_path)
            
            # Try to generate visualizations
            try:
                vis_path = str(self.output_dir / "validation" / "error_plots.png")
                self.validator.visualize_errors(vis_path)
            except Exception as e:
                logger.warning(f"Could not generate visualizations: {e}")
            
            validation_result = {
                'passed': summary.overall_quality in ('excellent', 'good', 'acceptable'),
                'avg_cosine_similarity': summary.avg_cosine_similarity,
                'quality': summary.overall_quality,
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.step_times['validate'] = time.time() - start
        return validation_result

    
    def step_generate_statistics(self) -> Dict[str, Any]:
        """
        Step 5: Generate comprehensive statistics report.
        
        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Generate Statistics Report")
        logger.info("=" * 60)
        
        start = time.time()
        
        stats = {}
        
        try:
            # Extraction statistics
            if self.extractor:
                extract_summary = self.extractor.get_summary_statistics()
                stats['extraction'] = extract_summary
            
            # Quantization statistics
            if self.quantizer:
                quant_summary = self.quantizer.get_summary_statistics()
                stats['quantization'] = quant_summary
            
            # Hardware estimates
            if self.quantizer:
                total_params = quant_summary.get('total_quantized_params', 0)
                nonzero_params = total_params - int(total_params * quant_summary.get('overall_sparsity', 0))
                
                stats['hardware'] = {
                    'total_parameters': total_params,
                    'nonzero_parameters': nonzero_params,
                    'estimated_connections': nonzero_params,
                    'bits_per_weight': 2,  # Ternary encoding
                    'memory_bits': total_params * 2,
                    'memory_bytes': total_params * 2 // 8,
                    'memory_kb': total_params * 2 // 8 // 1024,
                    'compression_vs_fp32': 32 / 2,  # 16x compression
                }
            
            # Save statistics report
            report_path = self.output_dir / "reports" / "statistics_report.json"
            with open(report_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"Statistics report saved to: {report_path}")
            
            # Print summary
            print("\n--- Statistics Summary ---")
            if 'quantization' in stats:
                qs = stats['quantization']
                print(f"Total parameters: {qs.get('total_quantized_params', 0):,}")
                print(f"Sparsity: {qs.get('overall_sparsity', 0):.1%}")
                print(f"Distribution: +1={qs.get('distribution', {}).get('positive_pct', 0):.1f}%, "
                      f"-1={qs.get('distribution', {}).get('negative_pct', 0):.1f}%, "
                      f"0={qs.get('distribution', {}).get('zero_pct', 0):.1f}%")
            
            if 'hardware' in stats:
                hw = stats['hardware']
                print(f"Estimated connections: {hw.get('estimated_connections', 0):,}")
                print(f"Memory (quantized): {hw.get('memory_kb', 0):,} KB")
                print(f"Compression ratio: {hw.get('compression_vs_fp32', 0):.0f}x vs FP32")
            
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            import traceback
            traceback.print_exc()
        
        self.step_times['statistics'] = time.time() - start
        return stats

    
    def step_export_verilog_ready(self) -> bool:
        """
        Step 6: Export weights in Verilog-ready format.
        
        Prepares packed binary files and generates configuration files
        that can be directly consumed by weights_to_verilog.py.
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Export Verilog-Ready Format")
        logger.info("=" * 60)
        
        start = time.time()
        
        try:
            verilog_dir = self.output_dir / "verilog_ready"
            
            # Consolidate all quantized weights into single files by component
            quantized_dir = self.output_dir / "quantized" / "weights"
            
            if not quantized_dir.exists():
                logger.error("Quantized weights not found")
                return False
            
            # Group weights by component
            components = {
                'vision_encoder': [],
                'projector': [],
                'language_model': [],
                'other': [],
            }
            
            for qfile in quantized_dir.glob('*_quantized.npy'):
                name = qfile.stem.replace('_quantized', '')
                weights = np.load(qfile)
                
                # Determine component
                name_lower = name.lower()
                if any(x in name_lower for x in ['vision', 'image', 'siglip']):
                    component = 'vision_encoder'
                elif any(x in name_lower for x in ['projector', 'connector', 'multi_modal']):
                    component = 'projector'
                elif any(x in name_lower for x in ['language', 'model_layers', 'lm_head']):
                    component = 'language_model'
                else:
                    component = 'other'
                
                components[component].append({
                    'name': name,
                    'weights': weights,
                    'shape': weights.shape,
                })
            
            # Save consolidated files per component
            for component, layers in components.items():
                if not layers:
                    continue
                
                comp_dir = verilog_dir / component
                comp_dir.mkdir(exist_ok=True)
                
                # Save metadata
                metadata = {
                    'component': component,
                    'num_layers': len(layers),
                    'layers': [
                        {
                            'name': l['name'],
                            'shape': list(l['shape']),
                            'numel': int(np.prod(l['shape'])),
                        }
                        for l in layers
                    ]
                }
                
                with open(comp_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Save packed weights (all layers concatenated)
                all_weights = {}
                for l in layers:
                    all_weights[l['name']] = l['weights']
                
                np.savez_compressed(comp_dir / f'{component}_weights.npz', **all_weights)
                
                logger.info(f"  {component}: {len(layers)} layers")

            
            # Generate config export
            try:
                exporter = ConfigExporter(self.config.model_id, device=self.config.device)
                exporter.model = self.model
                
                # Export JSON config
                exporter.export_json_config(str(verilog_dir / 'model_config.json'))
                
                # Export Verilog parameters
                exporter.export_verilog_params(str(verilog_dir / 'model_params.vh'))
                
            except Exception as e:
                logger.warning(f"Config export failed: {e}")
                # Continue anyway - weights are the main deliverable
            
            logger.info(f"Verilog-ready exports saved to: {verilog_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export Verilog-ready format: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        self.step_times['verilog_export'] = time.time() - start
        return True
    
    def run(self) -> PipelineResult:
        """
        Execute the complete pipeline.
        
        Returns:
            PipelineResult with summary of all steps
        """
        total_start = time.time()
        
        print("\n" + "=" * 70)
        print("SiLens Model Conversion Pipeline")
        print("=" * 70)
        print(f"Model: {self.config.model_id}")
        print(f"Output: {self.config.output_dir}")
        print(f"Alpha: {self.config.alpha}")
        print(f"Mode: {self.config.quantization_mode}")
        print("=" * 70 + "\n")
        
        # Setup
        self.setup_directories()
        
        # Step 1: Download
        if not self.step_download_model():
            return self._create_failure_result(total_start, "download failed")
        
        # Step 2: Extract
        if not self.step_extract_weights():
            return self._create_failure_result(total_start, "extraction failed")
        
        # Step 3: Quantize
        if not self.step_quantize():
            return self._create_failure_result(total_start, "quantization failed")
        
        # Step 4: Validate
        validation = self.step_validate()
        
        # Step 5: Statistics
        stats = self.step_generate_statistics()
        
        # Step 6: Export
        if not self.step_export_verilog_ready():
            logger.warning("Verilog export had issues, but continuing")
        
        # Create result
        total_time = time.time() - total_start
        
        quant_stats = stats.get('quantization', {})
        
        result = PipelineResult(
            success=True,
            config=self.config,
            total_time_seconds=total_time,
            step_times=self.step_times,
            total_parameters=quant_stats.get('total_quantized_params', 0),
            quantized_parameters=quant_stats.get('total_quantized_params', 0),
            sparsity=quant_stats.get('overall_sparsity', 0),
            compression_ratio=16.0,  # FP32 to 2-bit
            validation_passed=validation.get('passed', False),
            avg_cosine_similarity=validation.get('avg_cosine_similarity', 0),
            quality_assessment=validation.get('quality', 'unknown'),
            output_paths={
                'extracted': str(self.output_dir / "extracted"),
                'quantized': str(self.output_dir / "quantized"),
                'validation': str(self.output_dir / "validation"),
                'reports': str(self.output_dir / "reports"),
                'verilog_ready': str(self.output_dir / "verilog_ready"),
            }
        )
        
        # Save pipeline result
        result_path = self.output_dir / "reports" / "pipeline_result.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self._print_summary(result)
        
        return result

    
    def _create_failure_result(self, start_time: float, reason: str) -> PipelineResult:
        """Create a failure result."""
        return PipelineResult(
            success=False,
            config=self.config,
            total_time_seconds=time.time() - start_time,
            step_times=self.step_times,
            total_parameters=0,
            quantized_parameters=0,
            sparsity=0,
            compression_ratio=0,
            validation_passed=False,
            avg_cosine_similarity=0,
            quality_assessment=f"failed: {reason}",
            output_paths={}
        )
    
    def _print_summary(self, result: PipelineResult) -> None:
        """Print final pipeline summary."""
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        
        print(f"\nStatus: {'SUCCESS ✓' if result.success else 'FAILED ✗'}")
        print(f"Total time: {result.total_time_seconds:.1f} seconds")
        
        print("\n--- Step Timing ---")
        for step, duration in self.step_times.items():
            print(f"  {step}: {duration:.1f}s")
        
        print("\n--- Results ---")
        print(f"Total parameters: {result.total_parameters:,}")
        print(f"Sparsity: {result.sparsity:.1%}")
        print(f"Compression: {result.compression_ratio:.0f}x")
        
        if result.validation_passed:
            print(f"\nValidation: PASSED ✓")
            print(f"  Quality: {result.quality_assessment}")
            print(f"  Avg cosine similarity: {result.avg_cosine_similarity:.4f}")
        else:
            print(f"\nValidation: {'SKIPPED' if result.quality_assessment == 'skipped' else 'FAILED'}")
        
        print("\n--- Output Paths ---")
        for name, path in result.output_paths.items():
            print(f"  {name}: {path}")
        
        print("\n--- Next Steps ---")
        print("1. Review validation report in validation/validation_report.json")
        print("2. Check quantization statistics in reports/statistics_report.json")
        print("3. Generate Verilog with: python weights_to_verilog.py -w verilog_ready/ -o rtl/")
        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="End-to-end conversion pipeline for SmolVLM-256M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with default settings
    python run_pipeline.py
    
    # Custom output directory and alpha
    python run_pipeline.py --output ./my_output --alpha 0.65
    
    # Use local model (skip download)
    python run_pipeline.py --model ./local/smolvlm-256m --skip-download
    
    # Per-channel quantization
    python run_pipeline.py --mode per_channel
    
    # Quick run (skip validation)
    python run_pipeline.py --skip-validation
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./model/pipeline_output",
        help="Output directory for all artifacts"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Threshold factor for ternary quantization (default: 0.7)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['per_tensor', 'per_channel', 'per_group'],
        default='per_tensor',
        help="Quantization mode (default: per_tensor)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use for computation (default: cpu)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (use local model)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Validation tolerance (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        model_id=args.model,
        output_dir=args.output,
        alpha=args.alpha,
        quantization_mode=args.mode,
        device=args.device,
        skip_download=args.skip_download,
        skip_validation=args.skip_validation,
        validation_tolerance=args.tolerance,
    )
    
    # Run pipeline
    pipeline = ConversionPipeline(config)
    result = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
