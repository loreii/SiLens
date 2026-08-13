#!/usr/bin/env python3
"""
Convert quantized weights to Verilog modules for SiLens.

This script generates synthesizable Verilog modules where neural network
weights are implemented as hardwired connections:
  - Weight +1: assign out = in;          (direct connection)
  - Weight -1: assign out = -in;         (two's complement negation)
  - Weight  0: assign out = 0;           (tied to zero)

The generated modules use parameterized bit widths and support hierarchical
organization by layer type (vision encoder, projector, language model).

Usage:
    python weights_to_verilog.py --weights model/weights/quantized --output rtl/
    python weights_to_verilog.py --weights model/weights/quantized --output rtl/ --layer-filter "vision"
    
Example:
    # Generate all layer modules
    python weights_to_verilog.py -w model/weights/quantized -o rtl/
    
    # Generate only language model layers
    python weights_to_verilog.py -w model/weights/quantized -o rtl/ -f "language"

Output Structure:
    rtl/
    ├── vision_encoder/
    │   └── weights/
    │       ├── vision_block0_attn_qkv.v
    │       ├── vision_block0_attn_proj.v
    │       └── ...
    ├── projector/
    │   └── weights/
    │       ├── projector_linear1.v
    │       └── ...
    └── language_model/
        └── weights/
            ├── llm_block0_attn_qkv.v
            └── ...

License: Apache 2.0
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Constants
WEIGHT_POSITIVE = 1
WEIGHT_NEGATIVE = -1
WEIGHT_ZERO = 0

# Default bit widths
DEFAULT_ACT_WIDTH = 8
DEFAULT_ACC_WIDTH = 32


@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    name: str
    input_dim: int
    output_dim: int
    weights: np.ndarray
    component: str  # vision_encoder, projector, language_model


@dataclass
class VerilogConfig:
    """Configuration for Verilog generation."""
    act_width: int = DEFAULT_ACT_WIDTH
    acc_width: int = DEFAULT_ACC_WIDTH
    generate_testbench: bool = True
    include_comments: bool = True
    optimize_zeros: bool = True


def load_weights(weights_path: str) -> Dict[str, np.ndarray]:
    """
    Load quantized weights from directory.
    
    Supports multiple formats:
    - .npy files (NumPy)
    - .json files (for metadata)
    - .bin files (raw binary)
    
    Args:
        weights_path: Path to weights directory or single file
        
    Returns:
        Dictionary mapping layer names to weight arrays
    """
    weights_path = Path(weights_path)
    weights = {}
    
    if weights_path.is_file():
        # Single file
        if weights_path.suffix == '.npy':
            data = np.load(weights_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                weights = data.item()
            else:
                weights[weights_path.stem] = data
        elif weights_path.suffix == '.npz':
            data = np.load(weights_path)
            weights = {k: data[k] for k in data.files}
        else:
            raise ValueError(f"Unsupported file format: {weights_path.suffix}")
    elif weights_path.is_dir():
        # Directory with multiple files
        for f in weights_path.glob('*.npy'):
            weights[f.stem] = np.load(f)
        for f in weights_path.glob('*.npz'):
            data = np.load(f)
            for k in data.files:
                weights[k] = data[k]
    else:
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    print(f"Loaded {len(weights)} weight tensors from {weights_path}")
    return weights


def categorize_layer(name: str) -> str:
    """
    Categorize a layer by its component.
    
    Args:
        name: Layer name from model
        
    Returns:
        Component name: vision_encoder, projector, or language_model
    """
    name_lower = name.lower()
    
    if any(x in name_lower for x in ['vision', 'image', 'vit', 'siglip']):
        return 'vision_encoder'
    elif any(x in name_lower for x in ['projector', 'connector', 'multi_modal']):
        return 'projector'
    elif any(x in name_lower for x in ['language', 'llm', 'model.layers', 'lm_head']):
        return 'language_model'
    else:
        return 'other'


def sanitize_name(name: str) -> str:
    """
    Convert layer name to valid Verilog identifier.
    
    Args:
        name: Original layer name
        
    Returns:
        Sanitized name suitable for Verilog
    """
    # Replace dots and special chars with underscores
    sanitized = name.replace('.', '_').replace('-', '_')
    # Remove any remaining invalid characters
    sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
    # Ensure doesn't start with number
    if sanitized[0].isdigit():
        sanitized = 'layer_' + sanitized
    return sanitized.lower()


def quantize_to_ternary(weights: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Quantize weights to ternary values (-1, 0, +1).
    
    If weights are already quantized (values in {-1, 0, 1}), they pass through.
    Otherwise, applies threshold-based quantization.
    
    Args:
        weights: Weight array
        threshold: Threshold for quantization (relative to mean absolute value)
        
    Returns:
        Ternary weight array
    """
    unique_vals = np.unique(weights)
    
    # Check if already ternary
    if set(unique_vals).issubset({-1, 0, 1}):
        return weights.astype(np.int8)
    
    # Apply ternary quantization
    abs_mean = np.mean(np.abs(weights))
    thresh = threshold * abs_mean
    
    ternary = np.zeros_like(weights, dtype=np.int8)
    ternary[weights > thresh] = 1
    ternary[weights < -thresh] = -1
    
    return ternary


def generate_module_header(
    module_name: str,
    input_dim: int,
    output_dim: int,
    config: VerilogConfig,
    original_name: str = "",
    sparsity: float = 0.0
) -> str:
    """Generate Verilog module header with documentation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"""// =============================================================================
// SiLens Weight Module: {module_name}
// =============================================================================
// Auto-generated by weights_to_verilog.py
// Generated: {timestamp}
//
// Original layer: {original_name}
// Input dimension: {input_dim}
// Output dimension: {output_dim}
// Sparsity (zero weights): {sparsity:.1%}
//
// Weight encoding:
//   +1: Direct connection (assign out = in)
//   -1: Two's complement negation (assign out = -in)
//    0: Tied to zero (assign out = 0)
//
// License: Apache 2.0
// =============================================================================

"""
    return header


def generate_weight_module(
    module_name: str,
    weights: np.ndarray,
    config: VerilogConfig,
    original_name: str = ""
) -> str:
    """
    Generate a Verilog module for a weight matrix.
    
    The module computes: output[i] = sum(weights[i,j] * input[j]) for all j
    
    For ternary weights, this becomes:
    - Positive weights: add input
    - Negative weights: subtract input (add negation)
    - Zero weights: skip (no connection)
    
    Args:
        module_name: Name for the Verilog module
        weights: 2D weight array (output_dim x input_dim)
        config: Verilog generation configuration
        original_name: Original layer name for documentation
        
    Returns:
        Complete Verilog module as string
    """
    # Ensure weights are 2D
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    output_dim, input_dim = weights.shape
    act_width = config.act_width
    acc_width = config.acc_width
    
    # Calculate sparsity
    sparsity = np.mean(weights == 0)
    
    # Count weight types per output
    pos_counts = np.sum(weights > 0, axis=1)
    neg_counts = np.sum(weights < 0, axis=1)
    
    # Start building module
    lines = []
    
    # Header
    lines.append(generate_module_header(
        module_name, input_dim, output_dim, config, original_name, sparsity
    ))
    
    # Module declaration
    lines.append(f"module {module_name} #(")
    lines.append(f"    parameter ACT_WIDTH = {act_width},")
    lines.append(f"    parameter ACC_WIDTH = {acc_width}")
    lines.append(f")(")
    lines.append(f"    input  wire [ACT_WIDTH-1:0] in [{input_dim}-1:0],")
    lines.append(f"    output wire [ACC_WIDTH-1:0] out [{output_dim}-1:0]")
    lines.append(f");")
    lines.append("")
    
    # Generate output computations
    for out_idx in range(output_dim):
        row_weights = weights[out_idx, :]
        
        # Find positive and negative weight indices
        pos_indices = np.where(row_weights > 0)[0]
        neg_indices = np.where(row_weights < 0)[0]
        
        if config.include_comments:
            lines.append(f"    // Output[{out_idx}]: {len(pos_indices)} positive, {len(neg_indices)} negative weights")
        
        if len(pos_indices) == 0 and len(neg_indices) == 0:
            # All zeros - tie output to zero
            lines.append(f"    assign out[{out_idx}] = {acc_width}'d0;")
        elif len(pos_indices) == 1 and len(neg_indices) == 0:
            # Single positive weight - direct connection
            lines.append(f"    assign out[{out_idx}] = {{{{ACC_WIDTH-ACT_WIDTH{{in[{pos_indices[0]}][ACT_WIDTH-1]}}}}, in[{pos_indices[0]}]}};")
        elif len(pos_indices) == 0 and len(neg_indices) == 1:
            # Single negative weight - negate
            lines.append(f"    assign out[{out_idx}] = -{{{{ACC_WIDTH-ACT_WIDTH{{in[{neg_indices[0]}][ACT_WIDTH-1]}}}}, in[{neg_indices[0]}]}};")
        else:
            # Multiple weights - build sum expression
            # Use intermediate wires for clarity
            wire_name = f"sum_{out_idx}"
            
            # Build the sum expression
            terms = []
            for idx in pos_indices:
                terms.append(f"$signed(in[{idx}])")
            for idx in neg_indices:
                terms.append(f"(-$signed(in[{idx}]))")
            
            # Join terms
            if len(terms) <= 4:
                # Short expression - single line
                expr = " + ".join(terms)
                lines.append(f"    assign out[{out_idx}] = {expr};")
            else:
                # Long expression - use wire
                lines.append(f"    wire signed [ACC_WIDTH-1:0] {wire_name};")
                lines.append(f"    assign {wire_name} = ")
                # Split across lines
                for i, term in enumerate(terms):
                    if i == 0:
                        lines.append(f"        {term}")
                    elif i == len(terms) - 1:
                        lines.append(f"        + {term};")
                    else:
                        lines.append(f"        + {term}")
                lines.append(f"    assign out[{out_idx}] = {wire_name};")
        
        lines.append("")
    
    lines.append("endmodule")
    
    return "\n".join(lines)


def generate_weight_module_optimized(
    module_name: str,
    weights: np.ndarray,
    config: VerilogConfig,
    original_name: str = ""
) -> str:
    """
    Generate an optimized Verilog module using popcount for sparse weights.
    
    For highly sparse weight matrices, uses popcount-based computation:
    output = popcount(positive_mask & input) - popcount(negative_mask & input)
    
    This is more efficient when sparsity > 50%.
    
    Args:
        module_name: Name for the Verilog module
        weights: 2D weight array (output_dim x input_dim)
        config: Verilog generation configuration
        original_name: Original layer name for documentation
        
    Returns:
        Complete Verilog module as string
    """
    # Ensure weights are 2D
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    output_dim, input_dim = weights.shape
    act_width = config.act_width
    acc_width = config.acc_width
    
    # Calculate sparsity
    sparsity = np.mean(weights == 0)
    
    lines = []
    
    # Header
    lines.append(generate_module_header(
        module_name, input_dim, output_dim, config, original_name, sparsity
    ))
    
    # Module declaration
    lines.append(f"module {module_name} #(")
    lines.append(f"    parameter ACT_WIDTH = {act_width},")
    lines.append(f"    parameter ACC_WIDTH = {acc_width},")
    lines.append(f"    parameter IN_DIM = {input_dim},")
    lines.append(f"    parameter OUT_DIM = {output_dim}")
    lines.append(f")(")
    lines.append(f"    input  wire [ACT_WIDTH*IN_DIM-1:0]  in_flat,")
    lines.append(f"    output wire [ACC_WIDTH*OUT_DIM-1:0] out_flat")
    lines.append(f");")
    lines.append("")
    
    # Unflatten inputs
    lines.append("    // Unflatten input array")
    lines.append("    wire [ACT_WIDTH-1:0] in [0:IN_DIM-1];")
    lines.append("    genvar gi;")
    lines.append("    generate")
    lines.append("        for (gi = 0; gi < IN_DIM; gi = gi + 1) begin : unflatten_in")
    lines.append("            assign in[gi] = in_flat[gi*ACT_WIDTH +: ACT_WIDTH];")
    lines.append("        end")
    lines.append("    endgenerate")
    lines.append("")
    
    # Generate weight masks and computations
    lines.append("    // Output computations")
    lines.append("    wire [ACC_WIDTH-1:0] out [0:OUT_DIM-1];")
    lines.append("")
    
    for out_idx in range(output_dim):
        row_weights = weights[out_idx, :]
        pos_indices = np.where(row_weights > 0)[0].tolist()
        neg_indices = np.where(row_weights < 0)[0].tolist()
        
        if len(pos_indices) == 0 and len(neg_indices) == 0:
            lines.append(f"    assign out[{out_idx}] = {acc_width}'d0;")
        else:
            # Build computation
            terms = []
            for idx in pos_indices:
                terms.append(f"$signed(in[{idx}])")
            for idx in neg_indices:
                terms.append(f"(-$signed(in[{idx}]))")
            
            expr = " + ".join(terms) if terms else f"{acc_width}'d0"
            lines.append(f"    assign out[{out_idx}] = {expr};")
    
    lines.append("")
    
    # Flatten outputs
    lines.append("    // Flatten output array")
    lines.append("    generate")
    lines.append("        for (gi = 0; gi < OUT_DIM; gi = gi + 1) begin : flatten_out")
    lines.append("            assign out_flat[gi*ACC_WIDTH +: ACC_WIDTH] = out[gi];")
    lines.append("        end")
    lines.append("    endgenerate")
    lines.append("")
    
    lines.append("endmodule")
    
    return "\n".join(lines)


def generate_testbench(
    module_name: str,
    input_dim: int,
    output_dim: int,
    weights: np.ndarray,
    config: VerilogConfig
) -> str:
    """Generate a simple testbench for the weight module."""
    act_width = config.act_width
    acc_width = config.acc_width
    
    tb = f"""// =============================================================================
// Testbench for {module_name}
// =============================================================================
// Auto-generated by weights_to_verilog.py
// =============================================================================

`timescale 1ns/1ps

module {module_name}_tb;

    parameter ACT_WIDTH = {act_width};
    parameter ACC_WIDTH = {acc_width};
    parameter IN_DIM = {input_dim};
    parameter OUT_DIM = {output_dim};
    
    // Signals
    reg  [ACT_WIDTH-1:0] in [0:IN_DIM-1];
    wire [ACC_WIDTH-1:0] out [0:OUT_DIM-1];
    
    // DUT instantiation
    {module_name} #(
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .in(in),
        .out(out)
    );
    
    // Test procedure
    integer i;
    initial begin
        $display("Testing {module_name}");
        $display("Input dim: %d, Output dim: %d", IN_DIM, OUT_DIM);
        
        // Initialize inputs to zero
        for (i = 0; i < IN_DIM; i = i + 1) begin
            in[i] = 0;
        end
        #10;
        
        // Test with all ones
        for (i = 0; i < IN_DIM; i = i + 1) begin
            in[i] = 1;
        end
        #10;
        $display("All ones input - output[0] = %d", out[0]);
        
        // Test with identity-like pattern
        for (i = 0; i < IN_DIM; i = i + 1) begin
            in[i] = (i < OUT_DIM) ? (i + 1) : 0;
        end
        #10;
        $display("Identity pattern - output[0] = %d", out[0]);
        
        // Random test
        for (i = 0; i < IN_DIM; i = i + 1) begin
            in[i] = $random % (1 << ACT_WIDTH);
        end
        #10;
        $display("Random input - output[0] = %d", out[0]);
        
        $display("Test complete");
        $finish;
    end

endmodule
"""
    return tb


def generate_layer_include(
    layers: List[LayerConfig],
    component: str,
    output_dir: Path
) -> str:
    """Generate an include file listing all layer modules for a component."""
    include_lines = [
        f"// =============================================================================",
        f"// SiLens {component.replace('_', ' ').title()} Weight Modules",
        f"// =============================================================================",
        f"// Auto-generated include file",
        f"// =============================================================================",
        "",
    ]
    
    component_layers = [l for l in layers if l.component == component]
    
    for layer in component_layers:
        module_name = sanitize_name(layer.name) + "_weights"
        include_lines.append(f'`include "weights/{module_name}.v"')
    
    return "\n".join(include_lines)


def generate_summary_report(
    layers: List[LayerConfig],
    output_dir: Path
) -> str:
    """Generate a summary report of all generated modules."""
    report_lines = [
        "=" * 70,
        "SiLens Weight-to-Verilog Conversion Report",
        "=" * 70,
        "",
        f"Output directory: {output_dir}",
        f"Total layers processed: {len(layers)}",
        "",
    ]
    
    # Group by component
    from collections import defaultdict
    by_component = defaultdict(list)
    for layer in layers:
        by_component[layer.component].append(layer)
    
    total_params = 0
    total_nonzero = 0
    
    for component in sorted(by_component.keys()):
        comp_layers = by_component[component]
        comp_params = sum(l.weights.size for l in comp_layers)
        comp_nonzero = sum(np.count_nonzero(l.weights) for l in comp_layers)
        
        report_lines.append(f"\n{component.upper()}")
        report_lines.append("-" * 40)
        report_lines.append(f"  Layers: {len(comp_layers)}")
        report_lines.append(f"  Parameters: {comp_params:,}")
        report_lines.append(f"  Non-zero: {comp_nonzero:,} ({100*comp_nonzero/comp_params:.1f}%)")
        
        total_params += comp_params
        total_nonzero += comp_nonzero
    
    report_lines.extend([
        "",
        "=" * 70,
        f"TOTAL PARAMETERS: {total_params:,}",
        f"TOTAL NON-ZERO: {total_nonzero:,} ({100*total_nonzero/total_params:.1f}%)",
        f"ESTIMATED CONNECTIONS: {total_nonzero:,}",
        "=" * 70,
    ])
    
    return "\n".join(report_lines)


def convert_weights_to_verilog(
    weights_path: str,
    output_dir: str,
    layer_filter: Optional[str] = None,
    config: Optional[VerilogConfig] = None
) -> None:
    """
    Main conversion function.
    
    Args:
        weights_path: Path to quantized weights
        output_dir: Output directory for Verilog files
        layer_filter: Optional filter for layer names
        config: Verilog generation configuration
    """
    if config is None:
        config = VerilogConfig()
    
    output_dir = Path(output_dir)
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    weights_dict = load_weights(weights_path)
    
    # Process each layer
    layers = []
    for name, weights in weights_dict.items():
        # Apply filter
        if layer_filter and layer_filter.lower() not in name.lower():
            continue
        
        # Skip non-weight tensors (biases, norms, etc.)
        if weights.ndim < 2:
            print(f"  Skipping 1D tensor: {name}")
            continue
        
        # Quantize to ternary if needed
        weights_ternary = quantize_to_ternary(weights)
        
        # Flatten if needed (handle 3D+ tensors)
        if weights_ternary.ndim > 2:
            original_shape = weights_ternary.shape
            weights_ternary = weights_ternary.reshape(weights_ternary.shape[0], -1)
            print(f"  Flattened {name}: {original_shape} -> {weights_ternary.shape}")
        
        # Categorize
        component = categorize_layer(name)
        
        layers.append(LayerConfig(
            name=name,
            input_dim=weights_ternary.shape[1],
            output_dim=weights_ternary.shape[0],
            weights=weights_ternary,
            component=component
        ))
    
    print(f"\nProcessing {len(layers)} layers...")
    
    # Generate Verilog for each layer
    generated_files = []
    for layer in layers:
        module_name = sanitize_name(layer.name) + "_weights"
        component_dir = output_dir / layer.component / "weights"
        component_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate module
        verilog_code = generate_weight_module(
            module_name=module_name,
            weights=layer.weights,
            config=config,
            original_name=layer.name
        )
        
        # Write module file
        module_file = component_dir / f"{module_name}.v"
        with open(module_file, 'w') as f:
            f.write(verilog_code)
        generated_files.append(module_file)
        
        # Generate testbench if requested
        if config.generate_testbench:
            tb_code = generate_testbench(
                module_name=module_name,
                input_dim=layer.input_dim,
                output_dim=layer.output_dim,
                weights=layer.weights,
                config=config
            )
            tb_file = component_dir / f"{module_name}_tb.v"
            with open(tb_file, 'w') as f:
                f.write(tb_code)
        
        print(f"  Generated: {module_file.relative_to(output_dir)}")
    
    # Generate include files for each component
    for component in set(l.component for l in layers):
        include_code = generate_layer_include(layers, component, output_dir)
        include_file = output_dir / component / "weights.vh"
        include_file.parent.mkdir(parents=True, exist_ok=True)
        with open(include_file, 'w') as f:
            f.write(include_code)
        print(f"  Generated: {include_file.relative_to(output_dir)}")
    
    # Generate summary report
    report = generate_summary_report(layers, output_dir)
    report_file = output_dir / "conversion_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport written to: {report_file}")
    print("\n" + report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert quantized weights to Verilog modules for SiLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all weights
  python weights_to_verilog.py -w model/weights/quantized -o rtl/

  # Convert only vision encoder weights
  python weights_to_verilog.py -w model/weights/quantized -o rtl/ -f vision

  # Convert with 16-bit activations
  python weights_to_verilog.py -w model/weights/quantized -o rtl/ --act-width 16
"""
    )
    
    parser.add_argument(
        "-w", "--weights",
        type=str,
        required=True,
        help="Path to quantized weights directory or file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for Verilog files"
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        default=None,
        help="Filter layers by name (case-insensitive substring match)"
    )
    parser.add_argument(
        "--act-width",
        type=int,
        default=DEFAULT_ACT_WIDTH,
        help=f"Activation bit width (default: {DEFAULT_ACT_WIDTH})"
    )
    parser.add_argument(
        "--acc-width",
        type=int,
        default=DEFAULT_ACC_WIDTH,
        help=f"Accumulator bit width (default: {DEFAULT_ACC_WIDTH})"
    )
    parser.add_argument(
        "--no-testbench",
        action="store_true",
        help="Skip testbench generation"
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Omit comments in generated Verilog"
    )
    
    args = parser.parse_args()
    
    config = VerilogConfig(
        act_width=args.act_width,
        acc_width=args.acc_width,
        generate_testbench=not args.no_testbench,
        include_comments=not args.no_comments
    )
    
    print("=" * 70)
    print("SiLens Weights-to-Verilog Converter")
    print("=" * 70)
    print(f"Weights: {args.weights}")
    print(f"Output: {args.output}")
    print(f"Filter: {args.filter or 'None'}")
    print(f"Activation width: {config.act_width} bits")
    print(f"Accumulator width: {config.acc_width} bits")
    print("=" * 70)
    
    try:
        convert_weights_to_verilog(
            weights_path=args.weights,
            output_dir=args.output,
            layer_filter=args.filter,
            config=config
        )
        print("\n✓ Conversion complete!")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
