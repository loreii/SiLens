"""
SiLens RTL Simulation Interface
================================

Python module providing a clean interface to invoke end-to-end RTL simulation
of the SiLens accelerator. This module handles:

- Setting up simulation environment
- Invoking cocotb simulation via subprocess
- Parsing and returning results
- Error handling and logging

Usage:
    from sim_interface import run_e2e_simulation, SimulationConfig
    
    # Basic usage
    results = run_e2e_simulation(image_array, prompt_tokens)
    
    # With custom configuration
    config = SimulationConfig(simulator='verilator', timeout_sec=300)
    results = run_e2e_simulation(image_array, prompt_tokens, config=config)

Author: SiLens Team
License: Apache 2.0
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for RTL simulation."""
    # Simulator selection
    simulator: str = "icarus"  # "icarus" or "verilator"
    
    # Paths
    rtl_dir: Optional[str] = None
    tb_dir: Optional[str] = None
    build_dir: Optional[str] = None
    weights_dir: Optional[str] = None
    
    # Timing
    timeout_sec: int = 600  # 10 minutes default
    clock_freq_mhz: int = 100
    
    # Simulation parameters
    dump_waves: bool = False
    wave_format: str = "fst"  # "fst" or "vcd"
    verbose: bool = False
    
    # Model parameters (must match silens_top.v)
    img_size: int = 384
    patch_size: int = 16
    vision_dim: int = 768
    llm_dim: int = 576
    vocab_size: int = 49152
    
    def __post_init__(self):
        """Set default paths based on module location."""
        if self.rtl_dir is None:
            # Determine RTL directory from this file's location
            module_dir = Path(__file__).parent
            self.rtl_dir = str(module_dir.parent)
        
        if self.tb_dir is None:
            self.tb_dir = str(Path(self.rtl_dir) / "tb")
        
        if self.build_dir is None:
            self.build_dir = str(Path(self.tb_dir) / "build" / "e2e")


@dataclass
class SimulationResults:
    """Results from RTL simulation."""
    success: bool
    output_tokens: List[int]
    
    # Cycle counts
    total_cycles: int
    vision_cycles: int
    prefill_cycles: int
    decode_cycles: int
    
    # Timing
    vision_start_cycle: int = 0
    vision_end_cycle: int = 0
    prefill_start_cycle: int = 0
    prefill_end_cycle: int = 0
    decode_start_cycle: int = 0
    decode_end_cycle: int = 0
    
    # Token timing
    token_cycle_times: List[int] = field(default_factory=list)
    time_to_first_token: int = 0
    
    # Performance metrics
    tokens_per_second: float = 0.0
    throughput_tokens_per_cycle: float = 0.0
    
    # Error information
    error_message: Optional[str] = None
    
    # Raw simulator output
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    
    @classmethod
    def from_json(cls, json_path: str, stdout: str = "", stderr: str = "", 
                  return_code: int = 0) -> 'SimulationResults':
        """Load results from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            return cls(
                success=data.get('success', False),
                output_tokens=data.get('output_tokens', []),
                total_cycles=data.get('total_cycles', 0),
                vision_cycles=data.get('vision_cycles', 0),
                prefill_cycles=data.get('prefill_cycles', 0),
                decode_cycles=data.get('decode_cycles', 0),
                vision_start_cycle=data.get('vision_start_cycle', 0),
                vision_end_cycle=data.get('vision_end_cycle', 0),
                prefill_start_cycle=data.get('prefill_start_cycle', 0),
                prefill_end_cycle=data.get('prefill_end_cycle', 0),
                decode_start_cycle=data.get('decode_start_cycle', 0),
                decode_end_cycle=data.get('decode_end_cycle', 0),
                token_cycle_times=data.get('token_cycle_times', []),
                time_to_first_token=data.get('time_to_first_token', 0),
                tokens_per_second=data.get('tokens_per_second', 0.0),
                throughput_tokens_per_cycle=data.get('throughput_tokens_per_cycle', 0.0),
                error_message=data.get('error_message'),
                stdout=stdout,
                stderr=stderr,
                return_code=return_code
            )
        except Exception as e:
            return cls(
                success=False,
                output_tokens=[],
                total_cycles=0,
                vision_cycles=0,
                prefill_cycles=0,
                decode_cycles=0,
                error_message=f"Failed to parse results: {e}",
                stdout=stdout,
                stderr=stderr,
                return_code=return_code
            )
    
    @classmethod
    def from_error(cls, error: str, stdout: str = "", stderr: str = "", 
                   return_code: int = -1) -> 'SimulationResults':
        """Create error result."""
        return cls(
            success=False,
            output_tokens=[],
            total_cycles=0,
            vision_cycles=0,
            prefill_cycles=0,
            decode_cycles=0,
            error_message=error,
            stdout=stdout,
            stderr=stderr,
            return_code=return_code
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'output_tokens': self.output_tokens,
            'total_cycles': self.total_cycles,
            'vision_cycles': self.vision_cycles,
            'prefill_cycles': self.prefill_cycles,
            'decode_cycles': self.decode_cycles,
            'token_cycle_times': self.token_cycle_times,
            'time_to_first_token': self.time_to_first_token,
            'tokens_per_second': self.tokens_per_second,
            'throughput_tokens_per_cycle': self.throughput_tokens_per_cycle,
            'error_message': self.error_message,
        }
    
    def get_timing_summary(self, clock_freq_mhz: int = 100) -> Dict[str, float]:
        """Get timing summary in milliseconds."""
        cycles_to_ms = 1.0 / (clock_freq_mhz * 1000)
        return {
            'total_ms': self.total_cycles * cycles_to_ms,
            'vision_ms': self.vision_cycles * cycles_to_ms,
            'prefill_ms': self.prefill_cycles * cycles_to_ms,
            'decode_ms': self.decode_cycles * cycles_to_ms,
            'time_to_first_token_ms': self.time_to_first_token * cycles_to_ms,
        }
    
    def print_summary(self):
        """Print human-readable summary."""
        timing = self.get_timing_summary()
        
        print("\n" + "=" * 60)
        print("SiLens RTL Simulation Results")
        print("=" * 60)
        
        print(f"\nStatus: {'SUCCESS' if self.success else 'FAILED'}")
        if self.error_message:
            print(f"Error: {self.error_message}")
        
        print(f"\nTokens Generated: {len(self.output_tokens)}")
        if self.output_tokens:
            preview = self.output_tokens[:10]
            print(f"  Preview: {preview}{'...' if len(self.output_tokens) > 10 else ''}")
        
        print(f"\nCycle Counts:")
        print(f"  Total:   {self.total_cycles:>12,} cycles ({timing['total_ms']:.2f} ms)")
        print(f"  Vision:  {self.vision_cycles:>12,} cycles ({timing['vision_ms']:.2f} ms)")
        print(f"  Prefill: {self.prefill_cycles:>12,} cycles ({timing['prefill_ms']:.2f} ms)")
        print(f"  Decode:  {self.decode_cycles:>12,} cycles ({timing['decode_ms']:.2f} ms)")
        
        print(f"\nPerformance:")
        print(f"  Time to First Token: {self.time_to_first_token:,} cycles ({timing['time_to_first_token_ms']:.2f} ms)")
        print(f"  Tokens/Second @ 100MHz: {self.tokens_per_second:.2f}")
        
        if self.token_cycle_times:
            avg_cycles = sum(self.token_cycle_times) / len(self.token_cycle_times)
            print(f"  Avg Cycles/Token: {avg_cycles:.1f}")
        
        print("=" * 60 + "\n")


# =============================================================================
# RTL File Collection
# =============================================================================

def get_rtl_sources(rtl_dir: str) -> List[str]:
    """Collect all RTL source files."""
    rtl_path = Path(rtl_dir)
    sources = []
    
    # Order matters for dependencies
    source_dirs = [
        'common',
        'memory',
        'vision_encoder',
        'projector',
        'language_model',
        'pcie',
        'top',
    ]
    
    for subdir in source_dirs:
        dir_path = rtl_path / subdir
        if dir_path.exists():
            for vfile in sorted(dir_path.glob('*.v')):
                sources.append(str(vfile))
            for svfile in sorted(dir_path.glob('*.sv')):
                sources.append(str(svfile))
    
    return sources


# =============================================================================
# Main Simulation Interface
# =============================================================================

def run_e2e_simulation(
    image: Union[np.ndarray, str, Path],
    tokens: Union[List[int], np.ndarray],
    weights_dir: Optional[str] = None,
    config: Optional[SimulationConfig] = None
) -> SimulationResults:
    """
    Run end-to-end RTL simulation of SiLens accelerator.
    
    Args:
        image: Input image as numpy array (H, W, 3) or path to image/npy file
        tokens: Input prompt tokens as list of integers
        weights_dir: Optional directory containing weight files
        config: Optional simulation configuration
    
    Returns:
        SimulationResults object with outputs and timing information
    """
    if config is None:
        config = SimulationConfig()
    
    logger.info("Starting SiLens RTL simulation")
    logger.info(f"  Simulator: {config.simulator}")
    logger.info(f"  RTL Dir: {config.rtl_dir}")
    
    # Create temporary directory for this simulation
    temp_dir = tempfile.mkdtemp(prefix='silens_sim_')
    logger.info(f"  Temp Dir: {temp_dir}")
    
    try:
        # Prepare image input
        if isinstance(image, (str, Path)):
            image_path = str(image)
            if not Path(image_path).exists():
                return SimulationResults.from_error(f"Image file not found: {image_path}")
        else:
            # Save numpy array to temp file
            image_array = np.asarray(image, dtype=np.float32)
            if image_array.ndim != 3 or image_array.shape[2] != 3:
                return SimulationResults.from_error(
                    f"Invalid image shape: {image_array.shape}, expected (H, W, 3)"
                )
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            image_path = os.path.join(temp_dir, 'input_image.npy')
            np.save(image_path, image_array)
        
        # Prepare tokens input
        if isinstance(tokens, np.ndarray):
            tokens_list = tokens.tolist()
        else:
            tokens_list = list(tokens)
        tokens_path = os.path.join(temp_dir, 'input_tokens.npy')
        np.save(tokens_path, np.array(tokens_list, dtype=np.int32))
        
        # Results file path
        results_path = os.path.join(temp_dir, 'e2e_results.json')
        
        # Collect RTL sources
        rtl_sources = get_rtl_sources(config.rtl_dir)
        if not rtl_sources:
            return SimulationResults.from_error(f"No RTL sources found in {config.rtl_dir}")
        
        logger.info(f"  RTL Sources: {len(rtl_sources)} files")
        
        # Build environment
        env = os.environ.copy()
        env['SILENS_TEST_IMAGE'] = image_path
        env['SILENS_TEST_TOKENS'] = tokens_path
        env['SILENS_RESULTS_FILE'] = results_path
        env['COCOTB_RESOLVE_X'] = 'ZEROS'
        
        # Add to PYTHONPATH
        pythonpath = [
            config.tb_dir,
            str(Path(config.rtl_dir).parent / 'tests'),
            str(Path(config.rtl_dir).parent),
        ]
        if 'PYTHONPATH' in env:
            pythonpath.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = ':'.join(pythonpath)
        
        if weights_dir:
            env['SILENS_WEIGHTS_DIR'] = weights_dir
        
        # Determine simulator-specific arguments
        if config.simulator == 'icarus':
            compile_args = '-g2012'
            sim_args = '-fst' if config.dump_waves else ''
        elif config.simulator == 'verilator':
            compile_args = '--trace --trace-fst' if config.dump_waves else ''
            sim_args = ''
        else:
            return SimulationResults.from_error(f"Unknown simulator: {config.simulator}")
        
        # Build make command
        make_cmd = [
            'make',
            '-f', str(Path(config.tb_dir) / 'Makefile'),
            'test-e2e-inference',
            f'SIM={config.simulator}',
        ]
        
        if config.verbose:
            make_cmd.append('COCOTB_LOG_LEVEL=DEBUG')
        
        logger.info(f"Running: {' '.join(make_cmd)}")
        
        # Run simulation
        result = subprocess.run(
            make_cmd,
            cwd=config.tb_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec
        )
        
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
        
        if config.verbose:
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
        
        # Parse results
        if Path(results_path).exists():
            return SimulationResults.from_json(
                results_path, stdout, stderr, return_code
            )
        else:
            # Check for common error patterns
            error_msg = "Simulation did not produce results file"
            if "Error" in stderr or "error" in stderr:
                error_lines = [l for l in stderr.split('\n') if 'error' in l.lower()]
                if error_lines:
                    error_msg = error_lines[0]
            elif return_code != 0:
                error_msg = f"Simulation failed with return code {return_code}"
            
            return SimulationResults.from_error(error_msg, stdout, stderr, return_code)
    
    except subprocess.TimeoutExpired:
        return SimulationResults.from_error(
            f"Simulation timed out after {config.timeout_sec} seconds"
        )
    except Exception as e:
        return SimulationResults.from_error(f"Simulation exception: {e}")
    
    finally:
        # Cleanup temp directory
        if not config.verbose:  # Keep for debugging if verbose
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_quick_test(config: Optional[SimulationConfig] = None) -> SimulationResults:
    """
    Run a quick sanity test with minimal inputs.
    
    Useful for verifying simulation setup without full inference.
    """
    if config is None:
        config = SimulationConfig()
    
    # Small test image
    image = np.random.rand(config.img_size, config.img_size, 3).astype(np.float32)
    
    # Minimal tokens
    tokens = [1, 100, 2]  # BOS, single token, EOS
    
    return run_e2e_simulation(image, tokens, config=config)


# =============================================================================
# Utility Functions
# =============================================================================

def check_simulation_environment() -> Dict[str, bool]:
    """Check if simulation environment is properly set up."""
    checks = {
        'make': False,
        'python': False,
        'cocotb': False,
        'icarus': False,
        'verilator': False,
        'numpy': False,
    }
    
    # Check make
    try:
        result = subprocess.run(['make', '--version'], capture_output=True)
        checks['make'] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    # Check Python
    checks['python'] = True  # We're running, so Python works
    
    # Check cocotb
    try:
        import cocotb
        checks['cocotb'] = True
    except ImportError:
        pass
    
    # Check Icarus Verilog
    try:
        result = subprocess.run(['iverilog', '-V'], capture_output=True)
        checks['icarus'] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    # Check Verilator
    try:
        result = subprocess.run(['verilator', '--version'], capture_output=True)
        checks['verilator'] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    # Check numpy
    try:
        import numpy
        checks['numpy'] = True
    except ImportError:
        pass
    
    return checks


def print_environment_status():
    """Print simulation environment status."""
    checks = check_simulation_environment()
    
    print("\nSiLens RTL Simulation Environment")
    print("=" * 40)
    
    for tool, available in checks.items():
        status = "✓" if available else "✗"
        print(f"  {tool:<15} [{status}]")
    
    print()
    
    if not checks['icarus'] and not checks['verilator']:
        print("WARNING: No simulator found!")
        print("  Install Icarus Verilog: brew install icarus-verilog")
        print("  Or Verilator: brew install verilator")
    
    if not checks['cocotb']:
        print("WARNING: cocotb not found!")
        print("  Install: pip install cocotb")


def decode_tokens(tokens: List[int], vocab_path: Optional[str] = None) -> str:
    """
    Decode token IDs to text (placeholder implementation).
    
    In practice, this would use the actual tokenizer vocabulary.
    """
    # Placeholder: return token IDs as string
    return f"[tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}]"


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command line interface for simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SiLens RTL Simulation Interface')
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--tokens', type=str, help='Input tokens (comma-separated)')
    parser.add_argument('--simulator', type=str, default='icarus',
                       choices=['icarus', 'verilator'])
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds')
    parser.add_argument('--waves', action='store_true', help='Dump waveforms')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--check', action='store_true', help='Check environment only')
    parser.add_argument('--quick-test', action='store_true', help='Run quick sanity test')
    
    args = parser.parse_args()
    
    if args.check:
        print_environment_status()
        return
    
    config = SimulationConfig(
        simulator=args.simulator,
        timeout_sec=args.timeout,
        dump_waves=args.waves,
        verbose=args.verbose,
    )
    
    if args.quick_test:
        print("Running quick sanity test...")
        results = run_quick_test(config)
        results.print_summary()
        return
    
    # Prepare inputs
    if args.image:
        if args.image.endswith('.npy'):
            image = np.load(args.image)
        else:
            try:
                from PIL import Image
                img = Image.open(args.image).convert('RGB')
                img = img.resize((config.img_size, config.img_size))
                image = np.array(img).astype(np.float32) / 255.0
            except ImportError:
                print("PIL not available, using random image")
                image = np.random.rand(config.img_size, config.img_size, 3).astype(np.float32)
    else:
        # Default test image
        print("No image specified, using gradient test image")
        x = np.linspace(0, 1, config.img_size)
        y = np.linspace(0, 1, config.img_size)
        xv, yv = np.meshgrid(x, y)
        image = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.float32)
    
    if args.tokens:
        tokens = [int(t.strip()) for t in args.tokens.split(',')]
    else:
        # Default prompt tokens
        tokens = [1, 8612, 436, 2217, 2]
    
    print(f"Image shape: {image.shape}")
    print(f"Tokens: {tokens}")
    
    # Run simulation
    results = run_e2e_simulation(image, tokens, config=config)
    results.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if results.success else 1)


if __name__ == '__main__':
    main()
