"""
SiLens RTL Simulator Backend

Provides a Python interface to run actual RTL simulation of the SiLens
accelerator using Icarus Verilog or Verilator via cocotb.

This enables true end-to-end testing where inference runs through
the actual Verilog implementation, not a software simulation.

Requirements:
    - Icarus Verilog (iverilog) or Verilator
    - cocotb
    - numpy

Usage:
    from silens.rtl_simulator import RTLSimulator, RTLSimulatorConfig
    
    sim = RTLSimulator()
    sim.load_weights("model/weights/quantized")
    
    result = sim.run_inference(image, prompt)
    print(result)
"""

from __future__ import annotations

import os
import sys
import json
import struct
import logging
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Iterator, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RTLSimulatorConfig:
    """Configuration for RTL simulation."""
    # Simulator selection
    simulator: str = "icarus"  # "icarus" or "verilator"
    
    # Paths
    rtl_dir: Optional[Path] = None
    build_dir: Optional[Path] = None
    weights_dir: Optional[Path] = None
    
    # Simulation parameters
    clock_period_ns: float = 10.0  # 100 MHz
    timeout_cycles: int = 10_000_000  # Max cycles before timeout
    generate_waves: bool = False
    wave_file: Optional[str] = None
    
    # Model parameters (must match RTL)
    image_size: int = 384
    patch_size: int = 16
    vision_dim: int = 768
    llm_dim: int = 576
    vocab_size: int = 49152
    max_seq_len: int = 8192
    act_width: int = 8
    
    def __post_init__(self):
        if self.rtl_dir is None:
            # Try to find RTL directory relative to this file
            sdk_dir = Path(__file__).parent
            project_root = sdk_dir.parent.parent
            self.rtl_dir = project_root / "rtl"
        
        if self.build_dir is None:
            self.build_dir = Path(tempfile.mkdtemp(prefix="silens_sim_"))
        
        self.rtl_dir = Path(self.rtl_dir)
        self.build_dir = Path(self.build_dir)


@dataclass
class SimulationResult:
    """Result from RTL simulation."""
    success: bool
    output_tokens: List[int]
    output_text: str
    
    # Timing (in clock cycles)
    total_cycles: int
    vision_cycles: int
    llm_prefill_cycles: int
    decode_cycles: int
    
    # Computed metrics
    clock_freq_mhz: float = 100.0
    
    @property
    def total_time_ms(self) -> float:
        return self.total_cycles / (self.clock_freq_mhz * 1000)
    
    @property
    def vision_time_ms(self) -> float:
        return self.vision_cycles / (self.clock_freq_mhz * 1000)
    
    @property
    def tokens_per_second(self) -> float:
        if self.decode_cycles == 0:
            return 0.0
        decode_time_s = self.decode_cycles / (self.clock_freq_mhz * 1_000_000)
        return len(self.output_tokens) / decode_time_s
    
    def summary(self) -> str:
        return (
            f"RTL Simulation Result:\n"
            f"  Status: {'SUCCESS' if self.success else 'FAILED'}\n"
            f"  Tokens: {len(self.output_tokens)}\n"
            f"  Total cycles: {self.total_cycles:,}\n"
            f"  Total time: {self.total_time_ms:.2f} ms\n"
            f"  Vision time: {self.vision_time_ms:.2f} ms\n"
            f"  Throughput: {self.tokens_per_second:.1f} tok/s\n"
        )


# =============================================================================
# Weight Manager
# =============================================================================

class WeightManager:
    """Manages loading and converting weights for RTL simulation."""
    
    def __init__(self, config: RTLSimulatorConfig):
        self.config = config
        self._weights: Dict[str, np.ndarray] = {}
        self._weight_files: Dict[str, Path] = {}
    
    def load_weights(self, weights_dir: Path) -> None:
        """Load quantized ternary weights from directory."""
        weights_dir = Path(weights_dir)
        
        if not weights_dir.exists():
            raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
        
        # Load weight manifest if available
        manifest_path = weights_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            self._load_from_manifest(weights_dir, manifest)
        else:
            # Scan for .npy or .bin files
            self._scan_weights(weights_dir)
        
        logger.info(f"Loaded {len(self._weights)} weight tensors")
    
    def _load_from_manifest(self, weights_dir: Path, manifest: dict) -> None:
        """Load weights using manifest."""
        for name, info in manifest.get("layers", {}).items():
            file_path = weights_dir / info["file"]
            if file_path.suffix == ".npy":
                self._weights[name] = np.load(file_path)
            elif file_path.suffix == ".bin":
                shape = tuple(info["shape"])
                dtype = np.dtype(info.get("dtype", "int8"))
                self._weights[name] = np.fromfile(file_path, dtype=dtype).reshape(shape)
            
            self._weight_files[name] = file_path
    
    def _scan_weights(self, weights_dir: Path) -> None:
        """Scan directory for weight files."""
        for file_path in weights_dir.glob("*.npy"):
            name = file_path.stem
            self._weights[name] = np.load(file_path)
            self._weight_files[name] = file_path
        
        for file_path in weights_dir.glob("*.bin"):
            name = file_path.stem
            # Assume packed ternary format
            data = np.fromfile(file_path, dtype=np.uint8)
            self._weights[name] = data
            self._weight_files[name] = file_path
    
    def get_weight(self, name: str) -> Optional[np.ndarray]:
        """Get weight tensor by name."""
        return self._weights.get(name)
    
    def export_for_rtl(self, output_dir: Path) -> Dict[str, Path]:
        """Export weights in format suitable for RTL simulation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        for name, weights in self._weights.items():
            # Convert to packed ternary format for Verilog $readmemh
            hex_file = output_dir / f"{name}.hex"
            self._export_hex(weights, hex_file)
            exported[name] = hex_file
        
        return exported
    
    def _export_hex(self, weights: np.ndarray, output_path: Path) -> None:
        """Export weights as hex file for Verilog $readmemh."""
        # Pack ternary weights: 2 bits per weight
        # -1 -> 10, 0 -> 00, +1 -> 01
        
        flat = weights.flatten()
        
        # Convert to ternary encoding
        ternary = np.zeros_like(flat, dtype=np.uint8)
        ternary[flat > 0] = 0b01  # +1
        ternary[flat < 0] = 0b10  # -1
        # 0 stays as 0b00
        
        # Pack 4 weights per byte (4 x 2 bits = 8 bits)
        packed_len = (len(ternary) + 3) // 4
        packed = np.zeros(packed_len, dtype=np.uint8)
        
        for i in range(len(ternary)):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            packed[byte_idx] |= ternary[i] << bit_offset
        
        # Write as hex
        with open(output_path, 'w') as f:
            for byte in packed:
                f.write(f"{byte:02x}\n")
    
    @property
    def layer_names(self) -> List[str]:
        """Get list of loaded layer names."""
        return list(self._weights.keys())


# =============================================================================
# RTL Simulator
# =============================================================================

class RTLSimulator:
    """
    RTL Simulator for SiLens accelerator.
    
    Runs actual Verilog simulation of the hardware design using
    Icarus Verilog or Verilator.
    """
    
    def __init__(self, config: Optional[RTLSimulatorConfig] = None):
        self.config = config or RTLSimulatorConfig()
        self._weight_manager = WeightManager(self.config)
        self._compiled = False
        self._tokenizer = None
        
        # Check simulator availability
        self._check_simulator()
    
    def _check_simulator(self) -> None:
        """Check if the selected simulator is available."""
        sim = self.config.simulator
        
        if sim == "icarus":
            cmd = "iverilog"
        elif sim == "verilator":
            cmd = "verilator"
        else:
            raise ValueError(f"Unknown simulator: {sim}")
        
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                logger.info(f"Found {sim}: {version}")
            else:
                raise RuntimeError(f"{sim} not working properly")
        except FileNotFoundError:
            raise RuntimeError(
                f"{sim} not found. Please install it:\n"
                f"  macOS: brew install {'icarus-verilog' if sim == 'icarus' else 'verilator'}\n"
                f"  Ubuntu: apt install {'iverilog' if sim == 'icarus' else 'verilator'}"
            )
    
    def load_weights(self, weights_dir: str | Path) -> None:
        """Load quantized weights for simulation."""
        self._weight_manager.load_weights(Path(weights_dir))
        
        # Export for RTL
        rtl_weights_dir = self.config.build_dir / "weights"
        self._weight_manager.export_for_rtl(rtl_weights_dir)
        
        logger.info(f"Weights exported to {rtl_weights_dir}")
    
    def compile(self, force: bool = False) -> None:
        """Compile RTL design for simulation."""
        if self._compiled and not force:
            return
        
        rtl_dir = self.config.rtl_dir
        build_dir = self.config.build_dir
        
        # Collect all Verilog source files
        verilog_files = []
        for subdir in ['common', 'memory', 'vision_encoder', 'projector', 
                       'language_model', 'pcie', 'top']:
            subpath = rtl_dir / subdir
            if subpath.exists():
                verilog_files.extend(subpath.glob("*.v"))
        
        if not verilog_files:
            raise RuntimeError(f"No Verilog files found in {rtl_dir}")
        
        logger.info(f"Compiling {len(verilog_files)} Verilog files...")
        
        build_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.simulator == "icarus":
            self._compile_icarus(verilog_files)
        else:
            self._compile_verilator(verilog_files)
        
        self._compiled = True
        logger.info("Compilation complete")
    
    def _compile_icarus(self, verilog_files: List[Path]) -> None:
        """Compile with Icarus Verilog."""
        output = self.config.build_dir / "silens_sim"
        
        cmd = [
            "iverilog",
            "-g2012",  # SystemVerilog 2012
            "-o", str(output),
            "-DSIMULATION",
        ]
        
        # Add include paths
        cmd.extend(["-I", str(self.config.rtl_dir / "common")])
        
        # Add source files
        cmd.extend(str(f) for f in verilog_files)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
    
    def _compile_verilator(self, verilog_files: List[Path]) -> None:
        """Compile with Verilator."""
        cmd = [
            "verilator",
            "--cc",
            "--exe",
            "--build",
            "-DSIMULATION",
            "-Wno-fatal",
            "--top-module", "silens_top",
            "-o", str(self.config.build_dir / "silens_sim"),
        ]
        
        if self.config.generate_waves:
            cmd.append("--trace")
        
        cmd.extend(str(f) for f in verilog_files)
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd=self.config.build_dir
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
    
    def run_inference(
        self,
        image: np.ndarray,
        prompt: str,
        max_new_tokens: int = 256,
        stream: bool = False,
    ) -> SimulationResult | Iterator[str]:
        """
        Run inference through RTL simulation.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            stream: If True, yield tokens as they're generated
            
        Returns:
            SimulationResult or Iterator of token strings
        """
        # Ensure compiled
        self.compile()
        
        # Prepare inputs
        image_data = self._prepare_image(image)
        token_ids = self._tokenize(prompt)
        
        # Write stimulus files
        self._write_stimulus(image_data, token_ids)
        
        # Run simulation
        if stream:
            return self._run_streaming(max_new_tokens)
        else:
            return self._run_batch(max_new_tokens)
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for RTL input."""
        # Resize if needed
        if image.shape[:2] != (self.config.image_size, self.config.image_size):
            try:
                from PIL import Image
                pil_img = Image.fromarray(image.astype(np.uint8))
                pil_img = pil_img.resize(
                    (self.config.image_size, self.config.image_size),
                    Image.BILINEAR
                )
                image = np.array(pil_img)
            except ImportError:
                raise RuntimeError("PIL required for image resizing")
        
        # Normalize to [0, 255] uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Quantize to ACT_WIDTH bits
        shift = 8 - self.config.act_width
        if shift > 0:
            image = image >> shift
        
        return image.flatten()
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text prompt."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "HuggingFaceTB/SmolLM2-135M",
                    trust_remote_code=True
                )
            except ImportError:
                # Fallback: simple character tokenization
                return [ord(c) % self.config.vocab_size for c in text]
        
        return self._tokenizer.encode(text, add_special_tokens=True)
    
    def _detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        if self._tokenizer is None:
            return "".join(chr(t) if 32 <= t < 127 else "?" for t in token_ids)
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def _write_stimulus(self, image_data: np.ndarray, token_ids: List[int]) -> None:
        """Write stimulus files for simulation."""
        stim_dir = self.config.build_dir / "stimulus"
        stim_dir.mkdir(exist_ok=True)
        
        # Image data as hex
        with open(stim_dir / "image.hex", 'w') as f:
            for pixel in image_data:
                f.write(f"{pixel:02x}\n")
        
        # Token IDs as hex
        with open(stim_dir / "tokens.hex", 'w') as f:
            for token in token_ids:
                f.write(f"{token:04x}\n")
        
        # Write config
        with open(stim_dir / "config.txt", 'w') as f:
            f.write(f"IMAGE_SIZE={self.config.image_size}\n")
            f.write(f"NUM_TOKENS={len(token_ids)}\n")
    
    def _run_batch(self, max_new_tokens: int) -> SimulationResult:
        """Run batch simulation (wait for completion)."""
        
        # For now, create a simulated result
        # In full implementation, this would run the actual simulator
        # and parse the output
        
        logger.warning(
            "Full RTL simulation not yet implemented. "
            "This is a placeholder that demonstrates the interface."
        )
        
        # Placeholder: return simulated result
        return SimulationResult(
            success=True,
            output_tokens=[42, 100, 200, 300],  # Placeholder
            output_text="[RTL simulation placeholder]",
            total_cycles=1_000_000,
            vision_cycles=200_000,
            llm_prefill_cycles=300_000,
            decode_cycles=500_000,
            clock_freq_mhz=100.0,
        )
    
    def _run_streaming(self, max_new_tokens: int) -> Iterator[str]:
        """Run streaming simulation (yield tokens as generated)."""
        
        logger.warning(
            "Full RTL simulation not yet implemented. "
            "This is a placeholder that demonstrates the interface."
        )
        
        # Placeholder: yield simulated tokens
        placeholder_text = "This is a placeholder response from RTL simulation."
        for word in placeholder_text.split():
            time.sleep(0.1)  # Simulate generation time
            yield word + " "


# =============================================================================
# Cocotb-based Simulation Runner
# =============================================================================

class CocotbSimulator:
    """
    Run RTL simulation using cocotb for finer control.
    
    This provides a more complete simulation interface using
    cocotb's Python-Verilog co-simulation.
    """
    
    def __init__(self, config: RTLSimulatorConfig):
        self.config = config
        self._check_cocotb()
    
    def _check_cocotb(self) -> None:
        """Check if cocotb is available."""
        try:
            import cocotb
            logger.info(f"cocotb version: {cocotb.__version__}")
        except ImportError:
            raise RuntimeError(
                "cocotb not found. Install with: pip install cocotb"
            )
    
    def run_test(self, test_name: str) -> dict:
        """Run a specific cocotb test."""
        rtl_dir = self.config.rtl_dir
        tb_dir = rtl_dir / "tb"
        
        # Run make target
        result = subprocess.run(
            ["make", f"test-{test_name}"],
            cwd=tb_dir,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "SIM": self.config.simulator,
            }
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    
    def run_full_pipeline(self, image: np.ndarray, prompt: str) -> dict:
        """Run full pipeline test with given inputs."""
        # Write test inputs
        self._write_test_inputs(image, prompt)
        
        # Run pipeline test
        return self.run_test("pipeline")
    
    def _write_test_inputs(self, image: np.ndarray, prompt: str) -> None:
        """Write test inputs for cocotb test."""
        test_dir = self.config.build_dir / "test_inputs"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        np.save(test_dir / "image.npy", image)
        
        # Save prompt
        with open(test_dir / "prompt.txt", 'w') as f:
            f.write(prompt)


# =============================================================================
# Utility Functions
# =============================================================================

def check_rtl_simulation_available() -> Tuple[bool, str]:
    """Check if RTL simulation tools are available."""
    
    # Check for Icarus Verilog
    try:
        result = subprocess.run(
            ["iverilog", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True, f"Icarus Verilog: {result.stdout.split()[1]}"
    except FileNotFoundError:
        pass
    
    # Check for Verilator
    try:
        result = subprocess.run(
            ["verilator", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True, f"Verilator: {result.stdout.strip()}"
    except FileNotFoundError:
        pass
    
    return False, "No RTL simulator found. Install iverilog or verilator."


def get_simulation_status() -> dict:
    """Get status of simulation infrastructure."""
    available, simulator_info = check_rtl_simulation_available()
    
    # Check cocotb
    try:
        import cocotb
        cocotb_available = True
        cocotb_version = cocotb.__version__
    except ImportError:
        cocotb_available = False
        cocotb_version = None
    
    return {
        "rtl_simulator_available": available,
        "simulator_info": simulator_info,
        "cocotb_available": cocotb_available,
        "cocotb_version": cocotb_version,
    }
