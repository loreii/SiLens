"""
SiLens Performance Profiler.

Provides detailed performance analysis including:
- Layer-by-layer timing breakdown
- Memory bandwidth utilization
- Compute unit efficiency
- Token throughput analysis
- Thermal monitoring
- Power consumption tracking

Usage:
    from silens import SiLensDevice, Profiler
    
    with SiLensDevice.discover()[0] as device:
        profiler = Profiler(device)
        profiler.start()
        
        # Run inference
        result = device.infer(image, prompt)
        
        profiler.stop()
        report = profiler.get_report()
        profiler.print_summary()
"""

from __future__ import annotations

import json
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .device import SiLensDevice, Registers, StatusBits


class ProfileEventType(IntEnum):
    """Types of profiling events."""
    INFERENCE_START = 0
    INFERENCE_END = 1
    VISION_START = 2
    VISION_END = 3
    PROJECTION_START = 4
    PROJECTION_END = 5
    LLM_PREFILL_START = 6
    LLM_PREFILL_END = 7
    LLM_DECODE_START = 8
    LLM_DECODE_END = 9
    TOKEN_GENERATED = 10
    DMA_START = 11
    DMA_END = 12
    THERMAL_SAMPLE = 13
    POWER_SAMPLE = 14


@dataclass
class ProfileEvent:
    """A single profiling event."""
    timestamp: float
    event_type: ProfileEventType
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerProfile:
    """Profile data for a single layer."""
    name: str
    layer_type: str
    execution_time_ms: float
    memory_reads_bytes: int
    memory_writes_bytes: int
    compute_ops: int
    utilization_pct: float


@dataclass
class TokenProfile:
    """Profile data for token generation."""
    token_index: int
    token_id: int
    generation_time_ms: float
    cumulative_time_ms: float


@dataclass 
class ThermalSample:
    """Thermal monitoring sample."""
    timestamp: float
    temperature_c: float
    throttled: bool


@dataclass
class PowerSample:
    """Power consumption sample."""
    timestamp: float
    power_mw: float
    voltage_mv: float
    current_ma: float


@dataclass
class ProfileReport:
    """Complete profiling report."""
    # Timing summary
    total_time_ms: float
    vision_time_ms: float
    projection_time_ms: float
    llm_prefill_time_ms: float
    llm_decode_time_ms: float
    
    # Throughput
    tokens_generated: int
    tokens_per_second: float
    time_to_first_token_ms: float
    
    # Memory
    total_memory_reads_mb: float
    total_memory_writes_mb: float
    memory_bandwidth_gbps: float
    
    # Compute
    total_operations: int
    tops: float  # Tera ops per second
    compute_utilization_pct: float
    
    # Per-layer breakdown
    layer_profiles: List[LayerProfile] = field(default_factory=list)
    
    # Token timing
    token_profiles: List[TokenProfile] = field(default_factory=list)
    
    # Thermal and power
    thermal_samples: List[ThermalSample] = field(default_factory=list)
    power_samples: List[PowerSample] = field(default_factory=list)
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    avg_temperature_c: float = 0.0
    peak_temperature_c: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        return d
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ProfileReport':
        """Load report from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class Profiler:
    """
    Hardware performance profiler for SiLens devices.
    
    Collects detailed performance metrics during inference including
    timing, memory bandwidth, compute utilization, and thermal data.
    
    Example:
        profiler = Profiler(device)
        with profiler:
            result = device.infer(image, prompt)
        profiler.print_summary()
    """
    
    # Hardware debug register offsets for profiling
    DEBUG_CYCLE_COUNT = 0x200
    DEBUG_VISION_CYCLES = 0x204
    DEBUG_PROJECT_CYCLES = 0x208
    DEBUG_LLM_CYCLES = 0x20C
    DEBUG_TOKEN_COUNT = 0x210
    DEBUG_MEM_READS = 0x214
    DEBUG_MEM_WRITES = 0x218
    DEBUG_COMPUTE_ACTIVE = 0x21C
    DEBUG_TEMPERATURE = 0x220
    DEBUG_POWER = 0x224
    
    def __init__(self, device: SiLensDevice, 
                 sample_interval_ms: float = 10.0,
                 enable_thermal: bool = True,
                 enable_power: bool = True):
        """
        Initialize the profiler.
        
        Args:
            device: SiLens device to profile
            sample_interval_ms: Sampling interval for continuous metrics
            enable_thermal: Enable thermal monitoring
            enable_power: Enable power monitoring
        """
        self.device = device
        self.sample_interval_ms = sample_interval_ms
        self.enable_thermal = enable_thermal
        self.enable_power = enable_power
        
        self._events: List[ProfileEvent] = []
        self._token_times: List[float] = []
        self._thermal_samples: List[ThermalSample] = []
        self._power_samples: List[PowerSample] = []
        
        self._sampling_thread: Optional[threading.Thread] = None
        self._stop_sampling = threading.Event()
        self._is_profiling = False
        
        self._start_time: float = 0
        self._start_cycles: int = 0
        
    def __enter__(self) -> 'Profiler':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
    
    def start(self) -> None:
        """Start profiling."""
        if self._is_profiling:
            return
        
        self._events.clear()
        self._token_times.clear()
        self._thermal_samples.clear()
        self._power_samples.clear()
        
        self._start_time = time.perf_counter()
        self._start_cycles = self._read_cycle_count()
        
        self._record_event(ProfileEventType.INFERENCE_START)
        
        # Start background sampling thread
        self._stop_sampling.clear()
        self._sampling_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True
        )
        self._sampling_thread.start()
        
        self._is_profiling = True
    
    def stop(self) -> None:
        """Stop profiling."""
        if not self._is_profiling:
            return
        
        self._record_event(ProfileEventType.INFERENCE_END)
        
        # Stop sampling thread
        self._stop_sampling.set()
        if self._sampling_thread:
            self._sampling_thread.join(timeout=1.0)
        
        self._is_profiling = False

    def record_vision_start(self) -> None:
        """Record start of vision encoding."""
        self._record_event(ProfileEventType.VISION_START)
    
    def record_vision_end(self) -> None:
        """Record end of vision encoding."""
        self._record_event(ProfileEventType.VISION_END)
    
    def record_llm_prefill_start(self) -> None:
        """Record start of LLM prefill phase."""
        self._record_event(ProfileEventType.LLM_PREFILL_START)
    
    def record_llm_prefill_end(self) -> None:
        """Record end of LLM prefill phase."""
        self._record_event(ProfileEventType.LLM_PREFILL_END)
    
    def record_token_generated(self, token_id: int) -> None:
        """Record generation of a token."""
        self._token_times.append(time.perf_counter())
        self._record_event(ProfileEventType.TOKEN_GENERATED, {'token_id': token_id})
    
    def _record_event(self, event_type: ProfileEventType, 
                      data: Optional[Dict[str, Any]] = None) -> None:
        """Record a profiling event."""
        event = ProfileEvent(
            timestamp=time.perf_counter() - self._start_time,
            event_type=event_type,
            data=data or {}
        )
        self._events.append(event)
    
    def _read_cycle_count(self) -> int:
        """Read hardware cycle counter."""
        try:
            return self.device.read_reg(self.DEBUG_CYCLE_COUNT)
        except Exception:
            return 0
    
    def _read_temperature(self) -> float:
        """Read temperature in Celsius."""
        try:
            raw = self.device.read_reg(self.DEBUG_TEMPERATURE)
            return raw / 256.0  # Fixed-point conversion
        except Exception:
            return 0.0
    
    def _read_power(self) -> Tuple[float, float, float]:
        """Read power metrics (power_mw, voltage_mv, current_ma)."""
        try:
            raw = self.device.read_reg(self.DEBUG_POWER)
            power_mw = (raw >> 16) & 0xFFFF
            # Derive voltage and current from power (simplified)
            voltage_mv = 900  # Assume 0.9V core
            current_ma = power_mw * 1000 / voltage_mv if voltage_mv > 0 else 0
            return float(power_mw), float(voltage_mv), float(current_ma)
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _sampling_loop(self) -> None:
        """Background thread for continuous sampling."""
        while not self._stop_sampling.is_set():
            timestamp = time.perf_counter() - self._start_time
            
            if self.enable_thermal:
                temp = self._read_temperature()
                status = self.device.get_status()
                throttled = bool(status & 0x80)  # Assuming throttle bit
                self._thermal_samples.append(ThermalSample(
                    timestamp=timestamp,
                    temperature_c=temp,
                    throttled=throttled
                ))
            
            if self.enable_power:
                power, voltage, current = self._read_power()
                self._power_samples.append(PowerSample(
                    timestamp=timestamp,
                    power_mw=power,
                    voltage_mv=voltage,
                    current_ma=current
                ))
            
            time.sleep(self.sample_interval_ms / 1000.0)

    def get_report(self) -> ProfileReport:
        """
        Generate profiling report from collected data.
        
        Returns:
            ProfileReport with all metrics
        """
        # Find timing events
        inference_start = self._find_event(ProfileEventType.INFERENCE_START)
        inference_end = self._find_event(ProfileEventType.INFERENCE_END)
        vision_start = self._find_event(ProfileEventType.VISION_START)
        vision_end = self._find_event(ProfileEventType.VISION_END)
        prefill_start = self._find_event(ProfileEventType.LLM_PREFILL_START)
        prefill_end = self._find_event(ProfileEventType.LLM_PREFILL_END)
        
        # Calculate timings
        total_time_ms = 0.0
        if inference_start and inference_end:
            total_time_ms = (inference_end.timestamp - inference_start.timestamp) * 1000
        
        vision_time_ms = 0.0
        if vision_start and vision_end:
            vision_time_ms = (vision_end.timestamp - vision_start.timestamp) * 1000
        
        prefill_time_ms = 0.0
        if prefill_start and prefill_end:
            prefill_time_ms = (prefill_end.timestamp - prefill_start.timestamp) * 1000
        
        # Token metrics
        token_events = [e for e in self._events 
                       if e.event_type == ProfileEventType.TOKEN_GENERATED]
        tokens_generated = len(token_events)
        
        tokens_per_second = 0.0
        if tokens_generated > 0 and total_time_ms > 0:
            tokens_per_second = tokens_generated / (total_time_ms / 1000.0)
        
        ttft_ms = 0.0
        if token_events and inference_start:
            ttft_ms = (token_events[0].timestamp - inference_start.timestamp) * 1000
        
        # Calculate decode time
        decode_time_ms = 0.0
        if token_events:
            decode_time_ms = (token_events[-1].timestamp - token_events[0].timestamp) * 1000
        
        # Build token profiles
        token_profiles = []
        for i, event in enumerate(token_events):
            cumulative = (event.timestamp - (inference_start.timestamp if inference_start else 0)) * 1000
            gen_time = 0.0
            if i > 0:
                gen_time = (event.timestamp - token_events[i-1].timestamp) * 1000
            elif prefill_end:
                gen_time = (event.timestamp - prefill_end.timestamp) * 1000
            
            token_profiles.append(TokenProfile(
                token_index=i,
                token_id=event.data.get('token_id', 0),
                generation_time_ms=gen_time,
                cumulative_time_ms=cumulative
            ))
        
        # Memory metrics (from hardware registers)
        try:
            mem_reads = self.device.read_reg(self.DEBUG_MEM_READS) * 64  # Assume 64B per transaction
            mem_writes = self.device.read_reg(self.DEBUG_MEM_WRITES) * 64
        except Exception:
            mem_reads = 0
            mem_writes = 0
        
        mem_reads_mb = mem_reads / (1024 * 1024)
        mem_writes_mb = mem_writes / (1024 * 1024)
        
        bandwidth_gbps = 0.0
        if total_time_ms > 0:
            total_bytes = mem_reads + mem_writes
            bandwidth_gbps = (total_bytes / 1e9) / (total_time_ms / 1000.0)
        
        # Compute metrics
        # Estimate based on model architecture
        vision_ops = 576 * 768 * 768 * 12 * 4  # Patches * dim * dim * layers * (Q,K,V,O)
        llm_ops = 8192 * 576 * 576 * 30 * 4     # seq_len * dim * dim * layers
        total_ops = vision_ops + llm_ops
        
        tops = 0.0
        if total_time_ms > 0:
            tops = total_ops / (total_time_ms / 1000.0) / 1e12
        
        # Thermal stats
        avg_temp = 0.0
        peak_temp = 0.0
        if self._thermal_samples:
            temps = [s.temperature_c for s in self._thermal_samples]
            avg_temp = np.mean(temps)
            peak_temp = np.max(temps)
        
        # Power stats
        avg_power = 0.0
        peak_power = 0.0
        if self._power_samples:
            powers = [s.power_mw for s in self._power_samples]
            avg_power = np.mean(powers) / 1000.0  # Convert to W
            peak_power = np.max(powers) / 1000.0
        
        return ProfileReport(
            total_time_ms=total_time_ms,
            vision_time_ms=vision_time_ms,
            projection_time_ms=0.0,  # TODO: Add projection tracking
            llm_prefill_time_ms=prefill_time_ms,
            llm_decode_time_ms=decode_time_ms,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            time_to_first_token_ms=ttft_ms,
            total_memory_reads_mb=mem_reads_mb,
            total_memory_writes_mb=mem_writes_mb,
            memory_bandwidth_gbps=bandwidth_gbps,
            total_operations=total_ops,
            tops=tops,
            compute_utilization_pct=min(100.0, tops / 2.0 * 100),  # Assume 2 TOPS peak
            layer_profiles=[],  # TODO: Per-layer profiling
            token_profiles=token_profiles,
            thermal_samples=self._thermal_samples,
            power_samples=self._power_samples,
            avg_power_w=avg_power,
            peak_power_w=peak_power,
            avg_temperature_c=avg_temp,
            peak_temperature_c=peak_temp
        )
    
    def _find_event(self, event_type: ProfileEventType) -> Optional[ProfileEvent]:
        """Find first event of given type."""
        for event in self._events:
            if event.event_type == event_type:
                return event
        return None

    def print_summary(self) -> None:
        """Print human-readable summary of profiling results."""
        report = self.get_report()
        
        print("\n" + "=" * 70)
        print("SILENS PERFORMANCE PROFILE")
        print("=" * 70)
        
        print("\n📊 TIMING BREAKDOWN")
        print("-" * 40)
        print(f"  Total inference time:    {report.total_time_ms:>10.2f} ms")
        print(f"  Vision encoding:         {report.vision_time_ms:>10.2f} ms ({report.vision_time_ms/report.total_time_ms*100:.1f}%)" if report.total_time_ms > 0 else "  Vision encoding:         N/A")
        print(f"  LLM prefill:             {report.llm_prefill_time_ms:>10.2f} ms")
        print(f"  LLM decode:              {report.llm_decode_time_ms:>10.2f} ms")
        print(f"  Time to first token:     {report.time_to_first_token_ms:>10.2f} ms")
        
        print("\n🚀 THROUGHPUT")
        print("-" * 40)
        print(f"  Tokens generated:        {report.tokens_generated:>10}")
        print(f"  Tokens/second:           {report.tokens_per_second:>10.1f}")
        
        print("\n💾 MEMORY")
        print("-" * 40)
        print(f"  Memory reads:            {report.total_memory_reads_mb:>10.2f} MB")
        print(f"  Memory writes:           {report.total_memory_writes_mb:>10.2f} MB")
        print(f"  Bandwidth:               {report.memory_bandwidth_gbps:>10.2f} GB/s")
        
        print("\n⚡ COMPUTE")
        print("-" * 40)
        print(f"  Total operations:        {report.total_operations/1e9:>10.2f} GOP")
        print(f"  Performance:             {report.tops:>10.3f} TOPS")
        print(f"  Utilization:             {report.compute_utilization_pct:>10.1f}%")
        
        if report.thermal_samples:
            print("\n🌡️ THERMAL")
            print("-" * 40)
            print(f"  Average temperature:     {report.avg_temperature_c:>10.1f} °C")
            print(f"  Peak temperature:        {report.peak_temperature_c:>10.1f} °C")
            throttle_count = sum(1 for s in report.thermal_samples if s.throttled)
            print(f"  Throttled samples:       {throttle_count:>10} / {len(report.thermal_samples)}")
        
        if report.power_samples:
            print("\n🔋 POWER")
            print("-" * 40)
            print(f"  Average power:           {report.avg_power_w:>10.2f} W")
            print(f"  Peak power:              {report.peak_power_w:>10.2f} W")
            if report.total_time_ms > 0:
                energy_j = report.avg_power_w * (report.total_time_ms / 1000.0)
                print(f"  Energy consumed:         {energy_j:>10.3f} J")
                if report.tokens_generated > 0:
                    j_per_token = energy_j / report.tokens_generated
                    print(f"  Energy/token:            {j_per_token*1000:>10.2f} mJ")
        
        # Token timing distribution
        if report.token_profiles:
            gen_times = [t.generation_time_ms for t in report.token_profiles if t.generation_time_ms > 0]
            if gen_times:
                print("\n⏱️ TOKEN GENERATION TIMES")
                print("-" * 40)
                print(f"  Min:                     {min(gen_times):>10.2f} ms")
                print(f"  Max:                     {max(gen_times):>10.2f} ms")
                print(f"  Mean:                    {np.mean(gen_times):>10.2f} ms")
                print(f"  Std:                     {np.std(gen_times):>10.2f} ms")
                print(f"  P50:                     {np.percentile(gen_times, 50):>10.2f} ms")
                print(f"  P95:                     {np.percentile(gen_times, 95):>10.2f} ms")
                print(f"  P99:                     {np.percentile(gen_times, 99):>10.2f} ms")
        
        print("\n" + "=" * 70)


def compare_profiles(profiles: List[ProfileReport], 
                     labels: Optional[List[str]] = None) -> None:
    """
    Compare multiple profiling reports side-by-side.
    
    Args:
        profiles: List of ProfileReport objects to compare
        labels: Optional labels for each profile
    """
    if not profiles:
        return
    
    if labels is None:
        labels = [f"Profile {i+1}" for i in range(len(profiles))]
    
    # Determine column width
    col_width = max(len(l) for l in labels) + 2
    col_width = max(col_width, 12)
    
    print("\n" + "=" * (20 + col_width * len(profiles)))
    print("PROFILE COMPARISON")
    print("=" * (20 + col_width * len(profiles)))
    
    # Header
    header = f"{'Metric':<20}"
    for label in labels:
        header += f"{label:>{col_width}}"
    print(header)
    print("-" * (20 + col_width * len(profiles)))
    
    # Metrics to compare
    metrics = [
        ("Total time (ms)", lambda p: f"{p.total_time_ms:.2f}"),
        ("Vision time (ms)", lambda p: f"{p.vision_time_ms:.2f}"),
        ("TTFT (ms)", lambda p: f"{p.time_to_first_token_ms:.2f}"),
        ("Tokens/sec", lambda p: f"{p.tokens_per_second:.1f}"),
        ("TOPS", lambda p: f"{p.tops:.3f}"),
        ("Utilization (%)", lambda p: f"{p.compute_utilization_pct:.1f}"),
        ("Avg Power (W)", lambda p: f"{p.avg_power_w:.2f}"),
        ("Avg Temp (°C)", lambda p: f"{p.avg_temperature_c:.1f}"),
    ]
    
    for metric_name, getter in metrics:
        row = f"{metric_name:<20}"
        for profile in profiles:
            row += f"{getter(profile):>{col_width}}"
        print(row)
    
    print("=" * (20 + col_width * len(profiles)))
