# =============================================================================
# SiLens OpenLane Configuration
# =============================================================================
# OpenLane configuration file for SiLens vision-language AI accelerator
#
# This configuration is optimized for:
#   - SkyWater 130nm PDK (sky130A)
#   - High-density standard cell library (sky130_fd_sc_hd)
#   - Area-optimized synthesis (weight matrices are area-dominant)
#   - Relaxed timing (100 MHz target)
#
# Usage:
#   flow.tcl -design /path/to/synthesis -tag run_name
#
# License: Apache 2.0
# =============================================================================

# =============================================================================
# Design Information
# =============================================================================

set ::env(DESIGN_NAME) "silens_top"
set ::env(DESIGN_IS_CORE) 0

# Verilog source files
set ::env(VERILOG_FILES) [glob -directory $::env(DESIGN_DIR)/../rtl \
    top/*.v \
    common/*.v \
]

# Additional include paths
set ::env(VERILOG_INCLUDE_DIRS) [list \
    $::env(DESIGN_DIR)/../rtl/common \
    $::env(DESIGN_DIR)/../rtl/top \
]

# =============================================================================
# Clock Configuration
# =============================================================================

# Primary clock: 100 MHz (10 ns period)
set ::env(CLOCK_PORT) "clk"
set ::env(CLOCK_PERIOD) "10.0"
set ::env(CLOCK_NET) $::env(CLOCK_PORT)

# PCIe clock: 250 MHz (4 ns period) - handled as separate clock domain
# Note: Multi-clock support requires additional constraints

# Clock uncertainty (jitter + skew)
set ::env(SYNTH_CLOCK_UNCERTAINTY) "0.25"
set ::env(SYNTH_CLOCK_TRANSITION) "0.15"

# =============================================================================
# Die Area Configuration
# =============================================================================

# Target die size for full SiLens accelerator
# For initial synthesis, use a smaller test area
# Full accelerator: ~500 mm² (22,000 x 22,000 µm)
# Test configuration: 2 mm² (1,400 x 1,400 µm)

# Die area [x_min y_min x_max y_max] in µm
# Start with smaller area for faster iteration
set ::env(DIE_AREA) "0 0 1400 1400"

# Core area (inside I/O ring)
set ::env(CORE_AREA) "20 20 1380 1380"

# For full chip synthesis, uncomment:
# set ::env(DIE_AREA) "0 0 22000 22000"
# set ::env(CORE_AREA) "100 100 21900 21900"

# Utilization targets
set ::env(FP_CORE_UTIL) 50
set ::env(PL_TARGET_DENSITY) 0.55

# =============================================================================
# PDK Configuration
# =============================================================================

# PDK variant
set ::env(PDK) "sky130A"
set ::env(STD_CELL_LIBRARY) "sky130_fd_sc_hd"

# Use high-density library for area optimization
set ::env(STD_CELL_LIBRARY_OPT) "sky130_fd_sc_hd"

# =============================================================================
# Synthesis Configuration
# =============================================================================

# Synthesis strategy: area-optimized
# Options: AREA 0-3, DELAY 0-4
set ::env(SYNTH_STRATEGY) "AREA 3"

# Enable area optimization
set ::env(SYNTH_AREA_OPTIMIZATION) 1

# Timing-driven synthesis (still needed for functionality)
set ::env(SYNTH_TIMING_DERATE) 0.05

# Maximum fanout for high-fanout nets
set ::env(SYNTH_MAX_FANOUT) 8

# Buffering
set ::env(SYNTH_BUFFERING) 1
set ::env(SYNTH_SIZING) 1

# Register balancing for better timing
set ::env(SYNTH_SHARE_RESOURCES) 1

# Don't flatten - preserve hierarchy for debug
set ::env(SYNTH_FLAT_TOP) 0

# Read liberty files with DONT_USE cells excluded
set ::env(SYNTH_NO_FLAT) 0

# =============================================================================
# Floorplanning
# =============================================================================

# I/O placement mode
set ::env(FP_IO_MODE) 1

# Pin placement: random (let tool decide)
set ::env(FP_PIN_ORDER_CFG) ""

# Macro placement (if using hard macros)
set ::env(MACRO_PLACEMENT_CFG) ""

# Tap cell insertion
set ::env(FP_TAP_HORIZONTAL_HALO) 10
set ::env(FP_TAP_VERTICAL_HALO) 10

# Endcap cells
set ::env(FP_ENDCAP_CELL) "sky130_fd_sc_hd__decap_4"

# Power planning
set ::env(FP_PDN_CORE_RING) 1
set ::env(FP_PDN_ENABLE_RAILS) 1

# =============================================================================
# Placement
# =============================================================================

# Global placement parameters
set ::env(PL_BASIC_PLACEMENT) 0
set ::env(PL_SKIP_INITIAL_PLACEMENT) 0
set ::env(PL_RANDOM_GLB_PLACEMENT) 0

# Placement optimization
set ::env(PL_ROUTABILITY_DRIVEN) 1
set ::env(PL_TIME_DRIVEN) 1

# Resizer timing optimization
set ::env(PL_RESIZER_DESIGN_OPTIMIZATIONS) 1
set ::env(PL_RESIZER_TIMING_OPTIMIZATIONS) 1
set ::env(PL_RESIZER_BUFFER_INPUT_PORTS) 1
set ::env(PL_RESIZER_BUFFER_OUTPUT_PORTS) 1

# Detailed placement
set ::env(DPL_CELL_PADDING) 4
set ::env(CELL_PAD) 4

# =============================================================================
# Clock Tree Synthesis
# =============================================================================

# CTS parameters
set ::env(CTS_TARGET_SKEW) 200
set ::env(CTS_TOLERANCE) 100
set ::env(CTS_SINK_CLUSTERING_SIZE) 25
set ::env(CTS_SINK_CLUSTERING_MAX_DIAMETER) 50

# CTS root buffer
set ::env(CTS_ROOT_BUFFER) "sky130_fd_sc_hd__clkbuf_16"
set ::env(CTS_CLK_BUFFER_LIST) "sky130_fd_sc_hd__clkbuf_4 sky130_fd_sc_hd__clkbuf_8 sky130_fd_sc_hd__clkbuf_16"
set ::env(CTS_MAX_CAP) 1.5

# =============================================================================
# Routing
# =============================================================================

# Global routing
set ::env(GRT_ADJUSTMENT) 0.3
set ::env(GRT_ALLOW_CONGESTION) 0
set ::env(GRT_OVERFLOW_ITERS) 200

# Detailed routing
set ::env(DRT_OPT_ITERS) 64
set ::env(ROUTING_OPT_ITERS) 64

# Layer usage (metal layers)
set ::env(RT_MIN_LAYER) "met1"
set ::env(RT_MAX_LAYER) "met5"

# Antenna rules
set ::env(DIODE_INSERTION_STRATEGY) 4
set ::env(USE_ARC_ANTENNA_CHECK) 1

# =============================================================================
# Signoff
# =============================================================================

# STA corners
set ::env(STA_REPORT_POWER) 1

# Magic settings
set ::env(MAGIC_WRITE_FULL_LEF) 0
set ::env(MAGIC_DRC_USE_GDS) 1

# LVS settings
set ::env(RUN_LVS) 1
set ::env(LVS_CONNECT_BY_LABEL) 0

# DRC settings
set ::env(RUN_DRC) 1
set ::env(MAGIC_DRC_USE_GDS) 1

# =============================================================================
# Output Configuration
# =============================================================================

# Generate all output formats
set ::env(GENERATE_FINAL_SUMMARY_REPORT) 1

# =============================================================================
# Special Configuration for Weight Matrices
# =============================================================================

# The SiLens design has many constant weight connections.
# These optimizations help with synthesis:

# Don't merge equivalent cells (preserve structure)
# set ::env(SYNTH_NO_EQUIV_CHECK) 1

# Allow synthesis to optimize constant propagation
set ::env(SYNTH_READ_BLACKBOX_LIB) 0

# =============================================================================
# Debug and Logging
# =============================================================================

# Verbose logging
set ::env(PL_DEBUG_REPLACE_FLOW) 0
set ::env(GRT_DEBUG) 0
set ::env(DRT_DEBUG) 0

# Save intermediate results
set ::env(SAVE_NETLIST) 1
set ::env(SAVE_DEF) 1
set ::env(SAVE_GDS) 1
set ::env(SAVE_ODB) 1
set ::env(SAVE_SPEF) 1
set ::env(SAVE_SDF) 1

# =============================================================================
# Custom Scripts (Optional)
# =============================================================================

# Post-synthesis TCL script
# set ::env(SYNTH_POST_SCRIPT) "$::env(DESIGN_DIR)/scripts/post_synth.tcl"

# Post-floorplan TCL script
# set ::env(FP_POST_SCRIPT) "$::env(DESIGN_DIR)/scripts/post_fp.tcl"

# Post-placement TCL script
# set ::env(PL_POST_SCRIPT) "$::env(DESIGN_DIR)/scripts/post_pl.tcl"

# =============================================================================
# Notes
# =============================================================================

# This configuration is designed for iterative development:
#
# 1. Initial runs use small die area for fast turnaround
# 2. Synthesis strategy is area-optimized
# 3. Timing is relaxed (100 MHz is conservative for 130nm)
#
# For production tapeout:
# - Increase die area to full chip size
# - Tighten timing constraints
# - Enable all signoff checks
# - Review and address all DRC/LVS violations
#
# Expected metrics (test configuration):
# - Cell count: ~10K-50K (depending on instantiated weights)
# - Area utilization: 40-60%
# - Timing: ~100 MHz achievable
# - Power: TBD (depends on switching activity)
