# =============================================================================
# SiLens RTL Testbench Makefile (cocotb)
# =============================================================================
# Run RTL simulations using cocotb with Icarus Verilog or Verilator
#
# Usage:
#   make sim              - Run all tests
#   make test-popcount    - Test popcount module
#   make test-top         - Test top-level module
#   make waves            - Open waveform viewer
#   make clean            - Clean build artifacts
# =============================================================================

# Simulator selection (iverilog, verilator)
SIM ?= icarus

# Python executable
PYTHON ?= python3

# Project paths
PROJECT_ROOT := $(shell dirname $(shell dirname $(shell pwd)))
RTL_DIR := $(PROJECT_ROOT)/rtl
TB_DIR := $(PROJECT_ROOT)/rtl/tb
BUILD_DIR := $(TB_DIR)/build

# Verilog source files
RTL_SOURCES := \
	$(RTL_DIR)/common/popcount.v \
	$(RTL_DIR)/top/silens_top.v

# cocotb configuration
export COCOTB_RESOLVE_X := ZEROS
export PYTHONPATH := $(TB_DIR):$(PROJECT_ROOT)/tests:$(PYTHONPATH)

# Simulator-specific settings
ifeq ($(SIM),icarus)
	COMPILE_ARGS += -g2012
	SIM_ARGS += -fst
endif

ifeq ($(SIM),verilator)
	COMPILE_ARGS += --trace --trace-fst
	EXTRA_ARGS += --timing
endif

# Default target
.PHONY: all
all: sim

# =============================================================================
# Test Targets
# =============================================================================

## sim: Run all cocotb tests
.PHONY: sim
sim: test-popcount

## test-popcount: Test popcount module
.PHONY: test-popcount
test-popcount:
	@echo "=============================================="
	@echo "Testing: Popcount Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_common \
		TOPLEVEL=popcount \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/popcount.v" \
		SIM=$(SIM) \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-top: Test top-level module (placeholder)
.PHONY: test-top
test-top:
	@echo "=============================================="
	@echo "Testing: SiLens Top Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_top \
		TOPLEVEL=silens_top \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_SOURCES)" \
		SIM=$(SIM) \
		COMPILE_ARGS="$(COMPILE_ARGS)" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

# =============================================================================
# Waveform Viewing
# =============================================================================

## waves: Open waveform viewer (GTKWave)
.PHONY: waves
waves:
	@if [ -f $(BUILD_DIR)/dump.fst ]; then \
		gtkwave $(BUILD_DIR)/dump.fst &; \
	elif [ -f $(BUILD_DIR)/dump.vcd ]; then \
		gtkwave $(BUILD_DIR)/dump.vcd &; \
	else \
		echo "No waveform file found. Run simulation first."; \
	fi

## waves-popcount: View popcount test waveforms
.PHONY: waves-popcount
waves-popcount:
	@gtkwave $(BUILD_DIR)/popcount.fst &

# =============================================================================
# Golden Model Comparison
# =============================================================================

## golden: Generate golden reference outputs
.PHONY: golden
golden:
	@echo "Generating golden reference outputs..."
	$(PYTHON) golden_gen.py

## compare: Compare simulation results with golden
.PHONY: compare
compare:
	@echo "Comparing with golden reference..."
	$(PYTHON) compare_golden.py

# =============================================================================
# Regression Testing
# =============================================================================

## regression: Run full regression suite
.PHONY: regression
regression: test-popcount test-top
	@echo "=============================================="
	@echo "Regression Complete"
	@echo "=============================================="

## coverage: Generate coverage report
.PHONY: coverage
coverage:
	@echo "Generating coverage report..."
	verilator_coverage --annotate coverage_annotated coverage.dat

# =============================================================================
# Cleanup
# =============================================================================

## clean: Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf __pycache__ .pytest_cache
	rm -rf *.vcd *.fst *.xml
	rm -rf sim_build results.xml
	rm -rf *.o *.d

## clean-all: Clean everything including generated files
.PHONY: clean-all
clean-all: clean
	rm -rf golden_outputs/

# =============================================================================
# Help
# =============================================================================

## help: Show this help message
.PHONY: help
help:
	@echo "SiLens RTL Testbench"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Simulation:"
	@echo "  sim            Run all cocotb tests"
	@echo "  test-popcount  Test popcount module"
	@echo "  test-top       Test top-level module"
	@echo ""
	@echo "Waveforms:"
	@echo "  waves          Open waveform viewer"
	@echo ""
	@echo "Verification:"
	@echo "  golden         Generate golden reference"
	@echo "  compare        Compare with golden"
	@echo "  regression     Full regression suite"
	@echo "  coverage       Generate coverage report"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Clean build artifacts"
	@echo "  clean-all      Clean everything"
	@echo ""
	@echo "Environment Variables:"
	@echo "  SIM=icarus     Use Icarus Verilog (default)"
	@echo "  SIM=verilator  Use Verilator"
