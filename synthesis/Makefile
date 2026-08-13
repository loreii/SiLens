# =============================================================================
# SiLens Synthesis Makefile
# =============================================================================
# OpenLane-based ASIC synthesis flow for SiLens accelerator
#
# This Makefile provides targets for running the complete RTL-to-GDSII flow
# using the OpenLane toolchain with the SkyWater 130nm PDK.
#
# Usage:
#   make synth          # Run full synthesis flow
#   make synth-docker   # Run synthesis in Docker container
#   make reports        # Generate synthesis reports
#   make clean          # Clean synthesis artifacts
#
# Prerequisites:
#   - OpenLane installed (or Docker)
#   - SkyWater 130nm PDK
#   - Verilog source files in ../rtl/
#
# License: Apache 2.0
# =============================================================================

.PHONY: all synth synth-docker reports clean help
.PHONY: lint drc lvs sta

# =============================================================================
# Configuration
# =============================================================================

# Project paths
PROJECT_ROOT := $(shell cd .. && pwd)
RTL_DIR := $(PROJECT_ROOT)/rtl
SYNTH_DIR := $(shell pwd)
PDK_ROOT := $(PROJECT_ROOT)/pdk

# OpenLane paths (adjust based on installation)
OPENLANE_ROOT ?= $(HOME)/openlane
OPENLANE_TAG ?= latest

# PDK configuration
PDK ?= sky130A
STD_CELL_LIBRARY ?= sky130_fd_sc_hd
STD_CELL_LIBRARY_OPT ?= sky130_fd_sc_hd

# Design configuration
DESIGN_NAME := silens_top
DESIGN_CONFIG := $(SYNTH_DIR)/config.tcl
VERILOG_FILES := $(shell find $(RTL_DIR) -name "*.v" -not -name "*_tb.v")

# Output directories
RUNS_DIR := $(SYNTH_DIR)/runs
REPORTS_DIR := $(SYNTH_DIR)/reports
LOGS_DIR := $(SYNTH_DIR)/logs

# Docker configuration
DOCKER_IMAGE := efabless/openlane:$(OPENLANE_TAG)
DOCKER_MOUNT := -v $(PROJECT_ROOT):/work -v $(PDK_ROOT):/pdk

# Timestamp for run naming
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
RUN_TAG ?= $(TIMESTAMP)

# =============================================================================
# Default target
# =============================================================================

all: help

# =============================================================================
# Synthesis Targets
# =============================================================================

## synth: Run OpenLane synthesis flow (native installation)
synth: check-env prepare
	@echo "========================================"
	@echo "Running OpenLane synthesis..."
	@echo "Design: $(DESIGN_NAME)"
	@echo "Run tag: $(RUN_TAG)"
	@echo "========================================"
	cd $(OPENLANE_ROOT) && \
		./flow.tcl -design $(SYNTH_DIR) -tag $(RUN_TAG)
	@echo "✓ Synthesis complete"
	@echo "Results: $(RUNS_DIR)/$(RUN_TAG)"

## synth-docker: Run OpenLane synthesis in Docker container
synth-docker: check-docker prepare
	@echo "========================================"
	@echo "Running OpenLane synthesis (Docker)..."
	@echo "Design: $(DESIGN_NAME)"
	@echo "Run tag: $(RUN_TAG)"
	@echo "========================================"
	docker run --rm \
		$(DOCKER_MOUNT) \
		-e PDK_ROOT=/pdk \
		-e PDK=$(PDK) \
		$(DOCKER_IMAGE) \
		/bin/bash -c "cd /work/synthesis && \
			flow.tcl -design . -tag $(RUN_TAG)"
	@echo "✓ Synthesis complete"
	@echo "Results: $(RUNS_DIR)/$(RUN_TAG)"

## synth-interactive: Run OpenLane in interactive mode
synth-interactive: check-docker prepare
	@echo "Starting interactive OpenLane session..."
	docker run -it --rm \
		$(DOCKER_MOUNT) \
		-e PDK_ROOT=/pdk \
		-e PDK=$(PDK) \
		$(DOCKER_IMAGE) \
		/bin/bash

## synth-quick: Quick synthesis (skip routing for fast iteration)
synth-quick: check-docker prepare
	@echo "Running quick synthesis (synth + floorplan only)..."
	docker run --rm \
		$(DOCKER_MOUNT) \
		-e PDK_ROOT=/pdk \
		-e PDK=$(PDK) \
		$(DOCKER_IMAGE) \
		/bin/bash -c "cd /work/synthesis && \
			flow.tcl -design . -tag $(RUN_TAG)_quick -to floorplan"

# =============================================================================
# Verification Targets
# =============================================================================

## lint: Run Verilog linting with Verilator
lint:
	@echo "Running Verilog lint..."
	@mkdir -p $(LOGS_DIR)
	verilator --lint-only -Wall \
		-I$(RTL_DIR)/common \
		-I$(RTL_DIR)/top \
		$(RTL_DIR)/top/silens_top.v \
		2>&1 | tee $(LOGS_DIR)/lint.log
	@echo "✓ Lint complete - see $(LOGS_DIR)/lint.log"

## lint-iverilog: Run syntax check with Icarus Verilog
lint-iverilog:
	@echo "Running Icarus Verilog syntax check..."
	@mkdir -p $(LOGS_DIR)
	iverilog -g2012 -Wall \
		-I$(RTL_DIR)/common \
		-I$(RTL_DIR)/top \
		-o /dev/null \
		$(VERILOG_FILES) \
		2>&1 | tee $(LOGS_DIR)/iverilog_lint.log
	@echo "✓ Syntax check complete"

## drc: Run Design Rule Check (requires completed synthesis run)
drc:
	@echo "Running DRC..."
	@if [ ! -d "$(RUNS_DIR)" ]; then \
		echo "Error: No synthesis runs found. Run 'make synth' first."; \
		exit 1; \
	fi
	$(eval LATEST_RUN := $(shell ls -t $(RUNS_DIR) | head -1))
	@echo "Using run: $(LATEST_RUN)"
	# DRC is run automatically during flow, check results:
	@cat $(RUNS_DIR)/$(LATEST_RUN)/reports/signoff/*drc*.rpt || \
		echo "No DRC report found"

## lvs: Run Layout vs Schematic check
lvs:
	@echo "Running LVS..."
	$(eval LATEST_RUN := $(shell ls -t $(RUNS_DIR) | head -1))
	@echo "Using run: $(LATEST_RUN)"
	@cat $(RUNS_DIR)/$(LATEST_RUN)/reports/signoff/*lvs*.rpt || \
		echo "No LVS report found"

## sta: Run Static Timing Analysis
sta:
	@echo "Running STA..."
	$(eval LATEST_RUN := $(shell ls -t $(RUNS_DIR) | head -1))
	@echo "Using run: $(LATEST_RUN)"
	@cat $(RUNS_DIR)/$(LATEST_RUN)/reports/signoff/*sta*.rpt || \
		echo "No STA report found"

# =============================================================================
# Reports
# =============================================================================

## reports: Generate comprehensive synthesis reports
reports: check-runs
	@echo "========================================"
	@echo "Generating synthesis reports..."
	@echo "========================================"
	@mkdir -p $(REPORTS_DIR)
	$(eval LATEST_RUN := $(shell ls -t $(RUNS_DIR) | head -1))
	@echo "Using run: $(LATEST_RUN)"
	
	@echo "\n=== Area Report ===" > $(REPORTS_DIR)/summary.txt
	@cat $(RUNS_DIR)/$(LATEST_RUN)/reports/synthesis/*area*.rpt >> $(REPORTS_DIR)/summary.txt 2>/dev/null || true
	
	@echo "\n=== Timing Report ===" >> $(REPORTS_DIR)/summary.txt
	@cat $(RUNS_DIR)/$(LATEST_RUN)/reports/synthesis/*timing*.rpt >> $(REPORTS_DIR)/summary.txt 2>/dev/null || true
	
	@echo "\n=== Power Report ===" >> $(REPORTS_DIR)/summary.txt
	@cat $(RUNS_DIR)/$(LATEST_RUN)/reports/synthesis/*power*.rpt >> $(REPORTS_DIR)/summary.txt 2>/dev/null || true
	
	@echo "\n=== Utilization ===" >> $(REPORTS_DIR)/summary.txt
	@cat $(RUNS_DIR)/$(LATEST_RUN)/reports/floorplan/*utilization*.rpt >> $(REPORTS_DIR)/summary.txt 2>/dev/null || true
	
	# Copy key files
	@cp $(RUNS_DIR)/$(LATEST_RUN)/reports/metrics.csv $(REPORTS_DIR)/ 2>/dev/null || true
	@cp $(RUNS_DIR)/$(LATEST_RUN)/results/final/gds/*.gds $(REPORTS_DIR)/ 2>/dev/null || true
	
	@echo "✓ Reports generated in $(REPORTS_DIR)/"
	@echo ""
	@cat $(REPORTS_DIR)/summary.txt

## metrics: Display key synthesis metrics
metrics: check-runs
	@echo "========================================"
	@echo "Synthesis Metrics"
	@echo "========================================"
	$(eval LATEST_RUN := $(shell ls -t $(RUNS_DIR) | head -1))
	@if [ -f "$(RUNS_DIR)/$(LATEST_RUN)/reports/metrics.csv" ]; then \
		cat $(RUNS_DIR)/$(LATEST_RUN)/reports/metrics.csv | \
		awk -F',' 'NR==1{for(i=1;i<=NF;i++)h[i]=$$i} \
			NR>1{for(i=1;i<=NF;i++)printf "%s: %s\n",h[i],$$i}'; \
	else \
		echo "No metrics file found"; \
	fi

# =============================================================================
# Utility Targets
# =============================================================================

## prepare: Prepare synthesis environment
prepare:
	@echo "Preparing synthesis environment..."
	@mkdir -p $(RUNS_DIR)
	@mkdir -p $(REPORTS_DIR)
	@mkdir -p $(LOGS_DIR)
	@echo "Source files:"
	@for f in $(VERILOG_FILES); do echo "  $$f"; done | head -20
	@if [ $$(echo "$(VERILOG_FILES)" | wc -w) -gt 20 ]; then \
		echo "  ... and $$(($$(echo "$(VERILOG_FILES)" | wc -w) - 20)) more"; \
	fi

## check-env: Verify OpenLane environment
check-env:
	@echo "Checking OpenLane environment..."
	@if [ ! -d "$(OPENLANE_ROOT)" ]; then \
		echo "Error: OPENLANE_ROOT not found: $(OPENLANE_ROOT)"; \
		echo "Set OPENLANE_ROOT or use 'make synth-docker'"; \
		exit 1; \
	fi
	@echo "✓ OpenLane found at $(OPENLANE_ROOT)"

## check-docker: Verify Docker is available
check-docker:
	@echo "Checking Docker..."
	@docker --version > /dev/null 2>&1 || \
		(echo "Error: Docker not found. Install Docker or use native OpenLane." && exit 1)
	@echo "✓ Docker available"
	@docker pull $(DOCKER_IMAGE) > /dev/null 2>&1 || \
		echo "Warning: Could not pull latest image, using cached version"

## check-runs: Verify synthesis runs exist
check-runs:
	@if [ ! -d "$(RUNS_DIR)" ] || [ -z "$$(ls -A $(RUNS_DIR) 2>/dev/null)" ]; then \
		echo "Error: No synthesis runs found in $(RUNS_DIR)"; \
		echo "Run 'make synth' or 'make synth-docker' first"; \
		exit 1; \
	fi

## list-runs: List all synthesis runs
list-runs:
	@echo "Synthesis runs:"
	@if [ -d "$(RUNS_DIR)" ]; then \
		ls -lt $(RUNS_DIR) | tail -n +2; \
	else \
		echo "  No runs found"; \
	fi

## view-gds: Open GDS in KLayout (if installed)
view-gds: check-runs
	$(eval LATEST_RUN := $(shell ls -t $(RUNS_DIR) | head -1))
	$(eval GDS_FILE := $(shell ls $(RUNS_DIR)/$(LATEST_RUN)/results/final/gds/*.gds 2>/dev/null | head -1))
	@if [ -n "$(GDS_FILE)" ]; then \
		echo "Opening $(GDS_FILE) in KLayout..."; \
		klayout $(GDS_FILE) &; \
	else \
		echo "No GDS file found"; \
	fi

# =============================================================================
# Cleanup
# =============================================================================

## clean: Remove all synthesis artifacts
clean:
	@echo "Cleaning synthesis artifacts..."
	rm -rf $(RUNS_DIR)
	rm -rf $(REPORTS_DIR)
	rm -rf $(LOGS_DIR)
	rm -rf *.log
	@echo "✓ Clean complete"

## clean-runs: Remove synthesis runs (keep reports)
clean-runs:
	@echo "Cleaning synthesis runs..."
	rm -rf $(RUNS_DIR)
	@echo "✓ Runs cleaned"

## clean-reports: Remove generated reports
clean-reports:
	@echo "Cleaning reports..."
	rm -rf $(REPORTS_DIR)
	@echo "✓ Reports cleaned"

# =============================================================================
# Help
# =============================================================================

## help: Show this help message
help:
	@echo "SiLens Synthesis Makefile"
	@echo "========================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Synthesis:"
	@grep -E '^## (synth|lint)' $(MAKEFILE_LIST) | \
		sed -e 's/## //' | \
		awk -F': ' '{printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Verification:"
	@grep -E '^## (drc|lvs|sta)' $(MAKEFILE_LIST) | \
		sed -e 's/## //' | \
		awk -F': ' '{printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Reports:"
	@grep -E '^## (reports|metrics)' $(MAKEFILE_LIST) | \
		sed -e 's/## //' | \
		awk -F': ' '{printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Utilities:"
	@grep -E '^## (prepare|check|list|view|clean)' $(MAKEFILE_LIST) | \
		sed -e 's/## //' | \
		awk -F': ' '{printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment Variables:"
	@echo "  OPENLANE_ROOT   Path to OpenLane installation"
	@echo "  OPENLANE_TAG    Docker image tag (default: latest)"
	@echo "  PDK             PDK to use (default: sky130A)"
	@echo "  RUN_TAG         Custom run tag (default: timestamp)"
	@echo ""
	@echo "Examples:"
	@echo "  make synth-docker          # Run synthesis in Docker"
	@echo "  make synth RUN_TAG=test1   # Named synthesis run"
	@echo "  make reports               # Generate reports from latest run"
	@echo "  make lint                  # Lint Verilog files"
