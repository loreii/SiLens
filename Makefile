# =============================================================================
# SiLens Makefile
# =============================================================================

.PHONY: all setup model pdk sim synth clean help

# Project paths
PROJECT_ROOT := $(shell pwd)
RTL_DIR := $(PROJECT_ROOT)/rtl
MODEL_DIR := $(PROJECT_ROOT)/model
PDK_DIR := $(PROJECT_ROOT)/pdk
SIM_DIR := $(PROJECT_ROOT)/rtl/tb
SYNTH_DIR := $(PROJECT_ROOT)/synthesis

# Python
PYTHON := python3
PIP := pip3

# Default target
all: help

# =============================================================================
# Setup
# =============================================================================

## setup: Full project setup (dependencies + model + PDK)
setup: deps model pdk
	@echo "✓ Setup complete!"

## deps: Install Python dependencies
deps:
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt

## model: Download SmolVLM-256M model
model:
	@echo "Downloading SmolVLM-256M model..."
	$(PYTHON) tools/download_model.py --verify

## pdk: Setup SkyWater PDK (minimal)
pdk:
	@echo "Setting up SkyWater PDK..."
	./tools/setup_pdk.sh --minimal

## pdk-full: Setup full SkyWater PDK
pdk-full:
	@echo "Setting up full SkyWater PDK..."
	./tools/setup_pdk.sh --full

# =============================================================================
# Model Conversion
# =============================================================================

## convert: Convert model weights to Verilog
convert:
	@echo "Converting model weights to Verilog..."
	$(PYTHON) model/conversion/convert_weights.py

## quantize: Quantize model to 1-bit weights
quantize:
	@echo "Quantizing model..."
	$(PYTHON) model/conversion/quantize.py

## validate: Validate quantized model accuracy
validate:
	@echo "Validating model accuracy..."
	$(PYTHON) model/validation/validate.py

## weights-to-verilog: Convert quantized weights to Verilog modules
weights-to-verilog:
	@echo "Converting quantized weights to Verilog..."
	$(PYTHON) model/conversion/weights_to_verilog.py \
		-w $(MODEL_DIR)/weights/quantized \
		-o $(RTL_DIR)
	@echo "✓ Verilog weight modules generated in $(RTL_DIR)/"

## weights-to-verilog-vision: Convert only vision encoder weights
weights-to-verilog-vision:
	@echo "Converting vision encoder weights to Verilog..."
	$(PYTHON) model/conversion/weights_to_verilog.py \
		-w $(MODEL_DIR)/weights/quantized \
		-o $(RTL_DIR) \
		-f vision

## weights-to-verilog-llm: Convert only language model weights
weights-to-verilog-llm:
	@echo "Converting language model weights to Verilog..."
	$(PYTHON) model/conversion/weights_to_verilog.py \
		-w $(MODEL_DIR)/weights/quantized \
		-o $(RTL_DIR) \
		-f language

# =============================================================================
# Simulation
# =============================================================================

## sim: Run RTL simulation (cocotb)
sim:
	@echo "Running RTL simulation..."
	cd $(SIM_DIR) && $(MAKE) sim

## sim-vision: Simulate vision encoder only
sim-vision:
	@echo "Simulating vision encoder..."
	cd $(SIM_DIR) && $(MAKE) sim-vision

## sim-llm: Simulate language model only
sim-llm:
	@echo "Simulating language model..."
	cd $(SIM_DIR) && $(MAKE) sim-llm

## waves: Open waveform viewer
waves:
	@echo "Opening waveform viewer..."
	gtkwave $(SIM_DIR)/dump.vcd &

# =============================================================================
# Synthesis
# =============================================================================

## synth: Run OpenLane synthesis
synth:
	@echo "Running synthesis..."
	cd $(SYNTH_DIR) && $(MAKE) synth

## synth-docker: Run synthesis via Docker
synth-docker:
	@echo "Running synthesis in Docker..."
	cd $(SYNTH_DIR) && $(MAKE) synth-docker

## synth-quick: Quick synthesis (skip routing)
synth-quick:
	@echo "Running quick synthesis..."
	cd $(SYNTH_DIR) && $(MAKE) synth-quick

## synth-lint: Lint Verilog before synthesis
synth-lint:
	@echo "Linting Verilog..."
	cd $(SYNTH_DIR) && $(MAKE) lint-iverilog

## reports: Generate synthesis reports
reports:
	@echo "Generating reports..."
	cd $(SYNTH_DIR) && $(MAKE) reports

## metrics: Display synthesis metrics
metrics:
	@echo "Displaying metrics..."
	cd $(SYNTH_DIR) && $(MAKE) metrics

# =============================================================================
# Testing
# =============================================================================

## test: Run all tests
test:
	@echo "Running tests..."
	pytest tests/ -v

## test-model: Test model conversion
test-model:
	@echo "Testing model conversion..."
	pytest tests/test_model.py -v

## test-rtl: Test RTL modules
test-rtl:
	@echo "Testing RTL modules..."
	pytest tests/test_rtl.py -v

## test-quick: Quick test (essential tests only)
test-quick:
	@echo "Running quick tests..."
	pytest tests/ -v -x --timeout=60 \
		--ignore=tests/test_model_full.py \
		--ignore=tests/test_synthesis.py

# =============================================================================
# CI/Verification
# =============================================================================

## ci: Run full CI pipeline locally
ci: lint test-quick verilog-lint
	@echo "✓ CI pipeline passed"

## verilog-lint: Lint Verilog files with iverilog
verilog-lint:
	@echo "Checking Verilog syntax..."
	@VERILOG_FILES=$$(find $(RTL_DIR) -name "*.v" -not -name "*_tb.v" -type f); \
	if [ -z "$$VERILOG_FILES" ]; then \
		echo "No Verilog files found"; \
	else \
		iverilog -g2012 -Wall \
			-I$(RTL_DIR)/common \
			-I$(RTL_DIR)/top \
			-o /dev/null \
			$$VERILOG_FILES && echo "✓ Verilog syntax OK"; \
	fi

## verilog-sim: Run Verilog simulation tests
verilog-sim:
	@echo "Running Verilog simulations..."
	@if grep -q "SIMULATION" $(RTL_DIR)/common/popcount.v 2>/dev/null; then \
		cd $(RTL_DIR)/common && \
		iverilog -g2012 -DSIMULATION -o popcount_test popcount.v && \
		vvp popcount_test; \
	else \
		echo "No simulation found"; \
	fi

# =============================================================================
# Documentation
# =============================================================================

## docs: Build documentation
docs:
	@echo "Building documentation..."
	mkdocs build

## docs-serve: Serve documentation locally
docs-serve:
	@echo "Serving documentation..."
	mkdocs serve

# =============================================================================
# Utilities
# =============================================================================

## clean: Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf rtl/**/*.vcd rtl/**/work/
	rm -rf synthesis/runs/
	rm -rf __pycache__ **/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete

## clean-model: Remove downloaded model (large files)
clean-model:
	@echo "Removing model files..."
	rm -rf model/smolvlm-256m/

## clean-pdk: Remove PDK (very large)
clean-pdk:
	@echo "Removing PDK..."
	rm -rf pdk/skywater-pdk/

## lint: Run linters
lint:
	@echo "Running linters..."
	black --check .
	flake8 .

## format: Format code
format:
	@echo "Formatting code..."
	black .
	isort .

## info: Print project info
info:
	@echo "SiLens Project Info"
	@echo "==================="
	@echo "Project root: $(PROJECT_ROOT)"
	@echo "RTL directory: $(RTL_DIR)"
	@echo "Model directory: $(MODEL_DIR)"
	@echo "PDK directory: $(PDK_DIR)"
	@echo ""
	@echo "Python: $(shell $(PYTHON) --version 2>&1)"
	@echo ""
	@if [ -d "$(PDK_DIR)/skywater-pdk" ]; then \
		echo "PDK: Installed"; \
	else \
		echo "PDK: Not installed (run 'make pdk')"; \
	fi
	@if [ -d "$(MODEL_DIR)/smolvlm-256m" ]; then \
		echo "Model: Downloaded"; \
	else \
		echo "Model: Not downloaded (run 'make model')"; \
	fi

# =============================================================================
# Help
# =============================================================================

## help: Show this help message
help:
	@echo "SiLens - Open Source Vision-Language AI Accelerator"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' $(MAKEFILE_LIST) | \
		sed -e 's/## //' | \
		awk -F': ' '{printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make setup      # Full setup"
	@echo "  make model      # Download model only"
	@echo "  make sim        # Run simulation"
	@echo "  make test       # Run tests"
