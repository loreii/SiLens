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
	docker run --rm -v $(PROJECT_ROOT):/work efabless/openlane:latest \
		bash -c "cd /work/synthesis && make synth"

## reports: Generate synthesis reports
reports:
	@echo "Generating reports..."
	cd $(SYNTH_DIR) && $(MAKE) reports

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
