# =============================================================================
# SiLens RTL Testbench Makefile (cocotb)
# =============================================================================
# Run RTL simulations using cocotb with Icarus Verilog or Verilator
#
# Usage:
#   make sim                - Run all tests
#   make test-popcount      - Test popcount module
#   make test-ternary-mac   - Test ternary MAC module
#   make test-binary-dot    - Test binary dot product module
#   make test-softmax       - Test softmax approximation
#   make test-gelu          - Test GELU activation
#   make test-layernorm     - Test layer normalization
#   make test-attention     - Test attention components
#   make test-transformer   - Test transformer block
#   make test-vision        - Test vision encoder
#   make test-language      - Test language model
#   make test-pipeline      - Test full pipeline
#   make waves              - Open waveform viewer
#   make clean              - Clean build artifacts
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
TESTS_DIR := $(PROJECT_ROOT)/tests

# Verilog source files by module
RTL_COMMON := \
	$(RTL_DIR)/common/popcount.v \
	$(RTL_DIR)/common/ternary_mac.v \
	$(RTL_DIR)/common/binary_dot_product.v \
	$(RTL_DIR)/common/layer_norm.v \
	$(RTL_DIR)/common/softmax_approx.v \
	$(RTL_DIR)/common/gelu_approx.v

RTL_TOP := $(RTL_DIR)/top/silens_top.v

RTL_ALL := $(RTL_COMMON) $(RTL_TOP)

# cocotb configuration
export COCOTB_RESOLVE_X := ZEROS
export PYTHONPATH := $(TB_DIR):$(TESTS_DIR):$(PROJECT_ROOT):$(PYTHONPATH)

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
sim: test-popcount test-ternary-mac test-binary-dot test-softmax test-gelu test-layernorm

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

## test-ternary-mac: Test ternary MAC module
.PHONY: test-ternary-mac
test-ternary-mac:
	@echo "=============================================="
	@echo "Testing: Ternary MAC Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_attention \
		TOPLEVEL=ternary_mac \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/ternary_mac.v" \
		SIM=$(SIM) \
		TESTCASE="test_ternary_mac_all_positive,test_ternary_mac_all_negative,test_ternary_mac_all_zero,test_ternary_mac_mixed" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-binary-dot: Test binary dot product module
.PHONY: test-binary-dot
test-binary-dot:
	@echo "=============================================="
	@echo "Testing: Binary Dot Product Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_attention \
		TOPLEVEL=binary_dot_product \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/binary_dot_product.v $(RTL_DIR)/common/popcount.v" \
		SIM=$(SIM) \
		TESTCASE="test_binary_dot_all_same,test_binary_dot_all_opposite,test_binary_dot_random" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-softmax: Test softmax approximation module
.PHONY: test-softmax
test-softmax:
	@echo "=============================================="
	@echo "Testing: Softmax Approximation Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_attention \
		TOPLEVEL=softmax_approx \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/softmax_approx.v" \
		SIM=$(SIM) \
		TESTCASE="test_softmax_equal_inputs,test_softmax_dominant_input,test_softmax_golden_comparison" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-gelu: Test GELU activation module
.PHONY: test-gelu
test-gelu:
	@echo "=============================================="
	@echo "Testing: GELU Approximation Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_transformer_block \
		TOPLEVEL=gelu_approx \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/gelu_approx.v" \
		SIM=$(SIM) \
		TESTCASE="test_gelu_key_values,test_gelu_monotonicity,test_gelu_golden" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-layernorm: Test layer normalization module
.PHONY: test-layernorm
test-layernorm:
	@echo "=============================================="
	@echo "Testing: Layer Normalization Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_transformer_block \
		TOPLEVEL=layer_norm \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/layer_norm.v" \
		SIM=$(SIM) \
		TESTCASE="test_layer_norm_sequential,test_layer_norm_constant,test_layer_norm_golden" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-rmsnorm: Test RMS normalization module
.PHONY: test-rmsnorm
test-rmsnorm:
	@echo "=============================================="
	@echo "Testing: RMS Normalization Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_normalization \
		TOPLEVEL=rms_norm \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/rms_norm.v" \
		SIM=$(SIM) \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-power-controller: Test power management controller
.PHONY: test-power-controller
test-power-controller:
	@echo "=============================================="
	@echo "Testing: Power Controller Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_power_controller \
		TOPLEVEL=power_controller \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/power_controller.v" \
		SIM=$(SIM) \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-simd: Test SIMD vector unit
.PHONY: test-simd
test-simd:
	@echo "=============================================="
	@echo "Testing: SIMD Vector Unit"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_simd \
		TOPLEVEL=simd_vector_unit \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/simd_vector_unit.v" \
		SIM=$(SIM) \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-axi: Test AXI interface
.PHONY: test-axi
test-axi:
	@echo "=============================================="
	@echo "Testing: AXI Interface Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_axi \
		TOPLEVEL=axi_interface \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/axi_interface.v" \
		SIM=$(SIM) \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-vit-mlp: Test Vision Transformer MLP module
.PHONY: test-vit-mlp
test-vit-mlp:
	@echo "=============================================="
	@echo "Testing: ViT MLP Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_mlp \
		TOPLEVEL=vit_mlp \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/vision_encoder/vit_mlp.v $(RTL_DIR)/common/gelu_approx.v" \
		SIM=$(SIM) \
		TESTCASE="test_vit_mlp_reset,test_vit_mlp_all_positive_weights,test_vit_mlp_zeros" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-llm-mlp: Test Language Model MLP module (SwiGLU)
.PHONY: test-llm-mlp
test-llm-mlp:
	@echo "=============================================="
	@echo "Testing: LLM MLP Module (SwiGLU)"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_mlp \
		TOPLEVEL=llm_mlp \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/language_model/llm_mlp.v" \
		SIM=$(SIM) \
		TESTCASE="test_llm_mlp_reset,test_llm_mlp_swiglu,test_llm_mlp_throughput" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-projector: Test multimodal projector module
.PHONY: test-projector
test-projector:
	@echo "=============================================="
	@echo "Testing: Multimodal Projector"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_projector \
		TOPLEVEL=projector \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/projector/projector.v" \
		SIM=$(SIM) \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-memory: Test memory modules (activation buffer, KV cache)
.PHONY: test-memory
test-memory:
	@echo "=============================================="
	@echo "Testing: Memory Modules"
	@echo "=============================================="
	@echo "Note: Memory modules need to be implemented in RTL"

## test-attention: Test attention components (all)
.PHONY: test-attention
test-attention: test-softmax test-binary-dot test-ternary-mac
	@echo "All attention tests completed"

## test-transformer: Test transformer block components
.PHONY: test-transformer
test-transformer: test-layernorm test-gelu
	@echo "All transformer tests completed"

## test-vision: Test vision encoder (placeholder - needs RTL)
.PHONY: test-vision
test-vision:
	@echo "=============================================="
	@echo "Testing: Vision Encoder (placeholder)"
	@echo "=============================================="
	@echo "Vision encoder RTL not yet implemented"

## test-language: Test language model (placeholder - needs RTL)
.PHONY: test-language
test-language:
	@echo "=============================================="
	@echo "Testing: Language Model (placeholder)"
	@echo "=============================================="
	@echo "Language model RTL not yet implemented"

## test-pipeline: Test full pipeline (placeholder - needs RTL)
.PHONY: test-pipeline
test-pipeline:
	@echo "=============================================="
	@echo "Testing: Full Pipeline (placeholder)"
	@echo "=============================================="
	@echo "Full pipeline RTL not yet implemented"

## test-e2e-inference: End-to-end inference test through silens_top
.PHONY: test-e2e-inference
test-e2e-inference:
	@echo "=============================================="
	@echo "Testing: End-to-End Inference (silens_top)"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)/e2e
	cd $(BUILD_DIR)/e2e && \
		MODULE=e2e_inference_tb \
		TOPLEVEL=silens_top \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_DIR)/common/popcount.v \
			$(RTL_DIR)/common/ternary_mac.v \
			$(RTL_DIR)/common/binary_dot_product.v \
			$(RTL_DIR)/common/layer_norm.v \
			$(RTL_DIR)/common/softmax_approx.v \
			$(RTL_DIR)/common/gelu_approx.v \
			$(RTL_DIR)/common/rms_norm.v \
			$(RTL_DIR)/common/simd_vector_unit.v \
			$(RTL_DIR)/common/axi_interface.v \
			$(RTL_DIR)/common/power_controller.v \
			$(RTL_DIR)/projector/projector.v \
			$(RTL_DIR)/top/silens_top.v" \
		SIM=$(SIM) \
		COMPILE_ARGS="$(COMPILE_ARGS)" \
		TESTCASE="test_e2e_inference" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim
	@echo "=============================================="
	@echo "E2E Inference Test Complete"
	@echo "=============================================="

## test-e2e-debug: End-to-end debug interface test
.PHONY: test-e2e-debug
test-e2e-debug:
	@echo "=============================================="
	@echo "Testing: E2E Debug Interface"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)/e2e
	cd $(BUILD_DIR)/e2e && \
		MODULE=e2e_inference_tb \
		TOPLEVEL=silens_top \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_ALL)" \
		SIM=$(SIM) \
		COMPILE_ARGS="$(COMPILE_ARGS)" \
		TESTCASE="test_e2e_debug_interface" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-e2e-all: Run all E2E tests
.PHONY: test-e2e-all
test-e2e-all:
	@echo "=============================================="
	@echo "Running All E2E Tests"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)/e2e
	cd $(BUILD_DIR)/e2e && \
		MODULE=e2e_inference_tb \
		TOPLEVEL=silens_top \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_ALL)" \
		SIM=$(SIM) \
		COMPILE_ARGS="$(COMPILE_ARGS)" \
		$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

## test-top: Test top-level module
.PHONY: test-top
test-top:
	@echo "=============================================="
	@echo "Testing: SiLens Top Module"
	@echo "=============================================="
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
		MODULE=test_common \
		TOPLEVEL=silens_top \
		TOPLEVEL_LANG=verilog \
		VERILOG_SOURCES="$(RTL_ALL)" \
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
	@echo "  sim               Run all cocotb tests"
	@echo "  test-popcount     Test popcount module"
	@echo "  test-ternary-mac  Test ternary MAC module"
	@echo "  test-binary-dot   Test binary dot product module"
	@echo "  test-softmax      Test softmax approximation"
	@echo "  test-gelu         Test GELU activation"
	@echo "  test-layernorm    Test layer normalization"
	@echo "  test-attention    Test all attention components"
	@echo "  test-transformer  Test transformer block components"
	@echo "  test-vision       Test vision encoder"
	@echo "  test-language     Test language model"
	@echo "  test-pipeline     Test full pipeline"
	@echo "  test-top          Test top-level module"
	@echo ""
	@echo "End-to-End Inference:"
	@echo "  test-e2e-inference  Run complete E2E inference test"
	@echo "  test-e2e-debug      Test E2E debug interface"
	@echo "  test-e2e-all        Run all E2E tests"
	@echo ""
	@echo "Waveforms:"
	@echo "  waves             Open waveform viewer"
	@echo ""
	@echo "Verification:"
	@echo "  golden            Generate golden reference"
	@echo "  compare           Compare with golden"
	@echo "  regression        Full regression suite"
	@echo "  coverage          Generate coverage report"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean             Clean build artifacts"
	@echo "  clean-all         Clean everything"
	@echo ""
	@echo "Environment Variables:"
	@echo "  SIM=icarus        Use Icarus Verilog (default)"
	@echo "  SIM=verilator     Use Verilator"
	@echo "  SILENS_TEST_IMAGE Path to test image (.npy or image file)"
	@echo "  SILENS_TEST_TOKENS Path to test tokens (.npy)"
	@echo "  SILENS_RESULTS_FILE Path for results JSON output"
