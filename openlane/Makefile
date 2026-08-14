# =============================================================================
# SiLens Hierarchical Synthesis Makefile
# =============================================================================
# Bottom-up synthesis flow: Level 1 → Level 2 → Level 3 → Level 4
#
# Usage:
#   make level1           # Synthesize all Level 1 primitives
#   make level2           # Synthesize Level 2 blocks (requires Level 1)
#   make level3           # Synthesize Level 3 subsystems
#   make level4           # Top integration
#   make all              # Full flow
#   make clean            # Clean all runs
#
# Requirements:
#   - OpenLane installed and in PATH
#   - SKY130 PDK installed
# =============================================================================

OPENLANE_ROOT ?= $(HOME)/OpenLane
PDK_ROOT ?= $(HOME)/pdk
PDK ?= sky130A

# Flow command
FLOW = cd $(OPENLANE_ROOT) && ./flow.tcl -design

# =============================================================================
# Level 1: Compute Primitives
# =============================================================================

LEVEL1_BLOCKS = \
	level1/ternary_mac_array_64 \
	level1/rms_norm_block

.PHONY: level1
level1: $(LEVEL1_BLOCKS)

level1/ternary_mac_array_64: level1/ternary_mac_array_64/config.json
	@echo "=== Synthesizing ternary_mac_array_64 ==="
	$(FLOW) $(CURDIR)/$@ -tag run1
	@echo "=== DRC Check ==="
	@if [ -f $(CURDIR)/$@/runs/run1/reports/signoff/drc.rpt ]; then \
		echo "DRC violations:"; \
		grep -c "violation" $(CURDIR)/$@/runs/run1/reports/signoff/drc.rpt || echo "0"; \
	fi

level1/rms_norm_block: level1/rms_norm_block/config.json
	@echo "=== Synthesizing rms_norm_block ==="
	$(FLOW) $(CURDIR)/$@ -tag run1

.PHONY: level1-check
level1-check:
	@echo "=== Checking Level 1 DRC status ==="
	@for block in $(LEVEL1_BLOCKS); do \
		echo "$$block:"; \
		if [ -f $(CURDIR)/$$block/runs/run1/reports/signoff/drc.rpt ]; then \
			violations=$$(grep -c "violation" $(CURDIR)/$$block/runs/run1/reports/signoff/drc.rpt 2>/dev/null || echo "0"); \
			echo "  DRC violations: $$violations"; \
		else \
			echo "  Not synthesized yet"; \
		fi; \
	done

# =============================================================================
# Level 1: Additional blocks
# =============================================================================

LEVEL1_BLOCKS += \
	level1/layer_norm_block \
	level1/softmax_unit \
	level1/silu_unit \
	level1/attention_head \
	level1/mlp_block

level1/layer_norm_block: level1/layer_norm_block/config.json
	@echo "=== Synthesizing layer_norm_block ==="
	$(FLOW) $(CURDIR)/$@ -tag run1

level1/softmax_unit: level1/softmax_unit/config.json
	@echo "=== Synthesizing softmax_unit ==="
	$(FLOW) $(CURDIR)/$@ -tag run1

level1/silu_unit: level1/silu_unit/config.json
	@echo "=== Synthesizing silu_unit ==="
	$(FLOW) $(CURDIR)/$@ -tag run1

level1/attention_head: level1/attention_head/config.json
	@echo "=== Synthesizing attention_head ==="
	$(FLOW) $(CURDIR)/$@ -tag run1

level1/mlp_block: level1/mlp_block/config.json
	@echo "=== Synthesizing mlp_block ==="
	$(FLOW) $(CURDIR)/$@ -tag run1

# =============================================================================
# Level 2: Functional Blocks
# =============================================================================

LEVEL2_BLOCKS = \
	level2/transformer_block_llm \
	level2/transformer_block_vision \
	level2/projector_block \
	level2/embedding_block

.PHONY: level2
level2: level1 $(LEVEL2_BLOCKS)

level2/transformer_block_llm: level2/transformer_block_llm/config.json level1
	@echo "=== Synthesizing transformer_block_llm (13mm²) ==="
	@echo "Note: Requires Level 1 macros (rms_norm, attention_head, mlp_block)"
	$(FLOW) $(CURDIR)/$@ -tag run1

level2/transformer_block_vision: level2/transformer_block_vision/config.json level1
	@echo "=== Synthesizing transformer_block_vision (20mm²) ==="
	@echo "Note: Requires Level 1 macros (layer_norm, attention_head, mlp_block)"
	$(FLOW) $(CURDIR)/$@ -tag run1

level2/projector_block: level2/projector_block/config.json
	@echo "=== Synthesizing projector_block (10mm²) ==="
	@echo "Note: Vision-to-LLM projection (768→1152→576)"
	$(FLOW) $(CURDIR)/$@ -tag run1

level2/embedding_block: level2/embedding_block/config.json
	@echo "=== Synthesizing embedding_block (15mm²) ==="
	@echo "Note: Token + Position embeddings for LLM"
	$(FLOW) $(CURDIR)/$@ -tag run1

# =============================================================================
# Level 3: Subsystems
# =============================================================================

LEVEL3_BLOCKS = \
	level3/vision_subsystem \
	level3/llm_subsystem \
	level3/memory_subsystem \
	level3/io_subsystem

.PHONY: level3
level3: level2 $(LEVEL3_BLOCKS)

level3/vision_subsystem: level3/vision_subsystem/config.json level2
	@echo "=== Synthesizing vision_subsystem (250mm²) ==="
	@echo "Note: 12× transformer_block_vision"
	$(FLOW) $(CURDIR)/$@ -tag run1

level3/llm_subsystem: level3/llm_subsystem/config.json level2
	@echo "=== Synthesizing llm_subsystem (400mm²) ==="
	@echo "Note: 30× transformer_block_llm"
	$(FLOW) $(CURDIR)/$@ -tag run1

level3/memory_subsystem: level3/memory_subsystem/config.json
	@echo "=== Synthesizing memory_subsystem (50mm²) ==="
	@echo "Note: DDR3 PHY + Controller"
	$(FLOW) $(CURDIR)/$@ -tag run1

level3/io_subsystem: level3/io_subsystem/config.json
	@echo "=== Synthesizing io_subsystem (30mm²) ==="
	@echo "Note: Host Interface + SPI + GPIO"
	$(FLOW) $(CURDIR)/$@ -tag run1

# =============================================================================
# Level 4: Top Integration
# =============================================================================

.PHONY: level4
level4: level3
	@echo "=== Synthesizing silens_soc_top (800mm² - Level 4) ==="
	@echo "Note: Requires Level 3 macros (vision, llm, memory, io subsystems)"
	$(FLOW) $(CURDIR)/level4/silens_soc -tag run1

# =============================================================================
# Full flow
# =============================================================================

.PHONY: all
all: level1 level2 level3 level4
	@echo "=== Full hierarchical synthesis complete ==="

# =============================================================================
# Utilities
# =============================================================================

.PHONY: clean
clean:
	rm -rf level1/*/runs
	rm -rf level2/*/runs
	rm -rf level3/*/runs
	rm -rf level4/*/runs
	@echo "Cleaned all synthesis runs"

.PHONY: status
status:
	@echo "=== Synthesis Status ==="
	@echo ""
	@echo "Level 1 Primitives:"
	@for block in $(LEVEL1_BLOCKS); do \
		if [ -d "$$block/runs/run1" ]; then \
			echo "  ✓ $$block"; \
		else \
			echo "  ○ $$block (not started)"; \
		fi; \
	done
	@echo ""
	@echo "Level 2 Blocks:"
	@for block in $(LEVEL2_BLOCKS); do \
		if [ -d "$$block/runs/run1" ]; then \
			echo "  ✓ $$block"; \
		else \
			echo "  ○ $$block (not started)"; \
		fi; \
	done
	@echo ""
	@echo "Level 3 Subsystems:"
	@for block in $(LEVEL3_BLOCKS); do \
		if [ -d "$$block/runs/run1" ]; then \
			echo "  ✓ $$block"; \
		else \
			echo "  ○ $$block (not started)"; \
		fi; \
	done
	@echo ""
	@echo "Level 4 Top Integration:"
	@if [ -d "level4/silens_soc/runs/run1" ]; then \
		echo "  ✓ level4/silens_soc"; \
	else \
		echo "  ○ level4/silens_soc (not started)"; \
	fi

.PHONY: verify-rtl
verify-rtl:
	@echo "=== Verifying RTL Compilation ==="
	@echo ""
	@echo "Level 1 Blocks:"
	@for block in $(LEVEL1_BLOCKS); do \
		src=$$(ls $$block/src/*.v 2>/dev/null | head -1); \
		if [ -n "$$src" ]; then \
			if iverilog -o /dev/null -Wall "$$src" 2>/dev/null; then \
				echo "  ✓ $$block"; \
			else \
				echo "  ✗ $$block (compile errors)"; \
			fi; \
		else \
			echo "  ? $$block (no source found)"; \
		fi; \
	done
	@echo ""
	@echo "Level 2 Blocks:"
	@for block in $(LEVEL2_BLOCKS); do \
		src=$$(ls $$block/src/*.v 2>/dev/null | head -1); \
		if [ -n "$$src" ]; then \
			echo "  ○ $$block (requires Level 1 deps)"; \
		else \
			echo "  ? $$block (no source found)"; \
		fi; \
	done

.PHONY: help
help:
	@echo "SiLens Hierarchical Synthesis"
	@echo ""
	@echo "Targets:"
	@echo "  level1    - Synthesize Level 1 primitives (MAC, norms)"
	@echo "  level2    - Synthesize Level 2 blocks (transformer blocks)"
	@echo "  level3    - Synthesize Level 3 subsystems (vision, LLM)"
	@echo "  level4    - Top-level integration (800mm² SoC)"
	@echo "  all       - Full hierarchical flow"
	@echo "  status    - Show synthesis status"
	@echo "  clean     - Remove all run directories"
	@echo ""
	@echo "Environment variables:"
	@echo "  OPENLANE_ROOT - OpenLane installation (default: ~/OpenLane)"
	@echo "  PDK_ROOT      - PDK installation (default: ~/pdk)"
