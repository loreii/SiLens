# =============================================================================
# SiLens Hierarchical Synthesis Makefile - Multi-Variant Support
# =============================================================================
# Bottom-up synthesis flow: Level 1 → Level 2 → Level 3 → Level 4
#
# Usage:
#   make level1                        # Synthesize shared Level 1 primitives
#   make level2                        # Synthesize shared Level 2 blocks
#   make VARIANT=silens-vlm level3     # Variant-specific Level 3
#   make VARIANT=silens-vlm level4     # Variant-specific Level 4
#   make VARIANT=silens-vlm all        # Full flow for variant
#   make clean                         # Clean all runs
#
# Variants:
#   silens-vlm   - Full 800mm² Vision-Language Model (default)
#   silens-edge  - Compact 50mm² Edge Classifier
#
# Requirements:
#   - OpenLane installed and in PATH
#   - SKY130 PDK installed
# =============================================================================

OPENLANE_ROOT ?= $(HOME)/OpenLane
PDK_ROOT ?= $(HOME)/pdk
PDK ?= sky130A

# Default variant
VARIANT ?= silens-vlm

# Paths
SHARED_DIR := $(CURDIR)
VARIANT_DIR := $(CURDIR)/../variants/$(VARIANT)/openlane

# Flow command
FLOW = cd $(OPENLANE_ROOT) && ./flow.tcl -design

# =============================================================================
# Level 1: Compute Primitives (SHARED across all variants)
# =============================================================================

LEVEL1_BLOCKS = \
	level1/ternary_mac_array_64 \
	level1/rms_norm_block \
	level1/layer_norm_block \
	level1/softmax_unit \
	level1/silu_unit \
	level1/attention_head \
	level1/mlp_block

.PHONY: level1
level1: $(LEVEL1_BLOCKS)
	@echo "=== Level 1 (shared primitives) complete ==="

level1/ternary_mac_array_64: level1/ternary_mac_array_64/config.json
	@echo "=== Synthesizing ternary_mac_array_64 ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level1/rms_norm_block: level1/rms_norm_block/config.json
	@echo "=== Synthesizing rms_norm_block ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level1/layer_norm_block: level1/layer_norm_block/config.json
	@echo "=== Synthesizing layer_norm_block ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level1/softmax_unit: level1/softmax_unit/config.json
	@echo "=== Synthesizing softmax_unit ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level1/silu_unit: level1/silu_unit/config.json
	@echo "=== Synthesizing silu_unit ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level1/attention_head: level1/attention_head/config.json
	@echo "=== Synthesizing attention_head ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level1/mlp_block: level1/mlp_block/config.json
	@echo "=== Synthesizing mlp_block ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

# =============================================================================
# Level 2: Functional Blocks (SHARED across all variants)
# =============================================================================

LEVEL2_BLOCKS = \
	level2/transformer_block_llm \
	level2/transformer_block_vision \
	level2/projector_block \
	level2/embedding_block

.PHONY: level2
level2: level1 $(LEVEL2_BLOCKS)
	@echo "=== Level 2 (shared blocks) complete ==="

level2/transformer_block_llm: level2/transformer_block_llm/config.json level1
	@echo "=== Synthesizing transformer_block_llm ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level2/transformer_block_vision: level2/transformer_block_vision/config.json level1
	@echo "=== Synthesizing transformer_block_vision ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level2/projector_block: level2/projector_block/config.json
	@echo "=== Synthesizing projector_block ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

level2/embedding_block: level2/embedding_block/config.json
	@echo "=== Synthesizing embedding_block ==="
	$(FLOW) $(SHARED_DIR)/$@ -tag run1

# =============================================================================
# Level 3: Subsystems (VARIANT-SPECIFIC)
# =============================================================================

.PHONY: level3
level3: level2
	@echo "=== Synthesizing Level 3 for variant: $(VARIANT) ==="
	@if [ ! -d "$(VARIANT_DIR)/level3" ]; then \
		echo "ERROR: Level 3 not found for variant $(VARIANT)"; \
		echo "Expected: $(VARIANT_DIR)/level3"; \
		exit 1; \
	fi
	@for block in $$(ls -d $(VARIANT_DIR)/level3/*/); do \
		echo "=== Synthesizing $$(basename $$block) ==="; \
		$(FLOW) $$block -tag run1; \
	done
	@echo "=== Level 3 ($(VARIANT)) complete ==="

# =============================================================================
# Level 4: Top Integration (VARIANT-SPECIFIC)
# =============================================================================

.PHONY: level4
level4: level3
	@echo "=== Synthesizing Level 4 for variant: $(VARIANT) ==="
	@if [ ! -d "$(VARIANT_DIR)/level4" ]; then \
		echo "ERROR: Level 4 not found for variant $(VARIANT)"; \
		echo "Expected: $(VARIANT_DIR)/level4"; \
		exit 1; \
	fi
	@for block in $$(ls -d $(VARIANT_DIR)/level4/*/); do \
		echo "=== Synthesizing $$(basename $$block) ==="; \
		$(FLOW) $$block -tag run1; \
	done
	@echo "=== Level 4 ($(VARIANT)) complete ==="

# =============================================================================
# Full flow
# =============================================================================

.PHONY: all
all: level1 level2 level3 level4
	@echo "=== Full hierarchical synthesis complete for $(VARIANT) ==="

# =============================================================================
# Utilities
# =============================================================================

.PHONY: clean
clean:
	rm -rf level1/*/runs
	rm -rf level2/*/runs
	rm -rf $(CURDIR)/../variants/*/openlane/level3/*/runs
	rm -rf $(CURDIR)/../variants/*/openlane/level4/*/runs
	@echo "Cleaned all synthesis runs"

.PHONY: clean-variant
clean-variant:
	rm -rf $(VARIANT_DIR)/level3/*/runs
	rm -rf $(VARIANT_DIR)/level4/*/runs
	@echo "Cleaned synthesis runs for variant: $(VARIANT)"

.PHONY: status
status:
	@echo "=== Synthesis Status ==="
	@echo "Current variant: $(VARIANT)"
	@echo ""
	@echo "Level 1 Primitives (shared):"
	@for block in $(LEVEL1_BLOCKS); do \
		if [ -d "$$block/runs/run1" ]; then \
			echo "  ✓ $$block"; \
		else \
			echo "  ○ $$block"; \
		fi; \
	done
	@echo ""
	@echo "Level 2 Blocks (shared):"
	@for block in $(LEVEL2_BLOCKS); do \
		if [ -d "$$block/runs/run1" ]; then \
			echo "  ✓ $$block"; \
		else \
			echo "  ○ $$block"; \
		fi; \
	done
	@echo ""
	@echo "Level 3 Subsystems ($(VARIANT)):"
	@if [ -d "$(VARIANT_DIR)/level3" ]; then \
		for block in $$(ls -d $(VARIANT_DIR)/level3/*/ 2>/dev/null); do \
			name=$$(basename $$block); \
			if [ -d "$$block/runs/run1" ]; then \
				echo "  ✓ $$name"; \
			else \
				echo "  ○ $$name"; \
			fi; \
		done; \
	else \
		echo "  (not configured for this variant)"; \
	fi
	@echo ""
	@echo "Level 4 Top ($(VARIANT)):"
	@if [ -d "$(VARIANT_DIR)/level4" ]; then \
		for block in $$(ls -d $(VARIANT_DIR)/level4/*/ 2>/dev/null); do \
			name=$$(basename $$block); \
			if [ -d "$$block/runs/run1" ]; then \
				echo "  ✓ $$name"; \
			else \
				echo "  ○ $$name"; \
			fi; \
		done; \
	else \
		echo "  (not configured for this variant)"; \
	fi

.PHONY: list-variants
list-variants:
	@echo "Available variants:"
	@for v in $$(ls -d $(CURDIR)/../variants/*/); do \
		name=$$(basename $$v); \
		if [ -f "$$v/config.json" ]; then \
			desc=$$(grep '"variant_description"' $$v/config.json | cut -d'"' -f4); \
			echo "  $$name - $$desc"; \
		else \
			echo "  $$name"; \
		fi; \
	done

.PHONY: verify-rtl
verify-rtl:
	@echo "=== Verifying RTL Compilation ==="
	@echo ""
	@echo "Level 1 Blocks (shared):"
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

.PHONY: help
help:
	@echo "SiLens Hierarchical Synthesis - Multi-Variant Build System"
	@echo ""
	@echo "Variants:"
	@echo "  silens-vlm   - Full 800mm² Vision-Language Model (default)"
	@echo "  silens-edge  - Compact 50mm² Edge Classifier"
	@echo ""
	@echo "Targets:"
	@echo "  level1         - Synthesize shared Level 1 primitives"
	@echo "  level2         - Synthesize shared Level 2 blocks"
	@echo "  level3         - Synthesize variant-specific Level 3 subsystems"
	@echo "  level4         - Synthesize variant-specific Level 4 top"
	@echo "  all            - Full hierarchical flow for current variant"
	@echo "  status         - Show synthesis status"
	@echo "  list-variants  - List available variants"
	@echo "  clean          - Clean all runs (all variants)"
	@echo "  clean-variant  - Clean runs for current variant only"
	@echo ""
	@echo "Usage examples:"
	@echo "  make level1 level2              # Build shared primitives"
	@echo "  make VARIANT=silens-vlm all     # Build full 800mm² VLM"
	@echo "  make VARIANT=silens-edge all    # Build compact Edge classifier"
	@echo "  make VARIANT=silens-edge status # Check Edge build status"
	@echo ""
	@echo "Environment variables:"
	@echo "  VARIANT       - Target variant (default: silens-vlm)"
	@echo "  OPENLANE_ROOT - OpenLane installation (default: ~/OpenLane)"
	@echo "  PDK_ROOT      - PDK installation (default: ~/pdk)"
