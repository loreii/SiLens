#!/bin/bash
# =============================================================================
# SiLens Level 1 Synthesis Script
# =============================================================================
# Synthesizes all Level 1 compute primitives using OpenLane
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SILENS_ROOT="$(dirname "$SCRIPT_DIR")"
OPENLANE_ROOT="${OPENLANE_ROOT:-$HOME/OpenLane}"
PDK_ROOT="${PDK_ROOT:-$HOME/pdk}"

LEVEL1_BLOCKS=(
    "ternary_mac_array_64"
    "rms_norm_block"
    "layer_norm_block"
    "softmax_unit"
    "silu_unit"
    "attention_head"
    "mlp_block"
)

echo "=============================================="
echo "SiLens Level 1 Synthesis"
echo "=============================================="
echo "OpenLane: $OPENLANE_ROOT"
echo "PDK:      $PDK_ROOT"
echo "Blocks:   ${#LEVEL1_BLOCKS[@]}"
echo ""

# Track results
declare -A RESULTS
PASS=0
FAIL=0

for block in "${LEVEL1_BLOCKS[@]}"; do
    echo "=============================================="
    echo "[$((PASS + FAIL + 1))/${#LEVEL1_BLOCKS[@]}] Synthesizing: $block"
    echo "=============================================="
    
    SRC_DIR="$SILENS_ROOT/openlane/level1/$block"
    DEST_DIR="$OPENLANE_ROOT/designs/$block"
    
    # Check if source exists
    if [ ! -d "$SRC_DIR" ]; then
        echo "  ⚠ Source not found: $SRC_DIR"
        RESULTS[$block]="SKIP (no source)"
        continue
    fi
    
    # Check if config exists
    if [ ! -f "$SRC_DIR/config.json" ]; then
        echo "  ⚠ No config.json found"
        RESULTS[$block]="SKIP (no config)"
        continue
    fi
    
    # Copy design to OpenLane
    echo "  Copying design to OpenLane..."
    rm -rf "$DEST_DIR"
    mkdir -p "$DEST_DIR"
    cp -r "$SRC_DIR"/* "$DEST_DIR"/
    
    # Run synthesis
    echo "  Running OpenLane flow..."
    START_TIME=$(date +%s)
    
    if docker run --rm \
        -v "$OPENLANE_ROOT":/openlane \
        -v "$PDK_ROOT":/home/pdk \
        -e PDK_ROOT=/home/pdk \
        -e PDK=sky130A \
        ghcr.io/the-openroad-project/openlane:latest \
        ./flow.tcl -design "$block" -tag run1 2>&1 | tee "/tmp/synth_${block}.log" | grep -E "^\[INFO\]|\[WARNING\]|\[ERROR\]|\[SUCCESS\]|STEP"; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        # Check for success
        if grep -q "\[SUCCESS\]: Flow complete" "/tmp/synth_${block}.log"; then
            # Check for DRC violations
            DRC=$(grep -o "No DRC violations" "/tmp/synth_${block}.log" | head -1)
            if [ -n "$DRC" ]; then
                echo "  ✅ SUCCESS (${DURATION}s) - DRC clean"
                RESULTS[$block]="PASS (${DURATION}s)"
                PASS=$((PASS + 1))
            else
                echo "  ⚠ SUCCESS with DRC issues (${DURATION}s)"
                RESULTS[$block]="PASS with warnings (${DURATION}s)"
                PASS=$((PASS + 1))
            fi
        else
            echo "  ❌ FAILED"
            RESULTS[$block]="FAIL"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  ❌ FAILED (docker error)"
        RESULTS[$block]="FAIL (docker)"
        FAIL=$((FAIL + 1))
    fi
    
    echo ""
done

# Summary
echo "=============================================="
echo "Synthesis Summary"
echo "=============================================="
echo ""
printf "%-25s %s\n" "Block" "Result"
printf "%-25s %s\n" "-----" "------"
for block in "${LEVEL1_BLOCKS[@]}"; do
    printf "%-25s %s\n" "$block" "${RESULTS[$block]:-NOT RUN}"
done
echo ""
echo "Passed: $PASS / ${#LEVEL1_BLOCKS[@]}"
echo "Failed: $FAIL / ${#LEVEL1_BLOCKS[@]}"
echo ""

# Copy results back to SiLens
echo "Copying results to SiLens repo..."
for block in "${LEVEL1_BLOCKS[@]}"; do
    if [ -d "$OPENLANE_ROOT/designs/$block/runs/run1" ]; then
        mkdir -p "$SILENS_ROOT/openlane/level1/$block/runs"
        cp -r "$OPENLANE_ROOT/designs/$block/runs/run1" "$SILENS_ROOT/openlane/level1/$block/runs/" 2>/dev/null || true
    fi
done

echo "Done!"
