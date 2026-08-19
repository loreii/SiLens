#!/bin/bash
#
# check_rtl.sh - RTL compilation check for SiLens Edge variant
#
# This script verifies that all Edge variant Verilog files compile
# cleanly with Icarus Verilog. Run from the SiLens root directory.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================================="
echo "SiLens Edge RTL Compilation Check"
echo "=================================================="
echo ""

cd "$PROJECT_ROOT"

# Check if iverilog is available
if ! command -v iverilog &> /dev/null; then
    echo -e "${RED}ERROR: iverilog not found. Please install Icarus Verilog.${NC}"
    exit 1
fi

echo "Using iverilog version: $(iverilog -V 2>&1 | head -1)"
echo ""

ERRORS=0

# Test 1: Compile Edge SoC top-level with all dependencies
echo -e "${YELLOW}[1/2] Compiling Edge SoC top-level...${NC}"
if iverilog -g2012 -Wall \
    -I variants/silens-edge/openlane/level3/vision_nano/src \
    -I variants/silens-edge/openlane/level3/classifier_head/src \
    -I variants/silens-edge/openlane/level3/io_edge/src \
    -I variants/silens-edge/openlane/level4/silens_edge_soc/src \
    -o /dev/null \
    variants/silens-edge/openlane/level4/silens_edge_soc/src/silens_edge_soc.v \
    variants/silens-edge/openlane/level3/vision_nano/src/vision_nano.v \
    variants/silens-edge/openlane/level3/classifier_head/src/classifier_head.v \
    variants/silens-edge/openlane/level3/io_edge/src/io_edge.v 2>&1; then
    echo -e "${GREEN}  ✓ Edge SoC compilation passed${NC}"
else
    echo -e "${RED}  ✗ Edge SoC compilation FAILED${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Test 2: Compile FPGA wrapper with all dependencies
echo -e "${YELLOW}[2/2] Compiling Edge FPGA wrapper...${NC}"
if iverilog -g2012 -Wall -DSIMULATION \
    -I variants/silens-edge/openlane/level4/silens_edge_soc/src \
    -I variants/silens-edge/openlane/level3/vision_nano/src \
    -I variants/silens-edge/openlane/level3/classifier_head/src \
    -I variants/silens-edge/openlane/level3/io_edge/src \
    -o /dev/null \
    fpga/edge/silens_edge_fpga_wrapper.v \
    variants/silens-edge/openlane/level4/silens_edge_soc/src/silens_edge_soc.v \
    variants/silens-edge/openlane/level3/vision_nano/src/vision_nano.v \
    variants/silens-edge/openlane/level3/classifier_head/src/classifier_head.v \
    variants/silens-edge/openlane/level3/io_edge/src/io_edge.v 2>&1; then
    echo -e "${GREEN}  ✓ FPGA wrapper compilation passed${NC}"
else
    echo -e "${RED}  ✗ FPGA wrapper compilation FAILED${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Summary
echo "=================================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All RTL compilation checks PASSED${NC}"
    exit 0
else
    echo -e "${RED}$ERRORS compilation check(s) FAILED${NC}"
    exit 1
fi
