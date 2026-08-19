#!/bin/bash
# =============================================================================
# SiLens OpenLane Setup Script
# =============================================================================
# This script sets up OpenLane and the SkyWater SKY130 PDK for synthesis.
#
# Requirements:
#   - Docker Desktop installed and running
#   - ~20GB free disk space
#   - Internet connection
#
# Usage:
#   ./scripts/setup_openlane.sh
#
# =============================================================================

set -e

OPENLANE_ROOT="${OPENLANE_ROOT:-$HOME/OpenLane}"
PDK_ROOT="${PDK_ROOT:-$HOME/pdk}"
PDK="sky130A"

echo "=============================================="
echo "SiLens OpenLane Setup"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  OPENLANE_ROOT: $OPENLANE_ROOT"
echo "  PDK_ROOT:      $PDK_ROOT"
echo "  PDK:           $PDK"
echo ""

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------

echo "[1/5] Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker Desktop first."
    echo "  macOS: https://docs.docker.com/desktop/install/mac-install/"
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running."
    echo "  Please start Docker Desktop and try again."
    exit 1
fi

echo "  ✓ Docker is running"

# Check disk space (need ~20GB)
FREE_SPACE=$(df -g ~ | tail -1 | awk '{print $4}')
if [ "$FREE_SPACE" -lt 20 ]; then
    echo "WARNING: Less than 20GB free disk space ($FREE_SPACE GB available)"
    echo "  OpenLane + PDK requires ~15-20GB. Proceed with caution."
fi
echo "  ✓ Disk space: ${FREE_SPACE}GB available"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found."
    exit 1
fi
echo "  ✓ Python 3 found"

# -----------------------------------------------------------------------------
# Clone or update OpenLane
# -----------------------------------------------------------------------------

echo ""
echo "[2/5] Setting up OpenLane..."

if [ -d "$OPENLANE_ROOT" ]; then
    echo "  OpenLane directory exists. Updating..."
    cd "$OPENLANE_ROOT"
    git fetch origin
    git checkout master
    git pull origin master
else
    echo "  Cloning OpenLane..."
    git clone https://github.com/The-OpenROAD-Project/OpenLane.git "$OPENLANE_ROOT"
    cd "$OPENLANE_ROOT"
fi

echo "  ✓ OpenLane ready at $OPENLANE_ROOT"

# -----------------------------------------------------------------------------
# Pull OpenLane Docker image
# -----------------------------------------------------------------------------

echo ""
echo "[3/5] Pulling OpenLane Docker image (this may take 5-10 minutes)..."

cd "$OPENLANE_ROOT"
make pull-openlane

echo "  ✓ OpenLane Docker image pulled"

# -----------------------------------------------------------------------------
# Setup PDK
# -----------------------------------------------------------------------------

echo ""
echo "[4/5] Setting up SkyWater SKY130 PDK (this may take 10-20 minutes)..."

# Create PDK directory
mkdir -p "$PDK_ROOT"

# Use OpenLane's PDK setup
cd "$OPENLANE_ROOT"
make pdk

echo "  ✓ PDK installed at $PDK_ROOT"

# -----------------------------------------------------------------------------
# Verify installation
# -----------------------------------------------------------------------------

echo ""
echo "[5/5] Verifying installation..."

cd "$OPENLANE_ROOT"

# Run a quick test
echo "  Running OpenLane smoke test..."
if make test 2>&1 | tail -20 | grep -q "success\|PASSED"; then
    echo "  ✓ OpenLane smoke test passed"
else
    echo "  ⚠ Smoke test may have issues - check output above"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "OpenLane Setup Complete!"
echo "=============================================="
echo ""
echo "Environment variables to add to your shell profile:"
echo ""
echo "  export OPENLANE_ROOT=\"$OPENLANE_ROOT\""
echo "  export PDK_ROOT=\"$PDK_ROOT\""
echo "  export PDK=\"$PDK\""
echo ""
echo "To run synthesis on SiLens Edge:"
echo ""
echo "  cd $(dirname "$0")/../openlane"
echo "  make VARIANT=silens-edge level1  # Build Level 1 primitives"
echo "  make VARIANT=silens-edge level3  # Build Edge-specific blocks"
echo "  make VARIANT=silens-edge level4  # Build top integration"
echo ""
echo "Or for the full VLM variant:"
echo ""
echo "  make VARIANT=silens-vlm all"
echo ""
