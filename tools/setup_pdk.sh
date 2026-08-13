#!/bin/bash
# =============================================================================
# SiLens PDK Setup Script
# =============================================================================
#
# This script sets up the SkyWater SKY130 PDK for the SiLens project.
#
# The PDK is large (~7GB), so we provide options for:
# 1. Minimal install (just what's needed for synthesis)
# 2. Full install (complete PDK with all libraries)
#
# Usage:
#   ./tools/setup_pdk.sh           # Minimal install
#   ./tools/setup_pdk.sh --full    # Full install
#   ./tools/setup_pdk.sh --help    # Show help
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PDK_DIR="$PROJECT_ROOT/pdk"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

show_help() {
    cat << EOF
SiLens PDK Setup Script

Usage: $0 [OPTIONS]

Options:
    --minimal       Minimal install (default) - ~2GB
    --full          Full PDK install - ~7GB
    --submodule     Add as git submodule instead of clone
    --help          Show this help message

The SkyWater SKY130 PDK is required for:
- RTL synthesis
- Physical design
- Timing analysis
- DRC/LVS verification

More info: https://skywater-pdk.readthedocs.io/

EOF
}

check_dependencies() {
    print_header "Checking dependencies"
    
    local missing=0
    
    # Check git
    if command -v git &> /dev/null; then
        print_success "git found: $(git --version)"
    else
        print_error "git not found"
        missing=1
    fi
    
    # Check python
    if command -v python3 &> /dev/null; then
        print_success "python3 found: $(python3 --version)"
    else
        print_warning "python3 not found (optional for PDK scripts)"
    fi
    
    # Check disk space
    local available=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$available" -gt 10 ]; then
        print_success "Disk space: ${available}GB available"
    else
        print_warning "Low disk space: ${available}GB (recommend >10GB)"
    fi
    
    if [ $missing -eq 1 ]; then
        print_error "Missing required dependencies"
        exit 1
    fi
}

setup_minimal() {
    print_header "Setting up minimal PDK (synthesis only)"
    
    mkdir -p "$PDK_DIR"
    cd "$PDK_DIR"
    
    # Clone with depth=1 for faster download
    if [ -d "skywater-pdk" ]; then
        print_warning "skywater-pdk directory already exists"
        read -p "Remove and re-clone? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf skywater-pdk
        else
            print_success "Using existing PDK"
            return
        fi
    fi
    
    echo "Cloning SkyWater PDK (minimal)..."
    git clone --depth=1 https://github.com/google/skywater-pdk.git
    
    cd skywater-pdk
    
    # Only fetch essential libraries
    echo "Fetching essential libraries..."
    git submodule update --init libraries/sky130_fd_sc_hd/latest
    git submodule update --init libraries/sky130_fd_io/latest
    
    print_success "Minimal PDK setup complete"
}

setup_full() {
    print_header "Setting up full PDK"
    
    mkdir -p "$PDK_DIR"
    cd "$PDK_DIR"
    
    if [ -d "skywater-pdk" ]; then
        print_warning "skywater-pdk directory already exists"
        read -p "Remove and re-clone? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf skywater-pdk
        else
            print_success "Using existing PDK"
            return
        fi
    fi
    
    echo "Cloning SkyWater PDK (full - this will take a while)..."
    git clone https://github.com/google/skywater-pdk.git
    
    cd skywater-pdk
    
    echo "Fetching all submodules (this may take 30+ minutes)..."
    git submodule update --init --recursive
    
    print_success "Full PDK setup complete"
}

setup_submodule() {
    print_header "Adding PDK as git submodule"
    
    cd "$PROJECT_ROOT"
    
    if [ -f ".gitmodules" ] && grep -q "skywater-pdk" .gitmodules; then
        print_warning "Submodule already configured in .gitmodules"
        echo "Running: git submodule update --init pdk/skywater-pdk"
        git submodule update --init pdk/skywater-pdk
    else
        echo "Adding submodule..."
        git submodule add https://github.com/google/skywater-pdk.git pdk/skywater-pdk
    fi
    
    print_success "Submodule setup complete"
    echo ""
    echo "To fetch PDK contents, run:"
    echo "  git submodule update --init --recursive pdk/skywater-pdk"
}

setup_openlane() {
    print_header "Setting up OpenLane (optional)"
    
    echo "OpenLane is the RTL-to-GDSII flow. Install options:"
    echo ""
    echo "1. Docker (recommended):"
    echo "   docker pull efabless/openlane:latest"
    echo ""
    echo "2. Native installation:"
    echo "   See: https://openlane.readthedocs.io/en/latest/getting_started/installation.html"
    echo ""
    
    read -p "Install OpenLane via Docker now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v docker &> /dev/null; then
            echo "Pulling OpenLane Docker image..."
            docker pull efabless/openlane:latest
            print_success "OpenLane Docker image ready"
        else
            print_error "Docker not found. Please install Docker first."
        fi
    fi
}

create_env_file() {
    print_header "Creating environment file"
    
    local env_file="$PROJECT_ROOT/.env.pdk"
    
    cat > "$env_file" << EOF
# SiLens PDK Environment Variables
# Source this file before running synthesis:
#   source .env.pdk

export PDK_ROOT="$PDK_DIR/skywater-pdk"
export PDK="sky130A"
export STD_CELL_LIBRARY="sky130_fd_sc_hd"

# OpenLane paths (if using native install)
# export OPENLANE_ROOT="/path/to/openlane"

echo "PDK environment loaded:"
echo "  PDK_ROOT=\$PDK_ROOT"
echo "  PDK=\$PDK"
EOF

    print_success "Environment file created: $env_file"
    echo ""
    echo "To use, run: source .env.pdk"
}

verify_setup() {
    print_header "Verifying PDK setup"
    
    local pdk_path="$PDK_DIR/skywater-pdk"
    
    if [ ! -d "$pdk_path" ]; then
        print_error "PDK not found at $pdk_path"
        return 1
    fi
    
    # Check for essential files
    local checks=(
        "libraries/sky130_fd_sc_hd"
        "docs"
    )
    
    for check in "${checks[@]}"; do
        if [ -e "$pdk_path/$check" ]; then
            print_success "Found: $check"
        else
            print_warning "Missing: $check"
        fi
    done
    
    echo ""
    print_success "PDK setup verification complete"
}

# =============================================================================
# Main
# =============================================================================

INSTALL_TYPE="minimal"

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            INSTALL_TYPE="minimal"
            shift
            ;;
        --full)
            INSTALL_TYPE="full"
            shift
            ;;
        --submodule)
            INSTALL_TYPE="submodule"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

print_header "SiLens PDK Setup"

echo "Project root: $PROJECT_ROOT"
echo "PDK directory: $PDK_DIR"
echo "Install type: $INSTALL_TYPE"

check_dependencies

case $INSTALL_TYPE in
    minimal)
        setup_minimal
        ;;
    full)
        setup_full
        ;;
    submodule)
        setup_submodule
        ;;
esac

create_env_file
verify_setup

print_header "Setup Complete"

echo "Next steps:"
echo "  1. Source the environment: source .env.pdk"
echo "  2. Download the model: python tools/download_model.py"
echo "  3. Start development!"
echo ""
echo "Documentation:"
echo "  - SkyWater PDK: https://skywater-pdk.readthedocs.io/"
echo "  - OpenLane: https://openlane.readthedocs.io/"
echo "  - SiLens: ./docs/"
