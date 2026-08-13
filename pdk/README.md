# SkyWater PDK Setup

## Overview

SiLens uses the **SkyWater SKY130** open-source PDK (Process Design Kit) for ASIC implementation.

## PDK Information

| Property | Value |
|----------|-------|
| Process | 130nm CMOS |
| Metal layers | 5 |
| Core voltage | 1.8V |
| I/O voltage | 3.3V |
| Transistor types | NMOS, PMOS |
| License | Apache 2.0 |

## Installation

### Option 1: Git Submodule (Recommended)

```bash
# Initialize submodule
git submodule update --init pdk/skywater-pdk

# Fetch all libraries (large download)
cd pdk/skywater-pdk
git submodule update --init --recursive
```

### Option 2: Manual Clone

```bash
# Clone PDK
git clone https://github.com/google/skywater-pdk.git pdk/skywater-pdk

# Fetch standard cell library
cd pdk/skywater-pdk
git submodule update --init libraries/sky130_fd_sc_hd/latest
```

### Option 3: Setup Script

```bash
./tools/setup_pdk.sh --minimal  # Essential libraries only (~2GB)
./tools/setup_pdk.sh --full     # Complete PDK (~7GB)
```

## Directory Structure

After setup:

```
pdk/
├── skywater-pdk/           # Main PDK repository
│   ├── libraries/
│   │   ├── sky130_fd_sc_hd/    # High-density standard cells
│   │   ├── sky130_fd_sc_hs/    # High-speed standard cells
│   │   ├── sky130_fd_io/       # I/O cells
│   │   └── ...
│   ├── docs/
│   └── ...
└── open_pdks/              # OpenPDKs wrapper (optional)
```

## Standard Cell Libraries

| Library | Description | Use Case |
|---------|-------------|----------|
| sky130_fd_sc_hd | High density | Area-optimized logic |
| sky130_fd_sc_hs | High speed | Timing-critical paths |
| sky130_fd_sc_hdll | High density, low leakage | Power optimization |
| sky130_fd_io | I/O cells | Pad ring |

For SiLens, we primarily use `sky130_fd_sc_hd`.

## Environment Setup

```bash
# Source the environment file
source .env.pdk

# Or set manually
export PDK_ROOT=/path/to/pdk/skywater-pdk
export PDK=sky130A
export STD_CELL_LIBRARY=sky130_fd_sc_hd
```

## OpenLane Integration

OpenLane uses the PDK for synthesis, place-and-route, and signoff.

```bash
# Run synthesis with Docker
docker run --rm \
    -v $PDK_ROOT:/pdk \
    -v $(pwd):/work \
    -e PDK_ROOT=/pdk \
    efabless/openlane:latest \
    bash -c "cd /work && flow.tcl -design silens"
```

## Resources

- **PDK Documentation:** https://skywater-pdk.readthedocs.io/
- **GitHub:** https://github.com/google/skywater-pdk
- **OpenLane:** https://github.com/The-OpenROAD-Project/OpenLane
- **Efabless:** https://efabless.com/ (shuttle runs)

## Fabrication

To manufacture chips:

1. **MPW (Multi-Project Wafer):** Share wafer with other designs
   - Efabless chipIgnite: ~$10K for a slot
   - Google/Efabless free shuttles (periodic)

2. **Dedicated run:** Full wafer for your design
   - Contact SkyWater Technology directly
   - Higher cost, higher volume

## License

SkyWater SKY130 PDK is licensed under Apache 2.0.
