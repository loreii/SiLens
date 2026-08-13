# =============================================================================
# SiLens Intel SDC Constraints - Cyclone 10 GX
# =============================================================================
# Target: Intel Cyclone 10 GX Development Kit
#
# License: Apache 2.0
# =============================================================================

# =============================================================================
# Clock Constraints
# =============================================================================

# 50 MHz oscillator input
create_clock -name clk_osc -period 20.000 [get_ports clk_osc]

# 100 MHz PCIe reference clock
create_clock -name pcie_refclk -period 10.000 [get_ports pcie_refclk]

# =============================================================================
# PLL Generated Clocks
# =============================================================================

# Core clock (100 MHz) from PLL
create_generated_clock -name core_clk \
    -source [get_pins {pll_inst|outclk_0}] \
    -divide_by 1 \
    [get_pins {pll_inst|outclk_0}]

# Fast clock (200 MHz) from PLL
create_generated_clock -name fast_clk \
    -source [get_pins {pll_inst|outclk_1}] \
    -divide_by 1 \
    [get_pins {pll_inst|outclk_1}]

# =============================================================================
# Clock Domain Crossing
# =============================================================================

set_clock_groups -asynchronous \
    -group [get_clocks clk_osc] \
    -group [get_clocks pcie_refclk]

set_clock_groups -asynchronous \
    -group [get_clocks core_clk] \
    -group [get_clocks fast_clk]


# =============================================================================
# Reset Constraints
# =============================================================================

# Asynchronous reset - treat as false path
set_false_path -from [get_ports sys_rst_n]
set_false_path -from [get_ports pcie_rst_n]

# =============================================================================
# Input Constraints
# =============================================================================

# Button inputs (asynchronous, debounced externally)
set_false_path -from [get_ports {btn[*]}]

# Switch inputs (asynchronous)
set_false_path -from [get_ports {sw[*]}]

# =============================================================================
# Output Constraints
# =============================================================================

# LED outputs (slow, relaxed timing)
set_max_delay -to [get_ports {led[*]}] 20.000

# =============================================================================
# PCIe Constraints
# =============================================================================

# PCIe TX outputs
set_output_delay -clock pcie_refclk -max 3.000 [get_ports {pcie_tx_p[*]}]
set_output_delay -clock pcie_refclk -min -1.000 [get_ports {pcie_tx_p[*]}]

# PCIe RX inputs
set_input_delay -clock pcie_refclk -max 3.000 [get_ports {pcie_rx_p[*]}]
set_input_delay -clock pcie_refclk -min 0.000 [get_ports {pcie_rx_p[*]}]

# =============================================================================
# Multicycle Paths
# =============================================================================

# Weight ROM reads are static - allow multiple cycles
set_multicycle_path -setup 2 -from [get_registers {*weight_rom*}]
set_multicycle_path -hold 1 -from [get_registers {*weight_rom*}]

# =============================================================================
# False Paths
# =============================================================================

# Heartbeat counter does not need tight timing
set_false_path -to [get_registers {*heartbeat*}]

# Status LEDs
set_false_path -to [get_registers {*status_leds*}]
