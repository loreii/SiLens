# =============================================================================
# SiLens Intel SDC Constraints - Arria 10 GX
# =============================================================================
# Target: Intel Arria 10 GX Development Kit
#
# License: Apache 2.0
# =============================================================================

# =============================================================================
# Clock Constraints
# =============================================================================

# 100 MHz oscillator input (differential)
create_clock -name clk_osc -period 10.000 [get_ports clk_osc_p]

# 100 MHz PCIe reference clock
create_clock -name pcie_refclk -period 10.000 [get_ports pcie_refclk_p]

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

# Memory controller clock (300 MHz)
create_generated_clock -name mem_clk \
    -source [get_pins {pll_inst|outclk_2}] \
    -divide_by 1 \
    [get_pins {pll_inst|outclk_2}]

# =============================================================================
# Clock Domain Crossing
# =============================================================================

set_clock_groups -asynchronous \
    -group [get_clocks clk_osc] \
    -group [get_clocks pcie_refclk]

set_clock_groups -asynchronous \
    -group [get_clocks core_clk] \
    -group [get_clocks fast_clk] \
    -group [get_clocks mem_clk]

# =============================================================================
# Reset Constraints
# =============================================================================

set_false_path -from [get_ports sys_rst_n]
set_false_path -from [get_ports pcie_rst_n]

# =============================================================================
# I/O Constraints
# =============================================================================

set_false_path -from [get_ports {btn[*]}]
set_false_path -from [get_ports {sw[*]}]
set_max_delay -to [get_ports {led[*]}] 20.000


# =============================================================================
# PCIe Hard IP Constraints
# =============================================================================

# PCIe Gen3 x4 timing
set_output_delay -clock pcie_refclk -max 2.500 [get_ports {pcie_tx_p[*]}]
set_output_delay -clock pcie_refclk -min -0.500 [get_ports {pcie_tx_p[*]}]

set_input_delay -clock pcie_refclk -max 2.500 [get_ports {pcie_rx_p[*]}]
set_input_delay -clock pcie_refclk -min 0.000 [get_ports {pcie_rx_p[*]}]

# =============================================================================
# DDR4 Memory Interface Constraints
# =============================================================================

# Memory interface has its own clock domain (handled by EMIF IP)
# set_false_path between core domain and memory domain handled by EMIF

# =============================================================================
# Multicycle Paths
# =============================================================================

# Weight ROM reads
set_multicycle_path -setup 2 -from [get_registers {*weight_rom*}]
set_multicycle_path -hold 1 -from [get_registers {*weight_rom*}]

# Softmax lookup table
set_multicycle_path -setup 2 -from [get_registers {*softmax*lut*}]
set_multicycle_path -hold 1 -from [get_registers {*softmax*lut*}]

# =============================================================================
# Design Partitions (for incremental compilation)
# =============================================================================

# Vision encoder partition
# set_design_partition -name vision_encoder [get_entity_instances *vision_encoder*]

# Language model partition  
# set_design_partition -name language_model [get_entity_instances *language_model*]

# =============================================================================
# False Paths
# =============================================================================

set_false_path -to [get_registers {*heartbeat*}]
set_false_path -to [get_registers {*status_leds*}]
set_false_path -from [get_registers {*debug_reg*}]
