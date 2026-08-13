# =============================================================================
# SiLens Xilinx Constraints - Artix-7 (XC7A200T-1FBG676)
# =============================================================================
# Target: Nexys Video or similar Artix-7 200T development board
#
# License: Apache 2.0
# =============================================================================

# =============================================================================
# Clock Constraints
# =============================================================================

# 200 MHz differential system clock (LVDS)
set_property PACKAGE_PIN R4 [get_ports clk_p]
set_property PACKAGE_PIN T4 [get_ports clk_n]
set_property IOSTANDARD LVDS_25 [get_ports clk_p]
set_property IOSTANDARD LVDS_25 [get_ports clk_n]

create_clock -period 5.000 -name sys_clk [get_ports clk_p]

# PCIe reference clock (100 MHz differential)
set_property PACKAGE_PIN F10 [get_ports pcie_refclk_p]
set_property PACKAGE_PIN E10 [get_ports pcie_refclk_n]

create_clock -period 10.000 -name pcie_refclk [get_ports pcie_refclk_p]

# =============================================================================
# Generated Clock Constraints
# =============================================================================

# Core clock from MMCM (100 MHz)
create_generated_clock -name core_clk -source [get_pins mmcm_inst/CLKIN1] \
    -divide_by 2 [get_pins mmcm_inst/CLKOUT0]

# PCIe clock from MMCM (250 MHz)
create_generated_clock -name pcie_clk -source [get_pins mmcm_inst/CLKIN1] \
    -multiply_by 5 -divide_by 4 [get_pins mmcm_inst/CLKOUT1]

# =============================================================================
# Clock Domain Crossing
# =============================================================================

set_clock_groups -asynchronous \
    -group [get_clocks sys_clk] \
    -group [get_clocks pcie_refclk]

set_clock_groups -asynchronous \
    -group [get_clocks core_clk] \
    -group [get_clocks pcie_clk]

# =============================================================================
# Reset Constraints
# =============================================================================

set_property PACKAGE_PIN G4 [get_ports sys_rst_n]
set_property IOSTANDARD LVCMOS15 [get_ports sys_rst_n]

set_property PACKAGE_PIN F11 [get_ports pcie_rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports pcie_rst_n]

# Reset is asynchronous, synchronize internally
set_false_path -from [get_ports sys_rst_n]
set_false_path -from [get_ports pcie_rst_n]

# =============================================================================
# PCIe Lane Constraints (GTX Transceivers)
# =============================================================================

# PCIe x4 lanes - Bank 116
set_property PACKAGE_PIN B6 [get_ports {pcie_rx_p[0]}]
set_property PACKAGE_PIN A6 [get_ports {pcie_rx_n[0]}]
set_property PACKAGE_PIN B4 [get_ports {pcie_tx_p[0]}]
set_property PACKAGE_PIN A4 [get_ports {pcie_tx_n[0]}]

set_property PACKAGE_PIN D5 [get_ports {pcie_rx_p[1]}]
set_property PACKAGE_PIN C5 [get_ports {pcie_rx_n[1]}]
set_property PACKAGE_PIN D3 [get_ports {pcie_tx_p[1]}]
set_property PACKAGE_PIN C3 [get_ports {pcie_tx_n[1]}]

set_property PACKAGE_PIN F5 [get_ports {pcie_rx_p[2]}]
set_property PACKAGE_PIN E5 [get_ports {pcie_rx_n[2]}]
set_property PACKAGE_PIN F3 [get_ports {pcie_tx_p[2]}]
set_property PACKAGE_PIN E3 [get_ports {pcie_tx_n[2]}]

set_property PACKAGE_PIN H5 [get_ports {pcie_rx_p[3]}]
set_property PACKAGE_PIN G5 [get_ports {pcie_rx_n[3]}]
set_property PACKAGE_PIN H3 [get_ports {pcie_tx_p[3]}]
set_property PACKAGE_PIN G3 [get_ports {pcie_tx_n[3]}]

# =============================================================================
# User I/O Constraints
# =============================================================================

# Push buttons
set_property PACKAGE_PIN B22 [get_ports {btn[0]}]
set_property PACKAGE_PIN D22 [get_ports {btn[1]}]
set_property PACKAGE_PIN C22 [get_ports {btn[2]}]
set_property PACKAGE_PIN D14 [get_ports {btn[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[*]}]

# Switches
set_property PACKAGE_PIN E22 [get_ports {sw[0]}]
set_property PACKAGE_PIN F21 [get_ports {sw[1]}]
set_property PACKAGE_PIN G21 [get_ports {sw[2]}]
set_property PACKAGE_PIN G22 [get_ports {sw[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[*]}]

# LEDs
set_property PACKAGE_PIN T14 [get_ports {led[0]}]
set_property PACKAGE_PIN T15 [get_ports {led[1]}]
set_property PACKAGE_PIN T16 [get_ports {led[2]}]
set_property PACKAGE_PIN U16 [get_ports {led[3]}]
set_property PACKAGE_PIN V15 [get_ports {led[4]}]
set_property PACKAGE_PIN W16 [get_ports {led[5]}]
set_property PACKAGE_PIN W15 [get_ports {led[6]}]
set_property PACKAGE_PIN Y13 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

# =============================================================================
# UART Constraints
# =============================================================================

set_property PACKAGE_PIN AA19 [get_ports uart_rx]
set_property PACKAGE_PIN V18 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]

# =============================================================================
# DDR3 Memory Constraints (MIG-generated, placeholder pins)
# =============================================================================

# DDR3 uses dedicated SSTL15 banks - actual pins generated by MIG
set_property INTERNAL_VREF 0.750 [get_iobanks 34]

# =============================================================================
# Timing Exceptions
# =============================================================================

# Button inputs are asynchronous
set_input_delay -clock [get_clocks core_clk] -min 0.0 [get_ports {btn[*]}]
set_input_delay -clock [get_clocks core_clk] -max 5.0 [get_ports {btn[*]}]
set_false_path -from [get_ports {btn[*]}]

# Switch inputs are asynchronous
set_input_delay -clock [get_clocks core_clk] -min 0.0 [get_ports {sw[*]}]
set_input_delay -clock [get_clocks core_clk] -max 5.0 [get_ports {sw[*]}]
set_false_path -from [get_ports {sw[*]}]

# LED outputs can be slow
set_output_delay -clock [get_clocks core_clk] -min 0.0 [get_ports {led[*]}]
set_output_delay -clock [get_clocks core_clk] -max 5.0 [get_ports {led[*]}]

# =============================================================================
# Physical Constraints
# =============================================================================

# MMCM placement
set_property LOC MMCME2_ADV_X1Y2 [get_cells mmcm_inst]

# IDELAYCTRL placement
set_property LOC IDELAYCTRL_X1Y2 [get_cells idelayctrl_inst]

# =============================================================================
# Bitstream Configuration
# =============================================================================

set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]

# =============================================================================
# Debug Constraints (ILA)
# =============================================================================

# Reserve clock for ILA
# set_property C_CLK_INPUT_FREQ_HZ 100000000 [get_debug_cores dbg_hub]
# set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
