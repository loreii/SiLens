# =============================================================================
# SiLens Xilinx Constraints - Kintex-7 (XC7K325T-2FFG900)
# =============================================================================
# Target: KC705 or similar Kintex-7 development board
#
# License: Apache 2.0
# =============================================================================

# =============================================================================
# Clock Constraints
# =============================================================================

# 200 MHz differential system clock
set_property PACKAGE_PIN AD12 [get_ports clk_p]
set_property PACKAGE_PIN AD11 [get_ports clk_n]
set_property IOSTANDARD LVDS [get_ports clk_p]
set_property IOSTANDARD LVDS [get_ports clk_n]

create_clock -period 5.000 -name sys_clk [get_ports clk_p]

# PCIe reference clock (100 MHz)
set_property PACKAGE_PIN U8 [get_ports pcie_refclk_p]
set_property PACKAGE_PIN U7 [get_ports pcie_refclk_n]

create_clock -period 10.000 -name pcie_refclk [get_ports pcie_refclk_p]

# =============================================================================
# Generated Clocks
# =============================================================================

create_generated_clock -name core_clk -source [get_pins mmcm_inst/CLKIN1] \
    -divide_by 2 [get_pins mmcm_inst/CLKOUT0]

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
# Reset
# =============================================================================

set_property PACKAGE_PIN AB7 [get_ports sys_rst_n]
set_property IOSTANDARD LVCMOS15 [get_ports sys_rst_n]
set_false_path -from [get_ports sys_rst_n]

set_property PACKAGE_PIN L19 [get_ports pcie_rst_n]
set_property IOSTANDARD LVCMOS25 [get_ports pcie_rst_n]
set_false_path -from [get_ports pcie_rst_n]


# =============================================================================
# PCIe Lane Constraints (GTX Bank 118)
# =============================================================================

set_property PACKAGE_PIN M6 [get_ports {pcie_rx_p[0]}]
set_property PACKAGE_PIN M5 [get_ports {pcie_rx_n[0]}]
set_property PACKAGE_PIN L4 [get_ports {pcie_tx_p[0]}]
set_property PACKAGE_PIN L3 [get_ports {pcie_tx_n[0]}]

set_property PACKAGE_PIN P6 [get_ports {pcie_rx_p[1]}]
set_property PACKAGE_PIN P5 [get_ports {pcie_rx_n[1]}]
set_property PACKAGE_PIN N4 [get_ports {pcie_tx_p[1]}]
set_property PACKAGE_PIN N3 [get_ports {pcie_tx_n[1]}]

set_property PACKAGE_PIN R6 [get_ports {pcie_rx_p[2]}]
set_property PACKAGE_PIN R5 [get_ports {pcie_rx_n[2]}]
set_property PACKAGE_PIN P2 [get_ports {pcie_tx_p[2]}]
set_property PACKAGE_PIN P1 [get_ports {pcie_tx_n[2]}]

set_property PACKAGE_PIN T6 [get_ports {pcie_rx_p[3]}]
set_property PACKAGE_PIN T5 [get_ports {pcie_rx_n[3]}]
set_property PACKAGE_PIN R4 [get_ports {pcie_tx_p[3]}]
set_property PACKAGE_PIN R3 [get_ports {pcie_tx_n[3]}]

# =============================================================================
# User I/O
# =============================================================================

# Push buttons
set_property PACKAGE_PIN AA12 [get_ports {btn[0]}]
set_property PACKAGE_PIN AB12 [get_ports {btn[1]}]
set_property PACKAGE_PIN AC6 [get_ports {btn[2]}]
set_property PACKAGE_PIN AG5 [get_ports {btn[3]}]
set_property IOSTANDARD LVCMOS15 [get_ports {btn[*]}]
set_false_path -from [get_ports {btn[*]}]

# DIP switches
set_property PACKAGE_PIN Y29 [get_ports {sw[0]}]
set_property PACKAGE_PIN W29 [get_ports {sw[1]}]
set_property PACKAGE_PIN AA28 [get_ports {sw[2]}]
set_property PACKAGE_PIN Y28 [get_ports {sw[3]}]
set_property IOSTANDARD LVCMOS25 [get_ports {sw[*]}]
set_false_path -from [get_ports {sw[*]}]

# LEDs
set_property PACKAGE_PIN AB8 [get_ports {led[0]}]
set_property PACKAGE_PIN AA8 [get_ports {led[1]}]
set_property PACKAGE_PIN AC9 [get_ports {led[2]}]
set_property PACKAGE_PIN AB9 [get_ports {led[3]}]
set_property PACKAGE_PIN AE26 [get_ports {led[4]}]
set_property PACKAGE_PIN G19 [get_ports {led[5]}]
set_property PACKAGE_PIN E18 [get_ports {led[6]}]
set_property PACKAGE_PIN F16 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS15 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS15 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS15 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS15 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[4]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[5]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[6]}]
set_property IOSTANDARD LVCMOS25 [get_ports {led[7]}]


# =============================================================================
# UART
# =============================================================================

set_property PACKAGE_PIN K24 [get_ports uart_rx]
set_property PACKAGE_PIN M19 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS25 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS25 [get_ports uart_tx]

# =============================================================================
# DDR3 Memory (Bank 32/33/34)
# =============================================================================

set_property INTERNAL_VREF 0.750 [get_iobanks 32]
set_property INTERNAL_VREF 0.750 [get_iobanks 33]
set_property INTERNAL_VREF 0.750 [get_iobanks 34]

# =============================================================================
# Physical Constraints
# =============================================================================

set_property LOC MMCME2_ADV_X1Y2 [get_cells mmcm_inst]
set_property LOC IDELAYCTRL_X1Y2 [get_cells idelayctrl_inst]

# =============================================================================
# Bitstream Configuration
# =============================================================================

set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property CONFIG_VOLTAGE 2.5 [current_design]
set_property CFGBVS GND [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
