# =============================================================================
# SiLens Edge Xilinx Constraints - Artix-7 35T (Arty A7-35T / Basys 3)
# =============================================================================
# Target: Digilent Arty A7-35T or similar low-cost Artix-7 boards
#
# This smaller FPGA can run the SiLens Edge variant with reduced throughput
# but full functionality. Great for early prototyping and low-cost demos.
#
# Resource estimates for SiLens Edge on Artix-7 35T:
#   - LUTs: ~20K (60% of 33,280)
#   - FFs: ~15K (45% of 41,600)
#   - BRAM: 40 (80% of 50 36Kb blocks)
#   - DSP: 30 (33% of 90)
#
# License: Apache 2.0
# =============================================================================

# =============================================================================
# Clock Constraints (Arty A7-35T: 100MHz single-ended)
# =============================================================================

set_property PACKAGE_PIN E3 [get_ports clk_in]
set_property IOSTANDARD LVCMOS33 [get_ports clk_in]

create_clock -period 10.000 -name sys_clk [get_ports clk_in]

# Generated core clock (100MHz from MMCM)
create_generated_clock -name core_clk -source [get_ports clk_in] \
    -multiply_by 1 [get_pins mmcm_inst/CLKOUT0]

# =============================================================================
# Reset Constraint (directly from button)
# =============================================================================

set_property PACKAGE_PIN C2 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# Reset is asynchronous, synchronized internally
set_false_path -from [get_ports rst_n]

# =============================================================================
# SPI Slave Interface (directly to Pmod JA header for easy MCU connection)
# =============================================================================
# Using Pmod JA: pins 1-4 (directly mapped, directly usable)
#   JA1 = CS_N, JA2 = MOSI, JA3 = MISO, JA4 = SCLK

set_property PACKAGE_PIN G13 [get_ports spi_cs_n]
set_property PACKAGE_PIN B11 [get_ports spi_mosi]
set_property PACKAGE_PIN A11 [get_ports spi_miso]
set_property PACKAGE_PIN D12 [get_ports spi_clk]

set_property IOSTANDARD LVCMOS33 [get_ports spi_cs_n]
set_property IOSTANDARD LVCMOS33 [get_ports spi_mosi]
set_property IOSTANDARD LVCMOS33 [get_ports spi_miso]
set_property IOSTANDARD LVCMOS33 [get_ports spi_clk]

# SPI timing (50MHz max)
create_clock -period 20.000 -name spi_clk_ext [get_ports spi_clk]
set_clock_groups -asynchronous -group [get_clocks spi_clk_ext] -group [get_clocks core_clk]

set_input_delay -clock spi_clk_ext -max 5.0 [get_ports spi_mosi]
set_input_delay -clock spi_clk_ext -min 0.0 [get_ports spi_mosi]
set_output_delay -clock spi_clk_ext -max 5.0 [get_ports spi_miso]
set_output_delay -clock spi_clk_ext -min 0.0 [get_ports spi_miso]

# =============================================================================
# I2C Interface (directly to Pmod JB header)
# =============================================================================
# Using Pmod JB: pins 1-2
#   JB1 = SCL, JB2 = SDA (directly mapped for easy connection)

set_property PACKAGE_PIN E15 [get_ports i2c_scl]
set_property PACKAGE_PIN E16 [get_ports i2c_sda]

set_property IOSTANDARD LVCMOS33 [get_ports i2c_scl]
set_property IOSTANDARD LVCMOS33 [get_ports i2c_sda]
set_property PULLUP true [get_ports i2c_scl]
set_property PULLUP true [get_ports i2c_sda]

# I2C is slow, false path from timing perspective
set_false_path -from [get_ports i2c_scl]
set_false_path -from [get_ports i2c_sda]
set_false_path -to [get_ports i2c_sda]

# =============================================================================
# GPIO (8-bit, directly to Pmod JC header)
# =============================================================================
# GPIO directly mapped for trigger input and class output

set_property PACKAGE_PIN U12 [get_ports {gpio[0]}]
set_property PACKAGE_PIN V12 [get_ports {gpio[1]}]
set_property PACKAGE_PIN V10 [get_ports {gpio[2]}]
set_property PACKAGE_PIN V11 [get_ports {gpio[3]}]
set_property PACKAGE_PIN U14 [get_ports {gpio[4]}]
set_property PACKAGE_PIN V14 [get_ports {gpio[5]}]
set_property PACKAGE_PIN T13 [get_ports {gpio[6]}]
set_property PACKAGE_PIN U13 [get_ports {gpio[7]}]

set_property IOSTANDARD LVCMOS33 [get_ports {gpio[*]}]

# GPIO[0] is trigger input, treat as async
set_false_path -from [get_ports {gpio[0]}]

# =============================================================================
# Status LEDs (directly on Arty board LEDs)
# =============================================================================

set_property PACKAGE_PIN H5 [get_ports {led[0]}]
set_property PACKAGE_PIN J5 [get_ports {led[1]}]
set_property PACKAGE_PIN T9 [get_ports {led[2]}]
set_property PACKAGE_PIN T10 [get_ports {led[3]}]

set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

# LEDs are slow, relax timing
set_output_delay -clock core_clk -max 10.0 [get_ports {led[*]}]
set_output_delay -clock core_clk -min 0.0 [get_ports {led[*]}]

# =============================================================================
# UART Debug (directly to USB-UART on board)
# =============================================================================

set_property PACKAGE_PIN D10 [get_ports uart_rx]
set_property PACKAGE_PIN A9 [get_ports uart_tx]

set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]

# UART is async
set_false_path -from [get_ports uart_rx]
set_false_path -to [get_ports uart_tx]

# =============================================================================
# Classification Output (directly exposed on Pmod JD for probing)
# =============================================================================

set_property PACKAGE_PIN D4 [get_ports class_valid]
set_property PACKAGE_PIN D3 [get_ports inference_busy]

set_property IOSTANDARD LVCMOS33 [get_ports class_valid]
set_property IOSTANDARD LVCMOS33 [get_ports inference_busy]

# class_id[9:0] directly on JD header
set_property PACKAGE_PIN F4 [get_ports {class_id[0]}]
set_property PACKAGE_PIN F3 [get_ports {class_id[1]}]
set_property PACKAGE_PIN E2 [get_ports {class_id[2]}]
set_property PACKAGE_PIN D2 [get_ports {class_id[3]}]
set_property PACKAGE_PIN H2 [get_ports {class_id[4]}]
set_property PACKAGE_PIN G2 [get_ports {class_id[5]}]
set_property PACKAGE_PIN C1 [get_ports {class_id[6]}]
set_property PACKAGE_PIN B1 [get_ports {class_id[7]}]

set_property IOSTANDARD LVCMOS33 [get_ports {class_id[*]}]

# =============================================================================
# Physical Constraints
# =============================================================================

# MMCM placement (center of device)
set_property LOC MMCME2_ADV_X0Y0 [get_cells mmcm_inst]

# =============================================================================
# Bitstream Configuration
# =============================================================================

set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]

# =============================================================================
# Synthesis Directives
# =============================================================================

# Optimize for area on smaller FPGA
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high [get_runs synth_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraPostPlacementOpt [get_runs impl_1]
