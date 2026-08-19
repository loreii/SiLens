// =============================================================================
// SiLens Edge FPGA Wrapper for Xilinx FPGAs
// =============================================================================
// Simplified FPGA wrapper for the Edge variant (50mm² target).
// Uses SPI/I2C interface instead of PCIe for MCU integration.
//
// Target FPGAs:
//   - Xilinx Artix-7 35T (Arty A7-35T, Basys 3)
//   - Xilinx Artix-7 100T (Arty A7-100T, Nexys A7)
//   - Lattice iCE40 UP5K (low-cost, open toolchain)
//   - Lattice ECP5 (mid-range, open toolchain)
//
// Features:
//   - Simple PLL for clock generation (100-200MHz core)
//   - SPI slave interface (Mode 0, up to 50MHz)
//   - I2C slave interface (configuration)
//   - GPIO (8-bit) for triggers and status
//   - Optional UART debug interface
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_edge_fpga_wrapper #(
    // Clock parameters
    parameter INPUT_CLK_FREQ_MHZ  = 100,           // Input clock (board oscillator)
    parameter CORE_CLK_FREQ_MHZ   = 100,           // Core clock (scale with FPGA)
    
    // Model parameters (NanoViT-12M + Classifier)
    parameter HIDDEN_DIM          = 192,           // NanoViT hidden dimension
    parameter NUM_PATCHES         = 196,           // 14×14 patches
    parameter NUM_CLASSES         = 1000,          // Classification classes
    parameter IMAGE_SIZE          = 224,           // Input image size
    parameter PATCH_SIZE          = 16,            // Patch size
    parameter ACT_WIDTH           = 8,             // Activation width
    
    // IO parameters
    parameter GPIO_WIDTH          = 8,             // GPIO pins
    parameter I2C_ADDR            = 7'h50          // I2C slave address
)(
    // =========================================================================
    // Clock and Reset
    // =========================================================================
    input  wire         clk_in,             // Board oscillator (100MHz typical)
    input  wire         rst_n,              // Active-low reset (directly from button)
    
    // =========================================================================
    // SPI Slave Interface (to MCU host)
    // =========================================================================
    input  wire         spi_clk,            // SPI clock (up to 50MHz)
    input  wire         spi_mosi,           // Master Out Slave In
    output wire         spi_miso,           // Master In Slave Out
    input  wire         spi_cs_n,           // Chip select (active low)
    
    // =========================================================================
    // I2C Slave Interface (configuration)
    // =========================================================================
    input  wire         i2c_scl,            // I2C clock
    inout  wire         i2c_sda,            // I2C data (bidirectional)
    
    // =========================================================================
    // GPIO (directly mapped to FPGA pins)
    // =========================================================================
    inout  wire [GPIO_WIDTH-1:0] gpio,      // Bidirectional GPIO
    
    // =========================================================================
    // Status LEDs
    // =========================================================================
    output wire [3:0]   led,                // Status LEDs
    
    // =========================================================================
    // UART Debug (optional)
    // =========================================================================
    input  wire         uart_rx,
    output wire         uart_tx,
    
    // =========================================================================
    // Classification Output (directly exposed for easy probing)
    // =========================================================================
    output wire         class_valid,        // Classification result valid
    output wire [9:0]   class_id,           // Predicted class (0-1023)
    output wire         inference_busy      // Inference in progress
);

    // =========================================================================
    // Clock Generation (MMCM for Xilinx, HFOSC for iCE40)
    // =========================================================================
    
    wire clk_core;
    wire pll_locked;
    
    `ifdef XILINX
    // Xilinx MMCM-based PLL
    wire mmcm_feedback;
    
    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKFBOUT_MULT_F(10.0),                    // VCO = 100MHz * 10 = 1000MHz
        .CLKFBOUT_PHASE(0.0),
        .CLKIN1_PERIOD(10.0),                      // 100MHz input
        .CLKOUT0_DIVIDE_F(10.0),                   // 100MHz core clock
        .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT0_PHASE(0.0),
        .DIVCLK_DIVIDE(1),
        .REF_JITTER1(0.01),
        .STARTUP_WAIT("FALSE")
    ) mmcm_inst (
        .CLKOUT0(clk_core),
        .CLKOUT1(),
        .CLKOUT2(),
        .CLKOUT3(),
        .CLKOUT4(),
        .CLKOUT5(),
        .CLKOUT6(),
        .CLKFBOUT(mmcm_feedback),
        .LOCKED(pll_locked),
        .CLKIN1(clk_in),
        .PWRDWN(1'b0),
        .RST(~rst_n),
        .CLKFBIN(mmcm_feedback)
    );
    `else
    // Generic PLL placeholder (for simulation or other FPGAs)
    assign clk_core = clk_in;
    reg [3:0] lock_cnt = 0;
    reg locked_reg = 0;
    always @(posedge clk_in or negedge rst_n) begin
        if (!rst_n) begin
            lock_cnt <= 0;
            locked_reg <= 0;
        end else if (!locked_reg) begin
            lock_cnt <= lock_cnt + 1;
            if (lock_cnt == 4'hF) locked_reg <= 1;
        end
    end
    assign pll_locked = locked_reg;
    `endif
    
    // Global clock buffer (Xilinx-specific)
    `ifdef XILINX
    wire clk_core_buf;
    BUFG bufg_core (.I(clk_core), .O(clk_core_buf));
    `else
    wire clk_core_buf = clk_core;
    `endif
    
    // =========================================================================
    // Reset Synchronization
    // =========================================================================
    
    reg [3:0] rst_sync_r;
    wire      rst_n_sync;
    
    always @(posedge clk_core_buf or negedge rst_n) begin
        if (!rst_n) begin
            rst_sync_r <= 4'b0000;
        end else if (pll_locked) begin
            rst_sync_r <= {rst_sync_r[2:0], 1'b1};
        end else begin
            rst_sync_r <= 4'b0000;
        end
    end
    
    assign rst_n_sync = rst_sync_r[3];
    
    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // Control signals
    wire frame_start;
    wire abort;
    wire [1:0] mode;
    
    // Status signals
    wire busy;
    wire done;
    wire error;
    
    // Classification output
    wire [9:0] class_id_int;
    wire class_valid_int;
    
    // GPIO handling
    wire [GPIO_WIDTH-1:0] gpio_in;
    wire [GPIO_WIDTH-1:0] gpio_out;
    wire [GPIO_WIDTH-1:0] gpio_oe;
    
    genvar i;
    generate
        for (i = 0; i < GPIO_WIDTH; i = i + 1) begin : gpio_iobuf
            `ifdef XILINX
            IOBUF iobuf_inst (
                .O(gpio_in[i]),
                .IO(gpio[i]),
                .I(gpio_out[i]),
                .T(~gpio_oe[i])
            );
            `else
            assign gpio[i] = gpio_oe[i] ? gpio_out[i] : 1'bz;
            assign gpio_in[i] = gpio[i];
            `endif
        end
    endgenerate
    
    // =========================================================================
    // SiLens Edge SoC Core
    // =========================================================================
    
    silens_edge_soc #(
        .ACT_WIDTH(ACT_WIDTH),
        .HIDDEN_DIM(HIDDEN_DIM),
        .NUM_PATCHES(NUM_PATCHES),
        .NUM_CLASSES(NUM_CLASSES),
        .IMAGE_SIZE(IMAGE_SIZE),
        .PATCH_SIZE(PATCH_SIZE),
        .GPIO_WIDTH(GPIO_WIDTH)
    ) u_silens_edge_soc (
        // Clock and reset
        .clk_ref(clk_core_buf),
        .rst_n(rst_n_sync),
        
        // SPI interface
        .spi_clk(spi_clk),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .spi_cs_n(spi_cs_n),
        .spi_irq_n(),                           // Not directly exposed
        .spi_rdy(),                             // Not directly exposed
        
        // I2C interface
        .i2c_scl(i2c_scl),
        .i2c_sda(i2c_sda),
        
        // GPIO
        .gpio(gpio),
        
        // Classification output
        .class_valid(class_valid_int),
        .class_id(class_id_int),
        
        // Control
        .frame_start(frame_start),
        .abort(abort),
        .mode(mode),
        
        // Status
        .busy(busy),
        .done(done),
        .error(error),
        
        // Test mode
        .test_mode(1'b0)
    );
    
    // =========================================================================
    // Debug Controls (directly from GPIO[0] as trigger)
    // =========================================================================
    
    // GPIO[0] can trigger inference (directly from external signal)
    assign frame_start = gpio_in[0];
    assign abort = 1'b0;
    assign mode = 2'b00;
    
    // =========================================================================
    // Status LED Mapping
    // =========================================================================
    
    // LED[0]: PLL locked / system ready
    // LED[1]: Inference busy
    // LED[2]: Classification valid (latched)
    // LED[3]: Error indicator
    
    reg class_valid_latch;
    always @(posedge clk_core_buf or negedge rst_n_sync) begin
        if (!rst_n_sync)
            class_valid_latch <= 1'b0;
        else if (class_valid_int)
            class_valid_latch <= 1'b1;
        else if (frame_start)
            class_valid_latch <= 1'b0;
    end
    
    assign led[0] = pll_locked;
    assign led[1] = busy;
    assign led[2] = class_valid_latch;
    assign led[3] = error;
    
    // =========================================================================
    // Classification Output
    // =========================================================================
    
    assign class_valid = class_valid_int;
    assign class_id = class_id_int;
    assign inference_busy = busy;
    
    // =========================================================================
    // UART Debug Interface (optional, directly exposed)
    // =========================================================================
    
    // Simple loopback for now (or connect to debug module)
    assign uart_tx = uart_rx;  // Loopback or idle

endmodule

`default_nettype wire
