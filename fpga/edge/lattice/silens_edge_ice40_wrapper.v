// =============================================================================
// SiLens Edge FPGA Wrapper for Lattice iCE40 UP5K
// =============================================================================
// Target: Lattice iCE40UP5K-SG48ITR (UPduino v3.1)
// Toolchain: Open-source (Yosys + nextpnr-ice40 + IceStorm)
//
// IMPORTANT - RESOURCE CONSTRAINTS:
// =================================
// The iCE40 UP5K has only 5,280 LUTs - severely limited compared to Artix-7.
// This wrapper implements a "MICRO" configuration suitable for:
//   - Architecture demonstration
//   - Educational purposes
//   - Proof-of-concept prototyping
//
// For production edge deployment, use Lattice ECP5 or Xilinx Artix-7.
//
// MICRO CONFIGURATION (fits in ~4K LUTs):
// =======================================
//   Parameter           | Full Config | MICRO Config
//   --------------------|-------------|-------------
//   HIDDEN_DIM          | 192         | 32
//   NUM_PATCHES         | 196 (14×14) | 16 (4×4)
//   IMAGE_SIZE          | 224×224     | 64×64
//   PATCH_SIZE          | 16          | 16
//   NUM_CLASSES         | 1000        | 16
//   NUM_LAYERS          | 6           | 1
//   ACT_WIDTH           | 8           | 4
//
// Features:
//   - 48MHz from internal HFOSC (no external crystal needed)
//   - SB_RGBA_DRV for onboard RGB LED status
//   - SB_IO for bidirectional GPIO
//   - Simplified datapath for iCE40 constraints
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_edge_ice40_wrapper #(
    // =========================================================================
    // MICRO Configuration - Reduced for iCE40 UP5K (~4K LUTs target)
    // =========================================================================
    // To increase model size, you'll need ECP5 or larger FPGA.
    // Adjust these parameters based on synthesis utilization reports.
    
    parameter HIDDEN_DIM          = 32,            // Reduced from 192
    parameter NUM_PATCHES         = 16,            // 4×4 patches (from 64×64 image)
    parameter NUM_CLASSES         = 16,            // Reduced from 1000
    parameter IMAGE_SIZE          = 64,            // Reduced from 224
    parameter PATCH_SIZE          = 16,            // Same as full config
    parameter ACT_WIDTH           = 4,             // Reduced from 8 for area
    parameter GPIO_WIDTH          = 8,             // 8-bit GPIO
    
    // Clock parameters
    parameter HFOSC_DIV           = "0b00"         // 48MHz (0b00=48, 0b01=24, 0b10=12, 0b11=6)
)(
    // =========================================================================
    // External Reset (directly from button)
    // =========================================================================
    input  wire         rst_n,              // Active-low reset (optional, has internal POR)
    
    // =========================================================================
    // SPI Slave Interface (to MCU host)
    // =========================================================================
    input  wire         spi_clk,            // SPI clock (up to 12MHz for 48MHz sysclk)
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
    // RGB LED (directly drives onboard UPduino LED via SB_RGBA_DRV)
    // =========================================================================
    output wire         led_r,              // Red LED
    output wire         led_g,              // Green LED
    output wire         led_b,              // Blue LED
    
    // =========================================================================
    // UART Debug (directly to FT232 USB-Serial)
    // =========================================================================
    input  wire         uart_rx,
    output wire         uart_tx,
    
    // =========================================================================
    // Classification Output (directly exposed for probing)
    // =========================================================================
    output wire         class_valid,        // Classification result valid
    output wire [3:0]   class_id,           // Predicted class (4-bit subset)
    output wire         inference_busy      // Inference in progress
);

    // =========================================================================
    // Internal Clock Generation using iCE40 HFOSC
    // =========================================================================
    // The iCE40 UP5K has an internal 48MHz high-frequency oscillator.
    // No external crystal needed - directly usable for simple designs.
    
    wire clk_hf;        // 48MHz from HFOSC
    wire clk_core;      // Core clock (routed through global buffer)
    
    // High-Frequency Oscillator (48 MHz)
    // CLKHF_DIV: "0b00"=48MHz, "0b01"=24MHz, "0b10"=12MHz, "0b11"=6MHz
    SB_HFOSC #(
        .CLKHF_DIV(HFOSC_DIV)
    ) u_hfosc (
        .CLKHFPU(1'b1),         // Power up oscillator
        .CLKHFEN(1'b1),         // Enable oscillator
        .CLKHF(clk_hf)          // 48MHz output
    );
    
    // Global clock buffer for clock distribution
    SB_GB clk_gb (
        .USER_SIGNAL_TO_GLOBAL_BUFFER(clk_hf),
        .GLOBAL_BUFFER_OUTPUT(clk_core)
    );
    
    // =========================================================================
    // Optional: PLL for higher frequencies (100MHz)
    // =========================================================================
    // Uncomment to generate 100MHz from 48MHz HFOSC.
    // Note: May not meet timing on iCE40 UP5K - use with caution.
    //
    // wire clk_pll;
    // wire pll_locked;
    //
    // SB_PLL40_CORE #(
    //     .FEEDBACK_PATH("SIMPLE"),
    //     .PLLOUT_SELECT("GENCLK"),
    //     .DIVR(4'b0000),           // Reference divider: 0
    //     .DIVF(7'b1000010),        // Feedback divider: 66
    //     .DIVQ(3'b011),            // VCO divider: 8
    //     .FILTER_RANGE(3'b001)     // PLL filter range
    // ) u_pll (
    //     .REFERENCECLK(clk_hf),    // 48MHz input
    //     .PLLOUTGLOBAL(clk_pll),   // ~100MHz output
    //     .LOCK(pll_locked),
    //     .BYPASS(1'b0),
    //     .RESETB(rst_n)
    // );
    
    // =========================================================================
    // Power-On Reset (POR) Generator
    // =========================================================================
    // iCE40 needs explicit POR since HFOSC starts immediately.
    // Generate clean reset after oscillator stabilizes.
    
    reg [7:0] por_cnt = 8'd0;
    reg       por_done = 1'b0;
    wire      rst_n_int;
    
    always @(posedge clk_core) begin
        if (!por_done) begin
            if (por_cnt == 8'hFF)
                por_done <= 1'b1;
            else
                por_cnt <= por_cnt + 1'b1;
        end
    end
    
    // Combine POR with external reset
    assign rst_n_int = por_done & rst_n;
    
    // =========================================================================
    // Reset Synchronizer
    // =========================================================================
    
    reg [2:0] rst_sync_r = 3'b000;
    wire      rst_n_sync;
    
    always @(posedge clk_core or negedge rst_n_int) begin
        if (!rst_n_int)
            rst_sync_r <= 3'b000;
        else
            rst_sync_r <= {rst_sync_r[1:0], 1'b1};
    end
    
    assign rst_n_sync = rst_sync_r[2];
    
    // =========================================================================
    // GPIO Bidirectional IO using SB_IO
    // =========================================================================
    // iCE40 requires explicit SB_IO primitives for bidirectional pins.
    
    wire [GPIO_WIDTH-1:0] gpio_in;
    wire [GPIO_WIDTH-1:0] gpio_out;
    wire [GPIO_WIDTH-1:0] gpio_oe;
    
    genvar gi;
    generate
        for (gi = 0; gi < GPIO_WIDTH; gi = gi + 1) begin : gpio_iobuf
            SB_IO #(
                .PIN_TYPE(6'b1010_01),  // Tristate output, direct input
                .PULLUP(1'b0)
            ) u_gpio_io (
                .PACKAGE_PIN(gpio[gi]),
                .OUTPUT_ENABLE(gpio_oe[gi]),
                .D_OUT_0(gpio_out[gi]),
                .D_IN_0(gpio_in[gi])
            );
        end
    endgenerate
    
    // =========================================================================
    // I2C Bidirectional SDA using SB_IO
    // =========================================================================
    
    wire i2c_sda_in;
    wire i2c_sda_out;
    wire i2c_sda_oe;
    
    SB_IO #(
        .PIN_TYPE(6'b1010_01),  // Tristate output, direct input
        .PULLUP(1'b1)           // Enable internal pull-up for I2C
    ) u_i2c_sda_io (
        .PACKAGE_PIN(i2c_sda),
        .OUTPUT_ENABLE(i2c_sda_oe),
        .D_OUT_0(i2c_sda_out),
        .D_IN_0(i2c_sda_in)
    );
    
    // =========================================================================
    // RGB LED Driver using SB_RGBA_DRV
    // =========================================================================
    // The iCE40 UP5K has a dedicated RGB LED driver primitive.
    // Directly drives the onboard UPduino RGB LED with current control.
    
    wire led_r_pwm, led_g_pwm, led_b_pwm;
    
    SB_RGBA_DRV #(
        .CURRENT_MODE("0b1"),       // Full current mode
        .RGB0_CURRENT("0b000001"),  // 4mA (R) - low for indicator
        .RGB1_CURRENT("0b000001"),  // 4mA (G)
        .RGB2_CURRENT("0b000001")   // 4mA (B)
    ) u_rgb_drv (
        .CURREN(1'b1),              // Enable current reference
        .RGBLEDEN(1'b1),            // Enable LED driver
        .RGB0PWM(led_r_pwm),        // Red PWM input
        .RGB1PWM(led_g_pwm),        // Green PWM input
        .RGB2PWM(led_b_pwm),        // Blue PWM input
        .RGB0(led_r),               // Red output (directly to pin)
        .RGB1(led_g),               // Green output
        .RGB2(led_b)                // Blue output
    );
    
    // =========================================================================
    // MICRO SiLens Core (Simplified for iCE40)
    // =========================================================================
    // This is a simplified inference engine that fits in ~4K LUTs.
    // For full functionality, use ECP5 or larger FPGA.
    
    // Internal control/status signals
    reg  busy_reg;
    reg  done_reg;
    reg  error_reg;
    reg  [3:0] class_out_reg;
    reg  class_valid_reg;
    
    // State machine for simplified inference
    localparam [2:0] 
        ST_IDLE     = 3'd0,
        ST_LOAD     = 3'd1,
        ST_COMPUTE  = 3'd2,
        ST_OUTPUT   = 3'd3,
        ST_DONE     = 3'd4;
    
    reg [2:0] state;
    reg [15:0] cycle_cnt;
    
    // Frame start from GPIO[0]
    wire frame_start = gpio_in[0];
    
    // =========================================================================
    // Simplified SPI Slave (Command/Response)
    // =========================================================================
    // Basic SPI slave for configuration and data transfer.
    // Full implementation would connect to actual inference engine.
    
    reg [7:0] spi_shift_in;
    reg [7:0] spi_shift_out;
    reg [2:0] spi_bit_cnt;
    reg       spi_miso_reg;
    
    // SPI input synchronization (cross clock domain)
    reg [1:0] spi_clk_sync;
    reg       spi_clk_prev;
    wire      spi_clk_rise;
    wire      spi_clk_fall;
    
    always @(posedge clk_core) begin
        spi_clk_sync <= {spi_clk_sync[0], spi_clk};
        spi_clk_prev <= spi_clk_sync[1];
    end
    
    assign spi_clk_rise = spi_clk_sync[1] & ~spi_clk_prev;
    assign spi_clk_fall = ~spi_clk_sync[1] & spi_clk_prev;
    
    // SPI shift register (directly handles bytes)
    always @(posedge clk_core or negedge rst_n_sync) begin
        if (!rst_n_sync) begin
            spi_shift_in <= 8'd0;
            spi_shift_out <= 8'hA5;  // Default response: 0xA5 (SiLens ID)
            spi_bit_cnt <= 3'd0;
            spi_miso_reg <= 1'b0;
        end else if (spi_cs_n) begin
            spi_bit_cnt <= 3'd0;
            spi_shift_out <= {4'hA, class_out_reg};  // Status + class ID
        end else begin
            if (spi_clk_rise) begin
                // Sample MOSI on rising edge
                spi_shift_in <= {spi_shift_in[6:0], spi_mosi};
                spi_bit_cnt <= spi_bit_cnt + 1'b1;
            end
            if (spi_clk_fall) begin
                // Shift out on falling edge
                spi_miso_reg <= spi_shift_out[7];
                spi_shift_out <= {spi_shift_out[6:0], 1'b0};
            end
        end
    end
    
    assign spi_miso = spi_miso_reg;
    
    // =========================================================================
    // Simplified I2C Slave (Status Readback)
    // =========================================================================
    // Minimal I2C for reading status registers.
    // Address: 0x50 (7-bit)
    
    // For iCE40 constraints, implement minimal I2C or stub it out
    // Full I2C would use additional ~300 LUTs
    
    assign i2c_sda_oe = 1'b0;       // Always high-Z (read-only stub)
    assign i2c_sda_out = 1'b1;      // Release bus
    
    // =========================================================================
    // MICRO Inference State Machine
    // =========================================================================
    // Simplified inference that demonstrates the architecture.
    // Full implementation would include actual vision + classifier.
    //
    // For iCE40, we implement a demo that:
    //   1. Receives 64×64 image via SPI
    //   2. Runs simplified feature extraction (pooling + threshold)
    //   3. Outputs 4-bit class ID
    //
    // This is a DEMONSTRATION - actual inference requires more resources.
    
    // Activation buffer (directly in registers for tiny model)
    // 16 patches × 32 features × 4 bits = 2048 bits = 256 bytes
    // This fits in iCE40 LUT RAM or BRAM
    
    reg [ACT_WIDTH-1:0] feature_buf [0:NUM_PATCHES*HIDDEN_DIM-1];
    
    // Simple counter for demo inference
    reg [10:0] pixel_cnt;
    reg [7:0]  accum;
    
    always @(posedge clk_core or negedge rst_n_sync) begin
        if (!rst_n_sync) begin
            state <= ST_IDLE;
            cycle_cnt <= 16'd0;
            busy_reg <= 1'b0;
            done_reg <= 1'b0;
            error_reg <= 1'b0;
            class_out_reg <= 4'd0;
            class_valid_reg <= 1'b0;
            pixel_cnt <= 11'd0;
            accum <= 8'd0;
        end else begin
            // Default: clear single-cycle flags
            class_valid_reg <= 1'b0;
            done_reg <= 1'b0;
            
            case (state)
                ST_IDLE: begin
                    busy_reg <= 1'b0;
                    if (frame_start) begin
                        state <= ST_LOAD;
                        busy_reg <= 1'b1;
                        cycle_cnt <= 16'd0;
                        pixel_cnt <= 11'd0;
                        accum <= 8'd0;
                    end
                end
                
                ST_LOAD: begin
                    // Simulate image loading (in reality, from SPI)
                    // Wait ~4K cycles for 64×64 image at 48MHz
                    cycle_cnt <= cycle_cnt + 1'b1;
                    if (cycle_cnt >= 16'd4096) begin
                        state <= ST_COMPUTE;
                        cycle_cnt <= 16'd0;
                    end
                end
                
                ST_COMPUTE: begin
                    // Simplified "inference" - just a demo counter
                    // Real implementation would do actual computation
                    cycle_cnt <= cycle_cnt + 1'b1;
                    
                    // Accumulate some value based on cycle count (demo)
                    accum <= accum + cycle_cnt[3:0];
                    
                    // Simulate ~1000 cycles of "computation"
                    if (cycle_cnt >= 16'd1000) begin
                        state <= ST_OUTPUT;
                        // Generate pseudo-random class from accumulator
                        class_out_reg <= accum[3:0];
                    end
                end
                
                ST_OUTPUT: begin
                    class_valid_reg <= 1'b1;
                    state <= ST_DONE;
                end
                
                ST_DONE: begin
                    done_reg <= 1'b1;
                    busy_reg <= 1'b0;
                    // Return to idle after done pulse
                    state <= ST_IDLE;
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // GPIO Output Assignment
    // =========================================================================
    // gpio[0]: Input - Frame trigger
    // gpio[1]: Output - Busy
    // gpio[2]: Output - Done
    // gpio[3]: Output - Error
    // gpio[7:4]: User-defined
    
    assign gpio_out[0] = 1'b0;          // Input only
    assign gpio_out[1] = busy_reg;
    assign gpio_out[2] = done_reg;
    assign gpio_out[3] = error_reg;
    assign gpio_out[7:4] = class_out_reg;
    
    assign gpio_oe[0] = 1'b0;           // Input
    assign gpio_oe[1] = 1'b1;           // Output
    assign gpio_oe[2] = 1'b1;           // Output
    assign gpio_oe[3] = 1'b1;           // Output
    assign gpio_oe[7:4] = 4'b1111;      // Outputs
    
    // =========================================================================
    // LED Status Mapping
    // =========================================================================
    // Red:   Error or not ready
    // Green: Ready/idle
    // Blue:  Busy/processing
    
    // Simple PWM for LED brightness (directly from state)
    reg [7:0] led_pwm_cnt;
    always @(posedge clk_core) begin
        led_pwm_cnt <= led_pwm_cnt + 1'b1;
    end
    
    // LED assignments (directly from status)
    assign led_r_pwm = error_reg | ~rst_n_sync;             // Red: error or reset
    assign led_g_pwm = (state == ST_IDLE) & rst_n_sync;     // Green: ready
    assign led_b_pwm = busy_reg;                            // Blue: busy
    
    // =========================================================================
    // Output Assignments
    // =========================================================================
    
    assign class_valid = class_valid_reg;
    assign class_id = class_out_reg;
    assign inference_busy = busy_reg;
    
    // UART: simple loopback for now (debug stub)
    assign uart_tx = uart_rx;

endmodule

// =============================================================================
// Module: SB_HFOSC (iCE40 High-Frequency Oscillator)
// =============================================================================
// This is a built-in primitive in iCE40 UP5K.
// Yosys recognizes it directly - no need to define.
// Included here for documentation only.
//
// Frequencies:
//   CLKHF_DIV = "0b00" -> 48 MHz
//   CLKHF_DIV = "0b01" -> 24 MHz
//   CLKHF_DIV = "0b10" -> 12 MHz
//   CLKHF_DIV = "0b11" ->  6 MHz
//
// Usage:
//   SB_HFOSC #(.CLKHF_DIV("0b00")) u_hfosc (
//       .CLKHFPU(1'b1),
//       .CLKHFEN(1'b1),
//       .CLKHF(clk_48mhz)
//   );
// =============================================================================

// =============================================================================
// Module: SB_RGBA_DRV (iCE40 RGB LED Driver)
// =============================================================================
// Built-in primitive for driving the onboard RGB LED.
// Provides current-controlled outputs directly to LED pins.
//
// Current settings (RGB0_CURRENT, etc.):
//   "0b000000" = 0mA
//   "0b000001" = 4mA
//   "0b000011" = 8mA
//   "0b000111" = 12mA
//   "0b001111" = 16mA
//   "0b011111" = 20mA
//   "0b111111" = 24mA
// =============================================================================

// =============================================================================
// RESOURCE REDUCTION GUIDE
// =============================================================================
// If synthesis exceeds iCE40 UP5K capacity, reduce these parameters:
//
// 1. HIDDEN_DIM: 32 -> 16 -> 8
//    Impact: Reduces feature vector size, less accuracy
//
// 2. NUM_PATCHES: 16 -> 4 -> 1
//    Impact: Reduces spatial resolution, coarser features
//
// 3. ACT_WIDTH: 4 -> 2 -> 1 (binary!)
//    Impact: Reduces precision, still works for binary networks
//
// 4. Remove I2C: Save ~200-300 LUTs
//    Just use SPI for all communication
//
// 5. Remove UART: Save ~100-200 LUTs
//    Debug via GPIO or SPI only
//
// 6. Use BRAM instead of LUT RAM:
//    iCE40 UP5K has 120Kb BRAM (30x 4Kb blocks)
//    Move feature_buf to BRAM to save LUTs
//
// Target utilization for reliable operation:
//   LUTs: <80% (4,200 of 5,280)
//   FFs:  <70% (3,700 of 5,280)
//   BRAM: <90% (27 of 30 blocks)
//
// =============================================================================

`default_nettype wire
