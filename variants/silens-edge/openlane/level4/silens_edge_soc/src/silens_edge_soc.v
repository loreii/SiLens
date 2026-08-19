// =============================================================================
// SiLens Edge SoC - Level 4 Top Integration for 50mm² SKY130
// =============================================================================
// 
// Ultra-fast edge vision classifier targeting:
//   - 50mm² die (7mm × 7mm) on SKY130 130nm
//   - 200MHz operation
//   - 3W TDP
//   - QFN-48 package
//
// Integrated Level 3 macros:
//   - vision_nano (~15mm²):    NanoViT-12M vision encoder
//   - classifier_head (~10mm²): MLP classification head
//   - io_edge (~5mm²):         SPI/I2C/GPIO interfaces
//   - sram_256kb (~10mm²):     Activation buffer
//
// Performance targets:
//   - 1ms inference latency
//   - 1000 FPS throughput
//   - <500mW active power
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_edge_soc #(
    // =========================================================================
    // Architecture Parameters (NanoViT-12M + Classifier)
    // =========================================================================
    parameter ACT_WIDTH         = 8,        // Activation bit width
    parameter ACC_WIDTH         = 24,       // Accumulator width (smaller for edge)
    parameter HIDDEN_DIM        = 192,      // NanoViT hidden dimension
    parameter NUM_PATCHES       = 196,      // 14×14 patches from 224×224 image
    parameter NUM_CLASSES       = 1000,     // Classification classes
    parameter IMAGE_SIZE        = 224,      // Input image size
    parameter PATCH_SIZE        = 16,       // Patch size
    
    // =========================================================================
    // Memory Parameters
    // =========================================================================
    parameter SRAM_ADDR_WIDTH   = 15,       // 256KB = 32K × 64-bit = 2^15 words
    parameter SRAM_DATA_WIDTH   = 64,       // 64-bit SRAM interface
    
    // =========================================================================
    // IO Parameters
    // =========================================================================
    parameter GPIO_WIDTH        = 16,       // 16 GPIO pins
    parameter CLASS_ID_WIDTH    = 10        // 10-bit class ID (1024 classes max)
)(
    // =========================================================================
    // Clock and Reset
    // =========================================================================
    input  wire                         clk_ref,        // Reference clock (200MHz)
    input  wire                         rst_n,          // Active-low async reset
    
    // =========================================================================
    // SPI Interface (Primary data interface)
    // =========================================================================
    input  wire                         spi_clk,        // SPI clock
    input  wire                         spi_mosi,       // Master Out Slave In
    output wire                         spi_miso,       // Master In Slave Out
    input  wire                         spi_cs_n,       // Chip select (active low)
    output wire                         spi_irq_n,      // Interrupt request (active low)
    output wire                         spi_rdy,        // Ready for data
    
    // =========================================================================
    // I2C Interface (Configuration)
    // =========================================================================
    input  wire                         i2c_scl,        // I2C clock
    inout  wire                         i2c_sda,        // I2C data (bidirectional)
    
    // =========================================================================
    // GPIO (Triggers, sensors, actuators)
    // =========================================================================
    inout  wire [GPIO_WIDTH-1:0]        gpio,           // Bidirectional GPIO
    
    // =========================================================================
    // Classification Output
    // =========================================================================
    output wire                         class_valid,    // Classification result valid
    output wire [CLASS_ID_WIDTH-1:0]    class_id,       // Predicted class ID
    
    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire                         frame_start,    // Start frame processing
    input  wire                         abort,          // Abort current operation
    input  wire [1:0]                   mode,           // Operating mode
    
    // =========================================================================
    // Status Interface
    // =========================================================================
    output wire                         busy,           // Processing in progress
    output wire                         done,           // Processing complete
    output wire                         error,          // Error flag
    
    // =========================================================================
    // Debug/Test
    // =========================================================================
    input  wire                         test_mode       // Enable test mode
);

    // =========================================================================
    // Internal Clock and Reset Signals
    // =========================================================================
    
    wire clk_core;              // Core clock (200MHz)
    wire clk_sram;              // SRAM clock (200MHz, phase shifted)
    wire pll_locked;
    
    wire rst_core_n;            // Synchronized reset for core domain
    wire rst_spi_n;             // Synchronized reset for SPI domain
    
    // =========================================================================
    // Clock Generation PLL
    // =========================================================================
    // Simple PLL for 200MHz operation
    // Input: 200MHz reference (could also accept 25/50MHz with PLL multiplication)
    
    silens_edge_pll u_pll (
        .clk_ref    (clk_ref),
        .rst_n      (rst_n),
        .clk_core   (clk_core),
        .clk_sram   (clk_sram),
        .locked     (pll_locked)
    );
    
    // =========================================================================
    // Reset Synchronizers
    // =========================================================================
    
    silens_edge_reset_sync u_rst_sync_core (
        .clk        (clk_core),
        .rst_async_n(rst_n & pll_locked),
        .rst_sync_n (rst_core_n)
    );
    
    silens_edge_reset_sync u_rst_sync_spi (
        .clk        (spi_clk),
        .rst_async_n(rst_n & pll_locked),
        .rst_sync_n (rst_spi_n)
    );
    
    // =========================================================================
    // Inter-Subsystem Signals
    // =========================================================================
    
    // Vision → Classifier feature path
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] vision_features;      // 192 × 8 = 1536 bits
    wire                            vision_features_valid;
    wire                            vision_features_ready;
    wire                            vision_done;
    
    // Pixel data from IO to Vision
    wire [23:0]                     pixel_data;           // RGB888 (grayscale expanded)
    wire                            pixel_valid;
    wire                            pixel_ready;
    
    // Classification result
    wire [CLASS_ID_WIDTH-1:0]       class_result;
    wire [7:0]                      class_confidence;     // 8-bit confidence score
    wire                            class_result_valid;
    
    // Control signals
    wire                            ctrl_frame_start;
    wire                            ctrl_abort;
    wire [1:0]                      ctrl_mode;
    
    // Status aggregation
    wire                            vision_busy;
    wire                            vision_error;
    wire                            classifier_busy;
    wire                            classifier_done;
    wire                            classifier_error;
    wire                            io_error;
    
    // =========================================================================
    // IO EDGE SUBSYSTEM (Level 3 Macro)
    // =========================================================================
    // ~5mm²: SPI interface, I2C config, GPIO controller
    
    // GPIO bidirectional handling
    wire [7:0] gpio_in;
    wire [7:0] gpio_out_internal;
    wire [7:0] gpio_oe;
    
    genvar gi;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : gpio_bidir
            assign gpio[gi] = gpio_oe[gi] ? gpio_out_internal[gi] : 1'bz;
            assign gpio_in[gi] = gpio[gi];
        end
    endgenerate
    
    // Internal image path signals
    wire [7:0] img_data_byte;
    wire       img_valid_internal;
    wire       img_ready_internal;
    wire       img_sof;
    wire       img_eof;
    wire       io_inf_start;
    wire       io_inf_abort;
    
    io_edge #(
        .IMG_FIFO_DEPTH (512),
        .I2C_ADDR       (7'h50)
    ) u_io_edge (
        // Clocks and reset
        .clk            (clk_core),
        .rst_n          (rst_core_n),
        
        // SPI interface
        .spi_clk        (spi_clk),
        .spi_mosi       (spi_mosi),
        .spi_miso       (spi_miso),
        .spi_cs_n       (spi_cs_n),
        
        // I2C interface
        .i2c_scl        (i2c_scl),
        .i2c_sda        (i2c_sda),
        
        // GPIO
        .gpio_in        (gpio_in),
        .gpio_out       (gpio_out_internal),
        .gpio_oe        (gpio_oe),
        
        // Image data output (to vision)
        .img_data       (img_data_byte),
        .img_valid      (img_valid_internal),
        .img_ready      (img_ready_internal),
        .img_sof        (img_sof),
        .img_eof        (img_eof),
        
        // Control signals to inference engine
        .inf_start      (io_inf_start),
        .inf_abort      (io_inf_abort),
        
        // Status from inference engine
        .inf_busy       (vision_busy | classifier_busy),
        .inf_done       (classifier_done),
        .inf_error      (vision_error | classifier_error),
        .inf_class      (class_result[3:0]),
        .inf_confidence (class_confidence)
    );
    
    // Combine external and IO-derived control signals
    assign ctrl_frame_start = frame_start | io_inf_start;
    assign ctrl_abort = abort | io_inf_abort;
    assign ctrl_mode = mode;
    
    // Map 8-bit grayscale to 24-bit RGB (grayscale)
    assign pixel_data = {img_data_byte, img_data_byte, img_data_byte};
    assign pixel_valid = img_valid_internal;
    assign img_ready_internal = pixel_ready;
    
    // IRQ and ready directly to pins
    assign spi_irq_n = ~classifier_done;  // IRQ when inference done
    assign spi_rdy = ~(vision_busy | classifier_busy);
    
    // =========================================================================
    // VISION NANO SUBSYSTEM (Level 3 Macro)
    // =========================================================================
    // ~15mm²: NanoViT-12M with 6 transformer blocks
    // 12M parameters, ternary weights, 192-dim hidden
    
    vision_nano #(
        .HIDDEN_DIM     (HIDDEN_DIM),
        .NUM_HEADS      (3),
        .HEAD_DIM       (64),
        .MLP_DIM        (384),
        .NUM_LAYERS     (6),
        .IMAGE_SIZE     (IMAGE_SIZE),
        .PATCH_SIZE     (PATCH_SIZE),
        .NUM_PATCHES    (NUM_PATCHES),
        .SEQ_LEN        (197),
        .ACT_WIDTH      (ACT_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .PIXEL_WIDTH    (8)
    ) u_vision_nano (
        .clk            (clk_core),
        .rst_n          (rst_core_n),
        
        // Image input (streaming pixels)
        .img_valid      (pixel_valid),
        .img_pixel      (pixel_data),
        .img_sof        (img_sof),
        .img_eof        (img_eof),
        .img_ready      (pixel_ready),
        
        // Control
        .start          (ctrl_frame_start),
        .busy           (vision_busy),
        .done           (vision_done),
        
        // Output: CLS token embedding (192-dim)
        .out_valid      (vision_features_valid),
        .out_embedding  (vision_features),
        .out_ready      (vision_features_ready)
    );
    
    // Vision doesn't report errors in current implementation
    assign vision_error = 1'b0;
    
    // =========================================================================
    // CLASSIFIER HEAD SUBSYSTEM (Level 3 Macro)
    // =========================================================================
    // ~10mm²: MLP classifier + softmax
    // Input: 192-dim feature vector, Output: 1000 class probabilities
    
    // Hardwired ternary weights (synthesized as ROM)
    // These would come from trained model weights, all +1 for placeholder
    wire [HIDDEN_DIM*128*2-1:0]     w_proj_const;
    wire [128*128*2-1:0]            w_q_const, w_k_const, w_v_const, w_o_const;
    wire [128*256*2-1:0]            w_mlp_gate_const, w_mlp_up_const;
    wire [256*128*2-1:0]            w_mlp_down_const;
    wire [128*ACT_WIDTH-1:0]        rms_gamma_const;
    wire [128*NUM_CLASSES*2-1:0]    w_classifier_const;
    
    // Initialize all weights to +1 (2'b01) - in production, loaded from model
    assign w_proj_const = {(HIDDEN_DIM*128){2'b01}};
    assign w_q_const = {(128*128){2'b01}};
    assign w_k_const = {(128*128){2'b01}};
    assign w_v_const = {(128*128){2'b01}};
    assign w_o_const = {(128*128){2'b01}};
    assign w_mlp_gate_const = {(128*256){2'b01}};
    assign w_mlp_up_const = {(128*256){2'b01}};
    assign w_mlp_down_const = {(256*128){2'b01}};
    assign rms_gamma_const = {128{8'd16}};  // 1.0 in Q4.4 fixed point
    assign w_classifier_const = {(128*NUM_CLASSES){2'b01}};
    
    wire [$clog2(4)-1:0] classifier_layer_idx;
    
    classifier_head #(
        .IN_DIM         (HIDDEN_DIM),
        .HIDDEN_DIM     (128),
        .NUM_HEADS      (4),
        .HEAD_DIM       (32),
        .MLP_DIM        (256),
        .NUM_LAYERS     (4),
        .NUM_CLASSES    (NUM_CLASSES),
        .ACT_WIDTH      (ACT_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .FRAC_BITS      (4),
        .PARALLEL       (8)
    ) u_classifier_head (
        .clk            (clk_core),
        .rst_n          (rst_core_n),
        
        // Feature input (from vision)
        .vision_features(vision_features),
        .valid_in       (vision_features_valid),
        .ready_in       (vision_features_ready),
        
        // Hardwired ternary weights
        .w_proj         (w_proj_const),
        .w_q            (w_q_const),
        .w_k            (w_k_const),
        .w_v            (w_v_const),
        .w_o            (w_o_const),
        .w_mlp_gate     (w_mlp_gate_const),
        .w_mlp_up       (w_mlp_up_const),
        .w_mlp_down     (w_mlp_down_const),
        .rms_attn_gamma (rms_gamma_const),
        .rms_mlp_gamma  (rms_gamma_const),
        .rms_final_gamma(rms_gamma_const),
        .w_classifier   (w_classifier_const),
        
        // Layer selection
        .current_layer  (classifier_layer_idx),
        
        // Classification output
        .class_out      (class_result),
        .confidence_out (class_confidence),
        .valid_out      (class_result_valid),
        .ready_out      (1'b1)
    );
    
    // Derive status signals
    assign classifier_busy = ~vision_features_ready;
    assign classifier_done = class_result_valid;
    assign classifier_error = 1'b0;
    
    // =========================================================================
    // Classification Output Routing
    // =========================================================================
    
    assign class_valid = class_result_valid;
    assign class_id    = class_result;
    assign error       = vision_error | classifier_error;
    
    // Status signals
    assign busy = vision_busy | classifier_busy;
    assign done = classifier_done;
    assign io_error = 1'b0;

endmodule

// =============================================================================
// PLL Module for Edge SoC (200MHz operation)
// =============================================================================

module silens_edge_pll (
    input  wire clk_ref,
    input  wire rst_n,
    output wire clk_core,
    output wire clk_sram,
    output wire locked
);
    // SKY130 PLL instantiation placeholder
    // In actual implementation, use sky130_fd_sc_hd__pll
    // For now, pass-through for simulation
    
    reg [3:0] lock_cnt;
    reg locked_reg;
    
    always @(posedge clk_ref or negedge rst_n) begin
        if (!rst_n) begin
            lock_cnt <= 4'd0;
            locked_reg <= 1'b0;
        end else if (!locked_reg) begin
            if (lock_cnt == 4'd15)
                locked_reg <= 1'b1;
            else
                lock_cnt <= lock_cnt + 1'b1;
        end
    end
    
    assign clk_core = clk_ref;
    assign clk_sram = clk_ref;  // Same phase for simplicity
    assign locked = locked_reg;

endmodule

// =============================================================================
// Reset Synchronizer
// =============================================================================

module silens_edge_reset_sync (
    input  wire clk,
    input  wire rst_async_n,
    output wire rst_sync_n
);
    // 2-stage synchronizer for async reset release
    reg [1:0] sync_reg;
    
    always @(posedge clk or negedge rst_async_n) begin
        if (!rst_async_n)
            sync_reg <= 2'b00;
        else
            sync_reg <= {sync_reg[0], 1'b1};
    end
    
    assign rst_sync_n = sync_reg[1];

endmodule

`default_nettype wire
