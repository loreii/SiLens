// =============================================================================
// SiLens SoC - 800mm² Full Custom SKY130 Vision-Language Accelerator
// =============================================================================
// 
// Target: SkyWater SKY130 130nm CMOS
// Die Size: ~800mm² (fits 26mm × 32mm reticle field)
// Model: SmolVLM-256M (246M parameters, ternary quantized)
//
// Architecture:
//   - Vision Encoder: SigLIP-B/16 (93M params) - ~250mm²
//   - Projector: 768→576 MLP (18M params) - ~50mm²
//   - Language Model: SmolLM2-135M (135M params) - ~400mm²
//   - DDR3 PHY: 32-bit interface - ~30mm²
//   - Host Interface: Parallel bus for FPGA bridge - ~20mm²
//   - Clock/Power/IO: ~50mm²
//
// Memory Architecture:
//   - Weights: 61.5MB hardwired as metal routing
//   - KV Cache: External DDR3 (256MB-1GB)
//   - Activations: On-chip SRAM (~16MB)
//
// Interfaces:
//   - DDR3-1066 x32 (4.3 GB/s bandwidth)
//   - Parallel host bus (32-bit @ 100MHz = 400 MB/s)
//   - SPI for configuration/debug
//   - JTAG for test access
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_soc #(
    // =========================================================================
    // Clock Configuration
    // =========================================================================
    parameter CORE_CLK_MHZ      = 100,      // Core clock frequency
    parameter DDR_CLK_MHZ       = 533,      // DDR3-1066 (2x 533MHz)
    parameter HOST_CLK_MHZ      = 100,      // Host interface clock
    
    // =========================================================================
    // Vision Encoder Parameters (SigLIP-B/16)
    // =========================================================================
    parameter VISION_DIM        = 768,      // Hidden dimension
    parameter VISION_LAYERS     = 12,       // Transformer blocks
    parameter VISION_HEADS      = 12,       // Attention heads
    parameter VISION_MLP_DIM    = 3072,     // MLP intermediate dim (4x)
    parameter IMG_SIZE          = 384,      // Input image size
    parameter PATCH_SIZE        = 16,       // Patch size
    parameter NUM_PATCHES       = (IMG_SIZE/PATCH_SIZE) * (IMG_SIZE/PATCH_SIZE), // 576
    parameter IN_CHANNELS       = 3,        // RGB
    
    // =========================================================================
    // Language Model Parameters (SmolLM2-135M)
    // =========================================================================
    parameter LLM_DIM           = 576,      // Hidden dimension
    parameter LLM_LAYERS        = 30,       // Transformer blocks
    parameter LLM_HEADS         = 9,        // Attention heads (GQA)
    parameter LLM_KV_HEADS      = 9,        // KV heads (same as Q for now)
    parameter LLM_MLP_DIM       = 1536,     // MLP intermediate dim
    parameter VOCAB_SIZE        = 49152,    // Vocabulary size
    parameter MAX_SEQ_LEN       = 2048,     // Maximum sequence length
    
    // =========================================================================
    // Quantization Parameters
    // =========================================================================
    parameter ACT_WIDTH         = 8,        // Activation bit width
    parameter ACC_WIDTH         = 32,       // Accumulator bit width
    parameter FRAC_BITS         = 4,        // Fractional bits for fixed point
    parameter PARALLEL          = 64,       // SIMD parallelism factor
    
    // =========================================================================
    // Memory Parameters
    // =========================================================================
    parameter DDR_DATA_WIDTH    = 32,       // DDR3 data bus width
    parameter DDR_ADDR_WIDTH    = 28,       // DDR3 address width (256MB)
    parameter SRAM_DEPTH        = 262144,   // On-chip SRAM words (16MB)
    parameter SRAM_WIDTH        = 512,      // SRAM data width
    
    // =========================================================================
    // Host Interface Parameters
    // =========================================================================
    parameter HOST_DATA_WIDTH   = 32,       // Parallel bus width
    parameter HOST_ADDR_WIDTH   = 16,       // Register address space
    
    // =========================================================================
    // Computed Parameters
    // =========================================================================
    parameter VISION_PARAMS     = 93_000_000,
    parameter PROJ_PARAMS       = 18_000_000,
    parameter LLM_PARAMS        = 135_000_000,
    parameter TOTAL_PARAMS      = VISION_PARAMS + PROJ_PARAMS + LLM_PARAMS
)(
    // =========================================================================
    // Clock and Reset
    // =========================================================================
    input  wire                         clk_ref,        // Reference clock (100MHz)
    input  wire                         rst_n,          // Active-low reset
    
    // =========================================================================
    // DDR3 Interface (32-bit)
    // =========================================================================
    output wire [13:0]                  ddr3_addr,
    output wire [2:0]                   ddr3_ba,
    output wire                         ddr3_cas_n,
    output wire                         ddr3_ck_p,
    output wire                         ddr3_ck_n,
    output wire                         ddr3_cke,
    output wire                         ddr3_cs_n,
    output wire [3:0]                   ddr3_dm,
    inout  wire [31:0]                  ddr3_dq,
    inout  wire [3:0]                   ddr3_dqs_p,
    inout  wire [3:0]                   ddr3_dqs_n,
    output wire                         ddr3_odt,
    output wire                         ddr3_ras_n,
    output wire                         ddr3_reset_n,
    output wire                         ddr3_we_n,
    
    // =========================================================================
    // Parallel Host Interface (to FPGA bridge)
    // =========================================================================
    input  wire                         host_clk,       // Host clock
    input  wire [HOST_DATA_WIDTH-1:0]   host_data_in,
    output wire [HOST_DATA_WIDTH-1:0]   host_data_out,
    output wire                         host_data_oe,   // Tristate control
    input  wire [HOST_ADDR_WIDTH-1:0]   host_addr,
    input  wire                         host_rd_n,
    input  wire                         host_wr_n,
    input  wire                         host_cs_n,
    output wire                         host_ready,
    output wire                         host_irq,
    
    // =========================================================================
    // SPI Configuration Interface
    // =========================================================================
    input  wire                         spi_clk,
    input  wire                         spi_mosi,
    output wire                         spi_miso,
    input  wire                         spi_cs_n,
    
    // =========================================================================
    // JTAG Test Interface
    // =========================================================================
    input  wire                         jtag_tck,
    input  wire                         jtag_tms,
    input  wire                         jtag_tdi,
    output wire                         jtag_tdo,
    input  wire                         jtag_trst_n,
    
    // =========================================================================
    // GPIO and Status
    // =========================================================================
    input  wire [7:0]                   gpio_in,
    output wire [7:0]                   gpio_out,
    output wire [7:0]                   gpio_oe,
    
    output wire [3:0]                   status_led,
    output wire                         heartbeat,
    output wire                         error_led
);

    // =========================================================================
    // Internal Clocks
    // =========================================================================
    
    wire clk_core;          // Core logic clock (100MHz)
    wire clk_ddr;           // DDR interface clock (533MHz)
    wire clk_ddr_90;        // DDR clock 90° phase shifted
    wire pll_locked;
    
    // =========================================================================
    // Internal Resets (synchronized)
    // =========================================================================
    
    wire rst_core_n;
    wire rst_ddr_n;
    wire rst_host_n;
    
    // =========================================================================
    // Clock Generation (PLL)
    // =========================================================================
    
    silens_clock_gen #(
        .REF_CLK_MHZ(100),
        .CORE_CLK_MHZ(CORE_CLK_MHZ),
        .DDR_CLK_MHZ(DDR_CLK_MHZ)
    ) u_clock_gen (
        .clk_ref(clk_ref),
        .rst_n(rst_n),
        .clk_core(clk_core),
        .clk_ddr(clk_ddr),
        .clk_ddr_90(clk_ddr_90),
        .locked(pll_locked)
    );
    
    // =========================================================================
    // Reset Synchronizers
    // =========================================================================
    
    silens_reset_sync u_rst_core (
        .clk(clk_core),
        .rst_async_n(rst_n & pll_locked),
        .rst_sync_n(rst_core_n)
    );
    
    silens_reset_sync u_rst_ddr (
        .clk(clk_ddr),
        .rst_async_n(rst_n & pll_locked),
        .rst_sync_n(rst_ddr_n)
    );
    
    silens_reset_sync u_rst_host (
        .clk(host_clk),
        .rst_async_n(rst_n & pll_locked),
        .rst_sync_n(rst_host_n)
    );
    
    // =========================================================================
    // Internal Bus (AXI-Lite style for simplicity)
    // =========================================================================
    
    // Memory interface (from neural network to DDR)
    wire [DDR_ADDR_WIDTH-1:0]   mem_addr;
    wire [SRAM_WIDTH-1:0]       mem_wdata;
    wire [SRAM_WIDTH-1:0]       mem_rdata;
    wire                        mem_rd;
    wire                        mem_wr;
    wire                        mem_ready;
    
    // Host register interface
    wire [HOST_ADDR_WIDTH-1:0]  reg_addr;
    wire [31:0]                 reg_wdata;
    wire [31:0]                 reg_rdata;
    wire                        reg_rd;
    wire                        reg_wr;
    wire                        reg_ready;
    
    // =========================================================================
    // DDR3 Memory Controller
    // =========================================================================
    
    wire                        ddr_init_done;
    wire                        ddr_cal_complete;
    
    silens_ddr3_controller #(
        .DATA_WIDTH(DDR_DATA_WIDTH),
        .ADDR_WIDTH(DDR_ADDR_WIDTH)
    ) u_ddr3_ctrl (
        // System
        .clk_core(clk_core),
        .clk_ddr(clk_ddr),
        .clk_ddr_90(clk_ddr_90),
        .rst_n(rst_ddr_n),
        
        // User interface
        .user_addr(mem_addr),
        .user_wdata(mem_wdata[DDR_DATA_WIDTH-1:0]),  // Width adaptation
        .user_rdata(mem_rdata[DDR_DATA_WIDTH-1:0]),
        .user_rd(mem_rd),
        .user_wr(mem_wr),
        .user_ready(mem_ready),
        
        // DDR3 PHY interface
        .ddr3_addr(ddr3_addr),
        .ddr3_ba(ddr3_ba),
        .ddr3_cas_n(ddr3_cas_n),
        .ddr3_ck_p(ddr3_ck_p),
        .ddr3_ck_n(ddr3_ck_n),
        .ddr3_cke(ddr3_cke),
        .ddr3_cs_n(ddr3_cs_n),
        .ddr3_dm(ddr3_dm),
        .ddr3_dq(ddr3_dq),
        .ddr3_dqs_p(ddr3_dqs_p),
        .ddr3_dqs_n(ddr3_dqs_n),
        .ddr3_odt(ddr3_odt),
        .ddr3_ras_n(ddr3_ras_n),
        .ddr3_reset_n(ddr3_reset_n),
        .ddr3_we_n(ddr3_we_n),
        
        // Status
        .init_done(ddr_init_done),
        .cal_complete(ddr_cal_complete)
    );
    
    // Zero-extend mem_rdata upper bits
    assign mem_rdata[SRAM_WIDTH-1:DDR_DATA_WIDTH] = {(SRAM_WIDTH-DDR_DATA_WIDTH){1'b0}};
    
    // =========================================================================
    // Host Parallel Interface Controller
    // =========================================================================
    
    wire                        host_frame_start;
    wire                        host_seq_start;
    wire                        host_gen_start;
    wire                        host_abort;
    wire [15:0]                 host_token_in;
    wire                        host_token_in_valid;
    wire                        host_token_in_ready;
    wire [15:0]                 host_token_out;
    wire                        host_token_out_valid;
    wire                        host_token_out_ready;
    wire [31:0]                 host_status;
    
    silens_host_interface #(
        .DATA_WIDTH(HOST_DATA_WIDTH),
        .ADDR_WIDTH(HOST_ADDR_WIDTH)
    ) u_host_if (
        // Host side (FPGA bridge)
        .host_clk(host_clk),
        .host_rst_n(rst_host_n),
        .host_data_in(host_data_in),
        .host_data_out(host_data_out),
        .host_data_oe(host_data_oe),
        .host_addr(host_addr),
        .host_rd_n(host_rd_n),
        .host_wr_n(host_wr_n),
        .host_cs_n(host_cs_n),
        .host_ready(host_ready),
        .host_irq(host_irq),
        
        // Core side
        .core_clk(clk_core),
        .core_rst_n(rst_core_n),
        
        // Control outputs (synchronized to core_clk)
        .frame_start(host_frame_start),
        .seq_start(host_seq_start),
        .gen_start(host_gen_start),
        .abort(host_abort),
        
        // Token streaming
        .token_in(host_token_in),
        .token_in_valid(host_token_in_valid),
        .token_in_ready(host_token_in_ready),
        .token_out(host_token_out),
        .token_out_valid(host_token_out_valid),
        .token_out_ready(host_token_out_ready),
        
        // Status
        .status(host_status),
        .ddr_init_done(ddr_init_done)
    );
    
    // =========================================================================
    // Vision-Language Processing Core
    // =========================================================================
    
    wire                        core_vision_busy;
    wire                        core_llm_busy;
    wire                        core_inference_done;
    wire                        core_error;
    wire [3:0]                  core_state;
    
    // Image pixel input (from DMA)
    wire [IN_CHANNELS*ACT_WIDTH-1:0] pixel_data;
    wire                        pixel_valid;
    wire                        pixel_ready;
    
    // Generated token output
    wire [$clog2(VOCAB_SIZE)-1:0] token_out_core;
    wire                        token_out_valid_core;
    wire                        token_out_ready_core;
    
    silens_vlm_core #(
        // Vision parameters
        .VISION_DIM(VISION_DIM),
        .VISION_LAYERS(VISION_LAYERS),
        .VISION_HEADS(VISION_HEADS),
        .VISION_MLP_DIM(VISION_MLP_DIM),
        .IMG_SIZE(IMG_SIZE),
        .PATCH_SIZE(PATCH_SIZE),
        .NUM_PATCHES(NUM_PATCHES),
        .IN_CHANNELS(IN_CHANNELS),
        
        // LLM parameters
        .LLM_DIM(LLM_DIM),
        .LLM_LAYERS(LLM_LAYERS),
        .LLM_HEADS(LLM_HEADS),
        .LLM_KV_HEADS(LLM_KV_HEADS),
        .LLM_MLP_DIM(LLM_MLP_DIM),
        .VOCAB_SIZE(VOCAB_SIZE),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        
        // Precision
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL),
        
        // Memory
        .MEM_ADDR_WIDTH(DDR_ADDR_WIDTH),
        .MEM_DATA_WIDTH(SRAM_WIDTH)
    ) u_vlm_core (
        .clk(clk_core),
        .rst_n(rst_core_n),
        
        // Control
        .frame_start(host_frame_start),
        .seq_start(host_seq_start),
        .gen_start(host_gen_start),
        .abort(host_abort),
        
        // Pixel input (for vision)
        .pixel_in(pixel_data),
        .pixel_valid(pixel_valid),
        .pixel_ready(pixel_ready),
        
        // Token input (for text prompts)
        .token_in(host_token_in),
        .token_in_valid(host_token_in_valid),
        .token_in_ready(host_token_in_ready),
        
        // Token output (generated)
        .token_out(token_out_core),
        .token_out_valid(token_out_valid_core),
        .token_out_ready(token_out_ready_core),
        
        // External memory interface (KV cache)
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_rdata(mem_rdata),
        .mem_rd(mem_rd),
        .mem_wr(mem_wr),
        .mem_ready(mem_ready),
        
        // Status
        .vision_busy(core_vision_busy),
        .llm_busy(core_llm_busy),
        .inference_done(core_inference_done),
        .error_flag(core_error),
        .state_out(core_state)
    );
    
    // Token output width adaptation
    assign host_token_out = token_out_core[15:0];
    assign host_token_out_valid = token_out_valid_core;
    assign token_out_ready_core = host_token_out_ready;
    
    // Status composition
    assign host_status = {
        16'b0,
        4'b0, core_state,
        2'b0, core_error, core_inference_done,
        core_llm_busy, core_vision_busy,
        ddr_cal_complete, ddr_init_done
    };
    
    // =========================================================================
    // SPI Configuration Interface
    // =========================================================================
    
    wire [7:0] spi_reg_addr;
    wire [7:0] spi_reg_wdata;
    wire [7:0] spi_reg_rdata;
    wire       spi_reg_wr;
    
    silens_spi_slave u_spi_slave (
        .clk(clk_core),
        .rst_n(rst_core_n),
        .spi_clk(spi_clk),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .spi_cs_n(spi_cs_n),
        .reg_addr(spi_reg_addr),
        .reg_wdata(spi_reg_wdata),
        .reg_rdata(spi_reg_rdata),
        .reg_wr(spi_reg_wr)
    );
    
    // SPI register read data (basic status)
    assign spi_reg_rdata = (spi_reg_addr == 8'h00) ? {4'b0, core_state} :
                           (spi_reg_addr == 8'h01) ? {6'b0, core_error, core_inference_done} :
                           (spi_reg_addr == 8'h02) ? {6'b0, ddr_cal_complete, ddr_init_done} :
                           8'hFF;
    
    // =========================================================================
    // JTAG Boundary Scan (placeholder)
    // =========================================================================
    
    // Simple JTAG bypass for now
    reg jtag_bypass_reg;
    always @(posedge jtag_tck or negedge jtag_trst_n) begin
        if (!jtag_trst_n)
            jtag_bypass_reg <= 1'b0;
        else
            jtag_bypass_reg <= jtag_tdi;
    end
    assign jtag_tdo = jtag_bypass_reg;
    
    // =========================================================================
    // GPIO Controller
    // =========================================================================
    
    reg [7:0] gpio_out_reg;
    reg [7:0] gpio_oe_reg;
    
    always @(posedge clk_core) begin
        if (!rst_core_n) begin
            gpio_out_reg <= 8'b0;
            gpio_oe_reg  <= 8'b0;
        end else if (spi_reg_wr && spi_reg_addr == 8'h10) begin
            gpio_out_reg <= spi_reg_wdata;
        end else if (spi_reg_wr && spi_reg_addr == 8'h11) begin
            gpio_oe_reg <= spi_reg_wdata;
        end
    end
    
    assign gpio_out = gpio_out_reg;
    assign gpio_oe  = gpio_oe_reg;
    
    // =========================================================================
    // Status LEDs and Heartbeat
    // =========================================================================
    
    // Status LEDs show FSM state
    assign status_led = core_state;
    
    // Error LED
    assign error_led = core_error;
    
    // Heartbeat (toggles every ~0.5s at 100MHz)
    reg [25:0] heartbeat_cnt;
    always @(posedge clk_core) begin
        if (!rst_core_n)
            heartbeat_cnt <= 0;
        else
            heartbeat_cnt <= heartbeat_cnt + 1;
    end
    assign heartbeat = heartbeat_cnt[25];
    
    // =========================================================================
    // DMA Controller for Pixel Input (simplified)
    // =========================================================================
    
    // In full implementation, this would stream pixels from host memory
    // For now, connect to host interface tokens (simulating pixel data)
    assign pixel_data = {(IN_CHANNELS*ACT_WIDTH){1'b0}};  // Placeholder
    assign pixel_valid = 1'b0;  // Placeholder
    
endmodule

`default_nettype wire
