// =============================================================================
// SiLens SoC Top - Level 4 Integration for 800mm² SKY130
// =============================================================================
// 
// This is the TOP-LEVEL module that integrates Level 3 hardened macros:
//   - vision_subsystem (~250mm²)
//   - llm_subsystem (~400mm²)
//   - memory_subsystem (~50mm²)
//   - io_subsystem (~30mm²)
//
// Target: SkyWater SKY130 130nm, 26mm × 30.77mm die (800mm²)
// Clock: 100MHz (10ns period)
// Power: ~25W TDP
//
// This wrapper handles:
//   - Clock generation and distribution
//   - Reset synchronization
//   - Inter-subsystem routing
//   - Top-level IO pad connections
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_soc_top #(
    // =========================================================================
    // Architecture Parameters (fixed for SmolVLM-256M)
    // =========================================================================
    parameter ACT_WIDTH         = 8,        // Activation bit width
    parameter ACC_WIDTH         = 32,       // Accumulator width
    parameter VOCAB_SIZE        = 49152,    // Vocabulary size
    parameter MAX_SEQ_LEN       = 2048,     // Maximum sequence length
    parameter NUM_PATCHES       = 576,      // Vision patches (24×24)
    
    // =========================================================================
    // Memory Parameters
    // =========================================================================
    parameter DDR_DATA_WIDTH    = 32,
    parameter DDR_ADDR_WIDTH    = 28,       // 256MB addressable
    parameter AXI_DATA_WIDTH    = 512,      // Internal bus width
    parameter AXI_ADDR_WIDTH    = 32,
    
    // =========================================================================
    // Host Interface Parameters  
    // =========================================================================
    parameter HOST_DATA_WIDTH   = 32,
    parameter HOST_ADDR_WIDTH   = 16
)(
    // =========================================================================
    // Clock and Reset
    // =========================================================================
    input  wire                         clk_ref,        // Reference clock (100MHz)
    input  wire                         rst_n,          // Active-low async reset
    
    // =========================================================================
    // DDR3 Interface (to external memory)
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
    input  wire                         host_clk,
    input  wire [HOST_DATA_WIDTH-1:0]   host_data_in,
    output wire [HOST_DATA_WIDTH-1:0]   host_data_out,
    output wire                         host_data_oe,
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
    // Internal Clock and Reset Signals
    // =========================================================================
    
    wire clk_core;              // Core clock (100MHz)
    wire clk_ddr;               // DDR clock (533MHz for DDR3-1066)
    wire clk_ddr_90;            // DDR clock 90° shifted
    wire pll_locked;
    
    wire rst_core_n;            // Sync reset for core domain
    wire rst_ddr_n;             // Sync reset for DDR domain
    wire rst_host_n;            // Sync reset for host domain
    
    // =========================================================================
    // Clock Generation PLL
    // =========================================================================
    
    silens_pll u_pll (
        .clk_ref    (clk_ref),
        .rst_n      (rst_n),
        .clk_core   (clk_core),
        .clk_ddr    (clk_ddr),
        .clk_ddr_90 (clk_ddr_90),
        .locked     (pll_locked)
    );
    
    // =========================================================================
    // Reset Synchronizers
    // =========================================================================
    
    silens_reset_sync u_rst_sync_core (
        .clk        (clk_core),
        .rst_async_n(rst_n & pll_locked),
        .rst_sync_n (rst_core_n)
    );
    
    silens_reset_sync u_rst_sync_ddr (
        .clk        (clk_ddr),
        .rst_async_n(rst_n & pll_locked),
        .rst_sync_n (rst_ddr_n)
    );
    
    silens_reset_sync u_rst_sync_host (
        .clk        (host_clk),
        .rst_async_n(rst_n & pll_locked),
        .rst_sync_n (rst_host_n)
    );
    
    // =========================================================================
    // Inter-Subsystem AXI Buses
    // =========================================================================
    
    // Vision → Memory (read activations buffer)
    wire [AXI_ADDR_WIDTH-1:0]   vision_mem_araddr;
    wire                        vision_mem_arvalid;
    wire                        vision_mem_arready;
    wire [AXI_DATA_WIDTH-1:0]   vision_mem_rdata;
    wire                        vision_mem_rvalid;
    wire                        vision_mem_rready;
    wire [1:0]                  vision_mem_rresp;
    
    // LLM → Memory (read/write KV cache)
    wire [AXI_ADDR_WIDTH-1:0]   llm_mem_araddr;
    wire                        llm_mem_arvalid;
    wire                        llm_mem_arready;
    wire [AXI_DATA_WIDTH-1:0]   llm_mem_rdata;
    wire                        llm_mem_rvalid;
    wire                        llm_mem_rready;
    wire [1:0]                  llm_mem_rresp;
    wire [AXI_ADDR_WIDTH-1:0]   llm_mem_awaddr;
    wire                        llm_mem_awvalid;
    wire                        llm_mem_awready;
    wire [AXI_DATA_WIDTH-1:0]   llm_mem_wdata;
    wire                        llm_mem_wvalid;
    wire                        llm_mem_wready;
    wire                        llm_mem_bvalid;
    wire                        llm_mem_bready;
    wire [1:0]                  llm_mem_bresp;
    
    // Host DMA → Memory
    wire [AXI_ADDR_WIDTH-1:0]   dma_mem_araddr;
    wire                        dma_mem_arvalid;
    wire                        dma_mem_arready;
    wire [AXI_DATA_WIDTH-1:0]   dma_mem_rdata;
    wire                        dma_mem_rvalid;
    wire                        dma_mem_rready;
    wire [1:0]                  dma_mem_rresp;
    wire [AXI_ADDR_WIDTH-1:0]   dma_mem_awaddr;
    wire                        dma_mem_awvalid;
    wire                        dma_mem_awready;
    wire [AXI_DATA_WIDTH-1:0]   dma_mem_wdata;
    wire                        dma_mem_wvalid;
    wire                        dma_mem_wready;
    wire                        dma_mem_bvalid;
    wire                        dma_mem_bready;
    wire [1:0]                  dma_mem_bresp;
    
    // =========================================================================
    // Vision → LLM Data Path (Projected Features)
    // =========================================================================
    
    wire [ACT_WIDTH*576-1:0]    vision_features;    // 576 patches × 8-bit
    wire                        vision_features_valid;
    wire                        vision_features_ready;
    wire                        vision_done;
    
    // =========================================================================
    // Control Signals from IO Subsystem
    // =========================================================================
    
    wire                        ctrl_frame_start;
    wire                        ctrl_seq_start;
    wire                        ctrl_gen_start;
    wire                        ctrl_abort;
    wire [15:0]                 ctrl_token_in;
    wire                        ctrl_token_in_valid;
    wire                        ctrl_token_in_ready;
    wire [15:0]                 ctrl_token_out;
    wire                        ctrl_token_out_valid;
    wire                        ctrl_token_out_ready;
    
    // =========================================================================
    // Status Signals to IO Subsystem  
    // =========================================================================
    
    wire                        status_vision_busy;
    wire                        status_llm_busy;
    wire                        status_inference_done;
    wire                        status_error;
    wire [3:0]                  status_state;
    wire                        status_ddr_init_done;
    wire                        status_ddr_cal_done;
    
    // =========================================================================
    // VISION SUBSYSTEM (Level 3 Macro)
    // =========================================================================
    // ~250mm²: 12× vision transformers + patch embedding + projector
    
    vision_subsystem #(
        .ACT_WIDTH      (ACT_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .NUM_PATCHES    (NUM_PATCHES),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH)
    ) u_vision (
        .clk            (clk_core),
        .rst_n          (rst_core_n),
        
        // Control
        .frame_start    (ctrl_frame_start),
        .abort          (ctrl_abort),
        
        // Pixel input (from DMA)
        .pixel_data     (dma_pixel_data),
        .pixel_valid    (dma_pixel_valid),
        .pixel_ready    (dma_pixel_ready),
        
        // Projected features output (to LLM)
        .features_out   (vision_features),
        .features_valid (vision_features_valid),
        .features_ready (vision_features_ready),
        
        // Memory interface (activation buffer)
        .m_axi_araddr   (vision_mem_araddr),
        .m_axi_arvalid  (vision_mem_arvalid),
        .m_axi_arready  (vision_mem_arready),
        .m_axi_rdata    (vision_mem_rdata),
        .m_axi_rvalid   (vision_mem_rvalid),
        .m_axi_rready   (vision_mem_rready),
        .m_axi_rresp    (vision_mem_rresp),
        
        // Status
        .busy           (status_vision_busy),
        .done           (vision_done),
        .error          ()
    );
    
    // DMA pixel interface (directly from memory or host)
    wire [23:0]                 dma_pixel_data;
    wire                        dma_pixel_valid;
    wire                        dma_pixel_ready;
    
    // =========================================================================
    // LLM SUBSYSTEM (Level 3 Macro)
    // =========================================================================
    // ~400mm²: 30× LLM transformers + embedding + LM head
    
    llm_subsystem #(
        .ACT_WIDTH      (ACT_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .VOCAB_SIZE     (VOCAB_SIZE),
        .MAX_SEQ_LEN    (MAX_SEQ_LEN),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH)
    ) u_llm (
        .clk            (clk_core),
        .rst_n          (rst_core_n),
        
        // Control
        .seq_start      (ctrl_seq_start),
        .gen_start      (ctrl_gen_start),
        .abort          (ctrl_abort),
        
        // Vision features input (from Vision Subsystem)
        .vision_features    (vision_features),
        .vision_features_valid (vision_features_valid),
        .vision_features_ready (vision_features_ready),
        
        // Text token input
        .token_in       (ctrl_token_in),
        .token_in_valid (ctrl_token_in_valid),
        .token_in_ready (ctrl_token_in_ready),
        
        // Generated token output
        .token_out      (ctrl_token_out),
        .token_out_valid(ctrl_token_out_valid),
        .token_out_ready(ctrl_token_out_ready),
        
        // Memory interface (KV cache read)
        .m_axi_araddr   (llm_mem_araddr),
        .m_axi_arvalid  (llm_mem_arvalid),
        .m_axi_arready  (llm_mem_arready),
        .m_axi_rdata    (llm_mem_rdata),
        .m_axi_rvalid   (llm_mem_rvalid),
        .m_axi_rready   (llm_mem_rready),
        .m_axi_rresp    (llm_mem_rresp),
        
        // Memory interface (KV cache write)
        .m_axi_awaddr   (llm_mem_awaddr),
        .m_axi_awvalid  (llm_mem_awvalid),
        .m_axi_awready  (llm_mem_awready),
        .m_axi_wdata    (llm_mem_wdata),
        .m_axi_wvalid   (llm_mem_wvalid),
        .m_axi_wready   (llm_mem_wready),
        .m_axi_bvalid   (llm_mem_bvalid),
        .m_axi_bready   (llm_mem_bready),
        .m_axi_bresp    (llm_mem_bresp),
        
        // Status
        .busy           (status_llm_busy),
        .done           (status_inference_done),
        .error          (status_error),
        .state          (status_state)
    );
    
    // =========================================================================
    // MEMORY SUBSYSTEM (Level 3 Macro)
    // =========================================================================
    // ~50mm²: DDR3 PHY + controller + 4-port AXI arbiter
    
    memory_subsystem #(
        .DDR_DATA_WIDTH (DDR_DATA_WIDTH),
        .DDR_ADDR_WIDTH (DDR_ADDR_WIDTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH)
    ) u_memory (
        // Clocks
        .clk_core       (clk_core),
        .clk_ddr        (clk_ddr),
        .clk_ddr_90     (clk_ddr_90),
        .rst_n          (rst_ddr_n),
        
        // DDR3 PHY interface
        .ddr3_addr      (ddr3_addr),
        .ddr3_ba        (ddr3_ba),
        .ddr3_cas_n     (ddr3_cas_n),
        .ddr3_ck_p      (ddr3_ck_p),
        .ddr3_ck_n      (ddr3_ck_n),
        .ddr3_cke       (ddr3_cke),
        .ddr3_cs_n      (ddr3_cs_n),
        .ddr3_dm        (ddr3_dm),
        .ddr3_dq        (ddr3_dq),
        .ddr3_dqs_p     (ddr3_dqs_p),
        .ddr3_dqs_n     (ddr3_dqs_n),
        .ddr3_odt       (ddr3_odt),
        .ddr3_ras_n     (ddr3_ras_n),
        .ddr3_reset_n   (ddr3_reset_n),
        .ddr3_we_n      (ddr3_we_n),
        
        // AXI Port 0: LLM Read
        .s0_axi_araddr  (llm_mem_araddr),
        .s0_axi_arvalid (llm_mem_arvalid),
        .s0_axi_arready (llm_mem_arready),
        .s0_axi_rdata   (llm_mem_rdata),
        .s0_axi_rvalid  (llm_mem_rvalid),
        .s0_axi_rready  (llm_mem_rready),
        .s0_axi_rresp   (llm_mem_rresp),
        
        // AXI Port 1: LLM Write
        .s1_axi_awaddr  (llm_mem_awaddr),
        .s1_axi_awvalid (llm_mem_awvalid),
        .s1_axi_awready (llm_mem_awready),
        .s1_axi_wdata   (llm_mem_wdata),
        .s1_axi_wvalid  (llm_mem_wvalid),
        .s1_axi_wready  (llm_mem_wready),
        .s1_axi_bvalid  (llm_mem_bvalid),
        .s1_axi_bready  (llm_mem_bready),
        .s1_axi_bresp   (llm_mem_bresp),
        
        // AXI Port 2: Vision Read
        .s2_axi_araddr  (vision_mem_araddr),
        .s2_axi_arvalid (vision_mem_arvalid),
        .s2_axi_arready (vision_mem_arready),
        .s2_axi_rdata   (vision_mem_rdata),
        .s2_axi_rvalid  (vision_mem_rvalid),
        .s2_axi_rready  (vision_mem_rready),
        .s2_axi_rresp   (vision_mem_rresp),
        
        // AXI Port 3: Host DMA
        .s3_axi_araddr  (dma_mem_araddr),
        .s3_axi_arvalid (dma_mem_arvalid),
        .s3_axi_arready (dma_mem_arready),
        .s3_axi_rdata   (dma_mem_rdata),
        .s3_axi_rvalid  (dma_mem_rvalid),
        .s3_axi_rready  (dma_mem_rready),
        .s3_axi_rresp   (dma_mem_rresp),
        .s3_axi_awaddr  (dma_mem_awaddr),
        .s3_axi_awvalid (dma_mem_awvalid),
        .s3_axi_awready (dma_mem_awready),
        .s3_axi_wdata   (dma_mem_wdata),
        .s3_axi_wvalid  (dma_mem_wvalid),
        .s3_axi_wready  (dma_mem_wready),
        .s3_axi_bvalid  (dma_mem_bvalid),
        .s3_axi_bready  (dma_mem_bready),
        .s3_axi_bresp   (dma_mem_bresp),
        
        // Status
        .init_done      (status_ddr_init_done),
        .cal_complete   (status_ddr_cal_done)
    );
    
    // =========================================================================
    // IO SUBSYSTEM (Level 3 Macro)
    // =========================================================================
    // ~30mm²: Host interface + SPI + GPIO + Interrupt controller
    
    io_subsystem #(
        .HOST_DATA_WIDTH(HOST_DATA_WIDTH),
        .HOST_ADDR_WIDTH(HOST_ADDR_WIDTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH)
    ) u_io (
        // Core clock domain
        .clk_core       (clk_core),
        .rst_core_n     (rst_core_n),
        
        // Host clock domain
        .host_clk       (host_clk),
        .rst_host_n     (rst_host_n),
        
        // Parallel host interface
        .host_data_in   (host_data_in),
        .host_data_out  (host_data_out),
        .host_data_oe   (host_data_oe),
        .host_addr      (host_addr),
        .host_rd_n      (host_rd_n),
        .host_wr_n      (host_wr_n),
        .host_cs_n      (host_cs_n),
        .host_ready     (host_ready),
        .host_irq       (host_irq),
        
        // SPI interface
        .spi_clk        (spi_clk),
        .spi_mosi       (spi_mosi),
        .spi_miso       (spi_miso),
        .spi_cs_n       (spi_cs_n),
        
        // GPIO
        .gpio_in        (gpio_in),
        .gpio_out       (gpio_out),
        .gpio_oe        (gpio_oe),
        
        // Control outputs (to Vision/LLM)
        .frame_start    (ctrl_frame_start),
        .seq_start      (ctrl_seq_start),
        .gen_start      (ctrl_gen_start),
        .abort          (ctrl_abort),
        
        // Token streaming
        .token_in       (ctrl_token_in),
        .token_in_valid (ctrl_token_in_valid),
        .token_in_ready (ctrl_token_in_ready),
        .token_out      (ctrl_token_out),
        .token_out_valid(ctrl_token_out_valid),
        .token_out_ready(ctrl_token_out_ready),
        
        // DMA interface (for pixel data)
        .dma_pixel_data (dma_pixel_data),
        .dma_pixel_valid(dma_pixel_valid),
        .dma_pixel_ready(dma_pixel_ready),
        
        // DMA memory port
        .m_axi_araddr   (dma_mem_araddr),
        .m_axi_arvalid  (dma_mem_arvalid),
        .m_axi_arready  (dma_mem_arready),
        .m_axi_rdata    (dma_mem_rdata),
        .m_axi_rvalid   (dma_mem_rvalid),
        .m_axi_rready   (dma_mem_rready),
        .m_axi_rresp    (dma_mem_rresp),
        .m_axi_awaddr   (dma_mem_awaddr),
        .m_axi_awvalid  (dma_mem_awvalid),
        .m_axi_awready  (dma_mem_awready),
        .m_axi_wdata    (dma_mem_wdata),
        .m_axi_wvalid   (dma_mem_wvalid),
        .m_axi_wready   (dma_mem_wready),
        .m_axi_bvalid   (dma_mem_bvalid),
        .m_axi_bready   (dma_mem_bready),
        .m_axi_bresp    (dma_mem_bresp),
        
        // Status inputs
        .vision_busy    (status_vision_busy),
        .llm_busy       (status_llm_busy),
        .inference_done (status_inference_done),
        .error_flag     (status_error),
        .state          (status_state),
        .ddr_init_done  (status_ddr_init_done),
        .ddr_cal_done   (status_ddr_cal_done)
    );
    
    // =========================================================================
    // JTAG Boundary Scan (simplified bypass)
    // =========================================================================
    
    reg jtag_bypass_reg;
    always @(posedge jtag_tck or negedge jtag_trst_n) begin
        if (!jtag_trst_n)
            jtag_bypass_reg <= 1'b0;
        else
            jtag_bypass_reg <= jtag_tdi;
    end
    assign jtag_tdo = jtag_bypass_reg;
    
    // =========================================================================
    // Status LEDs
    // =========================================================================
    
    assign status_led = status_state;
    assign error_led = status_error;
    
    // Heartbeat (toggle every ~0.5s at 100MHz)
    reg [25:0] heartbeat_cnt;
    always @(posedge clk_core or negedge rst_core_n) begin
        if (!rst_core_n)
            heartbeat_cnt <= 26'd0;
        else
            heartbeat_cnt <= heartbeat_cnt + 1'b1;
    end
    assign heartbeat = heartbeat_cnt[25];

endmodule

`default_nettype wire
