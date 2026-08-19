// =============================================================================
// Memory Subsystem - Level 3 DDR3 Memory Controller
// =============================================================================
// DDR3-1066 Memory Controller for external KV cache and weight storage.
// Target: ~50mm² on SKY130 (7100µm × 7100µm)
//
// Architecture:
// - 4-port AXI4 Arbiter (LLM read, LLM write, Vision read, Host DMA)
// - DDR3 Controller (command scheduler, refresh, bank management)
// - DDR3 PHY (x32 data width, DQS strobe, ODT, read/write leveling)
//
// DDR3 Specifications:
// - Speed: DDR3-1066 (533 MHz clock, 1066 MT/s)
// - Width: x32 (4×x8 or 2×x16 chips)
// - Capacity: 512MB - 2GB range
// - Burst length: 8 (64 bytes per burst)
// - Bandwidth: 4.3 GB/s theoretical
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module memory_subsystem #(
    parameter NUM_AXI_PORTS = 4,
    parameter AXI_DATA_WIDTH = 64,
    parameter AXI_ADDR_WIDTH = 32,
    parameter AXI_ID_WIDTH = 4,
    parameter DDR_DATA_WIDTH = 32,
    parameter DDR_ADDR_WIDTH = 16,
    parameter DDR_BANK_WIDTH = 3,
    
    // DDR3-1066 Timing (cycles @ 533MHz)
    parameter tRCD  = 8,
    parameter tRP   = 8,
    parameter tRAS  = 20,
    parameter tRC   = 28,
    parameter tRFC  = 86,
    parameter tREFI = 4160,
    parameter CL    = 7,
    parameter CWL   = 6
)(
    // ==========================================================================
    // System Clocks and Reset
    // ==========================================================================
    input  wire                         clk_core,       // 100 MHz core clock
    input  wire                         clk_ddr,        // 533 MHz DDR clock
    input  wire                         clk_ddr_90,     // 533 MHz DDR clock 90° phase
    input  wire                         clk_ddr_180,    // 533 MHz DDR clock 180° phase
    input  wire                         clk_ref,        // Reference clock for PLL/DLL
    input  wire                         rst_n,
    
    // ==========================================================================
    // AXI4 Slave Port 0 - LLM Read (highest priority)
    // ==========================================================================
    input  wire [AXI_ID_WIDTH-1:0]      s0_axi_arid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s0_axi_araddr,
    input  wire [7:0]                   s0_axi_arlen,
    input  wire [2:0]                   s0_axi_arsize,
    input  wire [1:0]                   s0_axi_arburst,
    input  wire                         s0_axi_arvalid,
    output wire                         s0_axi_arready,
    output wire [AXI_ID_WIDTH-1:0]      s0_axi_rid,
    output wire [AXI_DATA_WIDTH-1:0]    s0_axi_rdata,
    output wire [1:0]                   s0_axi_rresp,
    output wire                         s0_axi_rlast,
    output wire                         s0_axi_rvalid,
    input  wire                         s0_axi_rready,
    
    // ==========================================================================
    // AXI4 Slave Port 1 - LLM Write (KV cache updates)
    // ==========================================================================
    input  wire [AXI_ID_WIDTH-1:0]      s1_axi_awid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s1_axi_awaddr,
    input  wire [7:0]                   s1_axi_awlen,
    input  wire [2:0]                   s1_axi_awsize,
    input  wire [1:0]                   s1_axi_awburst,
    input  wire                         s1_axi_awvalid,
    output wire                         s1_axi_awready,
    input  wire [AXI_DATA_WIDTH-1:0]    s1_axi_wdata,
    input  wire [AXI_DATA_WIDTH/8-1:0]  s1_axi_wstrb,
    input  wire                         s1_axi_wlast,
    input  wire                         s1_axi_wvalid,
    output wire                         s1_axi_wready,
    output wire [AXI_ID_WIDTH-1:0]      s1_axi_bid,
    output wire [1:0]                   s1_axi_bresp,
    output wire                         s1_axi_bvalid,
    input  wire                         s1_axi_bready,
    
    // ==========================================================================
    // AXI4 Slave Port 2 - Vision Read (frame processing)
    // ==========================================================================
    input  wire [AXI_ID_WIDTH-1:0]      s2_axi_arid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s2_axi_araddr,
    input  wire [7:0]                   s2_axi_arlen,
    input  wire [2:0]                   s2_axi_arsize,
    input  wire [1:0]                   s2_axi_arburst,
    input  wire                         s2_axi_arvalid,
    output wire                         s2_axi_arready,
    output wire [AXI_ID_WIDTH-1:0]      s2_axi_rid,
    output wire [AXI_DATA_WIDTH-1:0]    s2_axi_rdata,
    output wire [1:0]                   s2_axi_rresp,
    output wire                         s2_axi_rlast,
    output wire                         s2_axi_rvalid,
    input  wire                         s2_axi_rready,
    
    // ==========================================================================
    // AXI4 Slave Port 3 - Host DMA (lowest priority)
    // ==========================================================================
    input  wire [AXI_ID_WIDTH-1:0]      s3_axi_awid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s3_axi_awaddr,
    input  wire [7:0]                   s3_axi_awlen,
    input  wire [2:0]                   s3_axi_awsize,
    input  wire [1:0]                   s3_axi_awburst,
    input  wire                         s3_axi_awvalid,
    output wire                         s3_axi_awready,
    input  wire [AXI_DATA_WIDTH-1:0]    s3_axi_wdata,
    input  wire [AXI_DATA_WIDTH/8-1:0]  s3_axi_wstrb,
    input  wire                         s3_axi_wlast,
    input  wire                         s3_axi_wvalid,
    output wire                         s3_axi_wready,
    output wire [AXI_ID_WIDTH-1:0]      s3_axi_bid,
    output wire [1:0]                   s3_axi_bresp,
    output wire                         s3_axi_bvalid,
    input  wire                         s3_axi_bready,
    input  wire [AXI_ID_WIDTH-1:0]      s3_axi_arid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s3_axi_araddr,
    input  wire [7:0]                   s3_axi_arlen,
    input  wire [2:0]                   s3_axi_arsize,
    input  wire [1:0]                   s3_axi_arburst,
    input  wire                         s3_axi_arvalid,
    output wire                         s3_axi_arready,
    output wire [AXI_ID_WIDTH-1:0]      s3_axi_rid,
    output wire [AXI_DATA_WIDTH-1:0]    s3_axi_rdata,
    output wire [1:0]                   s3_axi_rresp,
    output wire                         s3_axi_rlast,
    output wire                         s3_axi_rvalid,
    input  wire                         s3_axi_rready,
    
    // ==========================================================================
    // DDR3 Physical Interface
    // ==========================================================================
    output wire                         ddr3_ck_p,
    output wire                         ddr3_ck_n,
    output wire                         ddr3_cs_n,
    output wire                         ddr3_ras_n,
    output wire                         ddr3_cas_n,
    output wire                         ddr3_we_n,
    output wire                         ddr3_cke,
    output wire                         ddr3_reset_n,
    output wire [DDR_BANK_WIDTH-1:0]    ddr3_ba,
    output wire [DDR_ADDR_WIDTH-1:0]    ddr3_addr,
    inout  wire [DDR_DATA_WIDTH-1:0]    ddr3_dq,
    inout  wire [DDR_DATA_WIDTH/8-1:0]  ddr3_dqs_p,
    inout  wire [DDR_DATA_WIDTH/8-1:0]  ddr3_dqs_n,
    output wire [DDR_DATA_WIDTH/8-1:0]  ddr3_dm,
    output wire                         ddr3_odt,
    
    // ==========================================================================
    // Status and Debug
    // ==========================================================================
    output wire                         init_done,
    output wire                         cal_complete,
    output wire [3:0]                   port_active,
    output wire [1:0]                   current_rd_port,
    output wire [1:0]                   current_wr_port,
    output wire [7:0]                   debug_state
);

    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // Arbiter to Controller AXI signals
    wire [AXI_ID_WIDTH-1:0]     arb_axi_awid;
    wire [AXI_ADDR_WIDTH-1:0]   arb_axi_awaddr;
    wire [7:0]                  arb_axi_awlen;
    wire [2:0]                  arb_axi_awsize;
    wire [1:0]                  arb_axi_awburst;
    wire                        arb_axi_awvalid;
    wire                        arb_axi_awready;
    
    wire [AXI_DATA_WIDTH-1:0]   arb_axi_wdata;
    wire [AXI_DATA_WIDTH/8-1:0] arb_axi_wstrb;
    wire                        arb_axi_wlast;
    wire                        arb_axi_wvalid;
    wire                        arb_axi_wready;
    
    wire [AXI_ID_WIDTH-1:0]     arb_axi_bid;
    wire [1:0]                  arb_axi_bresp;
    wire                        arb_axi_bvalid;
    wire                        arb_axi_bready;
    
    wire [AXI_ID_WIDTH-1:0]     arb_axi_arid;
    wire [AXI_ADDR_WIDTH-1:0]   arb_axi_araddr;
    wire [7:0]                  arb_axi_arlen;
    wire [2:0]                  arb_axi_arsize;
    wire [1:0]                  arb_axi_arburst;
    wire                        arb_axi_arvalid;
    wire                        arb_axi_arready;
    
    wire [AXI_ID_WIDTH-1:0]     arb_axi_rid;
    wire [AXI_DATA_WIDTH-1:0]   arb_axi_rdata;
    wire [1:0]                  arb_axi_rresp;
    wire                        arb_axi_rlast;
    wire                        arb_axi_rvalid;
    wire                        arb_axi_rready;
    
    // Controller to PHY signals
    wire [2:0]                  phy_cmd;
    wire                        phy_cmd_valid;
    wire                        phy_cmd_ready;
    wire [DDR_ADDR_WIDTH-1:0]   phy_addr;
    wire [DDR_BANK_WIDTH-1:0]   phy_bank;
    wire [DDR_DATA_WIDTH*2-1:0] phy_wdata;
    wire [DDR_DATA_WIDTH/4-1:0] phy_wdata_mask;
    wire                        phy_wdata_valid;
    wire                        phy_wdata_ready;
    wire [DDR_DATA_WIDTH*2-1:0] phy_rdata;
    wire                        phy_rdata_valid;
    wire                        phy_ready;

    // =========================================================================
    // Concatenate AXI slave ports for arbiter
    // =========================================================================
    
    // Write address channel - pack slave signals
    wire [NUM_AXI_PORTS*AXI_ID_WIDTH-1:0]    s_axi_awid_packed;
    wire [NUM_AXI_PORTS*AXI_ADDR_WIDTH-1:0]  s_axi_awaddr_packed;
    wire [NUM_AXI_PORTS*8-1:0]               s_axi_awlen_packed;
    wire [NUM_AXI_PORTS*3-1:0]               s_axi_awsize_packed;
    wire [NUM_AXI_PORTS*2-1:0]               s_axi_awburst_packed;
    wire [NUM_AXI_PORTS-1:0]                 s_axi_awvalid_packed;
    wire [NUM_AXI_PORTS-1:0]                 s_axi_awready_packed;
    
    // Port 0 has no write (read-only for LLM read path)
    assign s_axi_awid_packed[0*AXI_ID_WIDTH +: AXI_ID_WIDTH] = {AXI_ID_WIDTH{1'b0}};
    assign s_axi_awaddr_packed[0*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH] = {AXI_ADDR_WIDTH{1'b0}};
    assign s_axi_awlen_packed[0*8 +: 8] = 8'd0;
    assign s_axi_awsize_packed[0*3 +: 3] = 3'd0;
    assign s_axi_awburst_packed[0*2 +: 2] = 2'd0;
    assign s_axi_awvalid_packed[0] = 1'b0;
    
    // Port 1: LLM Write
    assign s_axi_awid_packed[1*AXI_ID_WIDTH +: AXI_ID_WIDTH] = s1_axi_awid;
    assign s_axi_awaddr_packed[1*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH] = s1_axi_awaddr;
    assign s_axi_awlen_packed[1*8 +: 8] = s1_axi_awlen;
    assign s_axi_awsize_packed[1*3 +: 3] = s1_axi_awsize;
    assign s_axi_awburst_packed[1*2 +: 2] = s1_axi_awburst;
    assign s_axi_awvalid_packed[1] = s1_axi_awvalid;
    assign s1_axi_awready = s_axi_awready_packed[1];
    
    // Port 2 has no write (read-only for Vision)
    assign s_axi_awid_packed[2*AXI_ID_WIDTH +: AXI_ID_WIDTH] = {AXI_ID_WIDTH{1'b0}};
    assign s_axi_awaddr_packed[2*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH] = {AXI_ADDR_WIDTH{1'b0}};
    assign s_axi_awlen_packed[2*8 +: 8] = 8'd0;
    assign s_axi_awsize_packed[2*3 +: 3] = 3'd0;
    assign s_axi_awburst_packed[2*2 +: 2] = 2'd0;
    assign s_axi_awvalid_packed[2] = 1'b0;
    
    // Port 3: Host DMA (read/write)
    assign s_axi_awid_packed[3*AXI_ID_WIDTH +: AXI_ID_WIDTH] = s3_axi_awid;
    assign s_axi_awaddr_packed[3*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH] = s3_axi_awaddr;
    assign s_axi_awlen_packed[3*8 +: 8] = s3_axi_awlen;
    assign s_axi_awsize_packed[3*3 +: 3] = s3_axi_awsize;
    assign s_axi_awburst_packed[3*2 +: 2] = s3_axi_awburst;
    assign s_axi_awvalid_packed[3] = s3_axi_awvalid;
    assign s3_axi_awready = s_axi_awready_packed[3];

    // Write data channel
    wire [NUM_AXI_PORTS*AXI_DATA_WIDTH-1:0]   s_axi_wdata_packed;
    wire [NUM_AXI_PORTS*AXI_DATA_WIDTH/8-1:0] s_axi_wstrb_packed;
    wire [NUM_AXI_PORTS-1:0]                  s_axi_wlast_packed;
    wire [NUM_AXI_PORTS-1:0]                  s_axi_wvalid_packed;
    wire [NUM_AXI_PORTS-1:0]                  s_axi_wready_packed;
    
    assign s_axi_wdata_packed[0*AXI_DATA_WIDTH +: AXI_DATA_WIDTH] = {AXI_DATA_WIDTH{1'b0}};
    assign s_axi_wstrb_packed[0*AXI_DATA_WIDTH/8 +: AXI_DATA_WIDTH/8] = {(AXI_DATA_WIDTH/8){1'b0}};
    assign s_axi_wlast_packed[0] = 1'b0;
    assign s_axi_wvalid_packed[0] = 1'b0;
    
    assign s_axi_wdata_packed[1*AXI_DATA_WIDTH +: AXI_DATA_WIDTH] = s1_axi_wdata;
    assign s_axi_wstrb_packed[1*AXI_DATA_WIDTH/8 +: AXI_DATA_WIDTH/8] = s1_axi_wstrb;
    assign s_axi_wlast_packed[1] = s1_axi_wlast;
    assign s_axi_wvalid_packed[1] = s1_axi_wvalid;
    assign s1_axi_wready = s_axi_wready_packed[1];
    
    assign s_axi_wdata_packed[2*AXI_DATA_WIDTH +: AXI_DATA_WIDTH] = {AXI_DATA_WIDTH{1'b0}};
    assign s_axi_wstrb_packed[2*AXI_DATA_WIDTH/8 +: AXI_DATA_WIDTH/8] = {(AXI_DATA_WIDTH/8){1'b0}};
    assign s_axi_wlast_packed[2] = 1'b0;
    assign s_axi_wvalid_packed[2] = 1'b0;
    
    assign s_axi_wdata_packed[3*AXI_DATA_WIDTH +: AXI_DATA_WIDTH] = s3_axi_wdata;
    assign s_axi_wstrb_packed[3*AXI_DATA_WIDTH/8 +: AXI_DATA_WIDTH/8] = s3_axi_wstrb;
    assign s_axi_wlast_packed[3] = s3_axi_wlast;
    assign s_axi_wvalid_packed[3] = s3_axi_wvalid;
    assign s3_axi_wready = s_axi_wready_packed[3];
    
    // Write response channel
    wire [NUM_AXI_PORTS*AXI_ID_WIDTH-1:0] s_axi_bid_packed;
    wire [NUM_AXI_PORTS*2-1:0]            s_axi_bresp_packed;
    wire [NUM_AXI_PORTS-1:0]              s_axi_bvalid_packed;
    wire [NUM_AXI_PORTS-1:0]              s_axi_bready_packed;
    
    assign s_axi_bready_packed[0] = 1'b0;
    assign s1_axi_bid = s_axi_bid_packed[1*AXI_ID_WIDTH +: AXI_ID_WIDTH];
    assign s1_axi_bresp = s_axi_bresp_packed[1*2 +: 2];
    assign s1_axi_bvalid = s_axi_bvalid_packed[1];
    assign s_axi_bready_packed[1] = s1_axi_bready;
    assign s_axi_bready_packed[2] = 1'b0;
    assign s3_axi_bid = s_axi_bid_packed[3*AXI_ID_WIDTH +: AXI_ID_WIDTH];
    assign s3_axi_bresp = s_axi_bresp_packed[3*2 +: 2];
    assign s3_axi_bvalid = s_axi_bvalid_packed[3];
    assign s_axi_bready_packed[3] = s3_axi_bready;

    // Read address channel
    wire [NUM_AXI_PORTS*AXI_ID_WIDTH-1:0]    s_axi_arid_packed;
    wire [NUM_AXI_PORTS*AXI_ADDR_WIDTH-1:0]  s_axi_araddr_packed;
    wire [NUM_AXI_PORTS*8-1:0]               s_axi_arlen_packed;
    wire [NUM_AXI_PORTS*3-1:0]               s_axi_arsize_packed;
    wire [NUM_AXI_PORTS*2-1:0]               s_axi_arburst_packed;
    wire [NUM_AXI_PORTS-1:0]                 s_axi_arvalid_packed;
    wire [NUM_AXI_PORTS-1:0]                 s_axi_arready_packed;
    
    // Port 0: LLM Read
    assign s_axi_arid_packed[0*AXI_ID_WIDTH +: AXI_ID_WIDTH] = s0_axi_arid;
    assign s_axi_araddr_packed[0*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH] = s0_axi_araddr;
    assign s_axi_arlen_packed[0*8 +: 8] = s0_axi_arlen;
    assign s_axi_arsize_packed[0*3 +: 3] = s0_axi_arsize;
    assign s_axi_arburst_packed[0*2 +: 2] = s0_axi_arburst;
    assign s_axi_arvalid_packed[0] = s0_axi_arvalid;
    assign s0_axi_arready = s_axi_arready_packed[0];
    
    // Port 1: No read (write-only for LLM KV cache)
    assign s_axi_arid_packed[1*AXI_ID_WIDTH +: AXI_ID_WIDTH] = {AXI_ID_WIDTH{1'b0}};
    assign s_axi_araddr_packed[1*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH] = {AXI_ADDR_WIDTH{1'b0}};
    assign s_axi_arlen_packed[1*8 +: 8] = 8'd0;
    assign s_axi_arsize_packed[1*3 +: 3] = 3'd0;
    assign s_axi_arburst_packed[1*2 +: 2] = 2'd0;
    assign s_axi_arvalid_packed[1] = 1'b0;
    
    // Port 2: Vision Read
    assign s_axi_arid_packed[2*AXI_ID_WIDTH +: AXI_ID_WIDTH] = s2_axi_arid;
    assign s_axi_araddr_packed[2*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH] = s2_axi_araddr;
    assign s_axi_arlen_packed[2*8 +: 8] = s2_axi_arlen;
    assign s_axi_arsize_packed[2*3 +: 3] = s2_axi_arsize;
    assign s_axi_arburst_packed[2*2 +: 2] = s2_axi_arburst;
    assign s_axi_arvalid_packed[2] = s2_axi_arvalid;
    assign s2_axi_arready = s_axi_arready_packed[2];
    
    // Port 3: Host DMA Read
    assign s_axi_arid_packed[3*AXI_ID_WIDTH +: AXI_ID_WIDTH] = s3_axi_arid;
    assign s_axi_araddr_packed[3*AXI_ADDR_WIDTH +: AXI_ADDR_WIDTH] = s3_axi_araddr;
    assign s_axi_arlen_packed[3*8 +: 8] = s3_axi_arlen;
    assign s_axi_arsize_packed[3*3 +: 3] = s3_axi_arsize;
    assign s_axi_arburst_packed[3*2 +: 2] = s3_axi_arburst;
    assign s_axi_arvalid_packed[3] = s3_axi_arvalid;
    assign s3_axi_arready = s_axi_arready_packed[3];

    // Read data channel
    wire [NUM_AXI_PORTS*AXI_ID_WIDTH-1:0]   s_axi_rid_packed;
    wire [NUM_AXI_PORTS*AXI_DATA_WIDTH-1:0] s_axi_rdata_packed;
    wire [NUM_AXI_PORTS*2-1:0]              s_axi_rresp_packed;
    wire [NUM_AXI_PORTS-1:0]                s_axi_rlast_packed;
    wire [NUM_AXI_PORTS-1:0]                s_axi_rvalid_packed;
    wire [NUM_AXI_PORTS-1:0]                s_axi_rready_packed;
    
    assign s0_axi_rid = s_axi_rid_packed[0*AXI_ID_WIDTH +: AXI_ID_WIDTH];
    assign s0_axi_rdata = s_axi_rdata_packed[0*AXI_DATA_WIDTH +: AXI_DATA_WIDTH];
    assign s0_axi_rresp = s_axi_rresp_packed[0*2 +: 2];
    assign s0_axi_rlast = s_axi_rlast_packed[0];
    assign s0_axi_rvalid = s_axi_rvalid_packed[0];
    assign s_axi_rready_packed[0] = s0_axi_rready;
    
    assign s_axi_rready_packed[1] = 1'b0;
    
    assign s2_axi_rid = s_axi_rid_packed[2*AXI_ID_WIDTH +: AXI_ID_WIDTH];
    assign s2_axi_rdata = s_axi_rdata_packed[2*AXI_DATA_WIDTH +: AXI_DATA_WIDTH];
    assign s2_axi_rresp = s_axi_rresp_packed[2*2 +: 2];
    assign s2_axi_rlast = s_axi_rlast_packed[2];
    assign s2_axi_rvalid = s_axi_rvalid_packed[2];
    assign s_axi_rready_packed[2] = s2_axi_rready;
    
    assign s3_axi_rid = s_axi_rid_packed[3*AXI_ID_WIDTH +: AXI_ID_WIDTH];
    assign s3_axi_rdata = s_axi_rdata_packed[3*AXI_DATA_WIDTH +: AXI_DATA_WIDTH];
    assign s3_axi_rresp = s_axi_rresp_packed[3*2 +: 2];
    assign s3_axi_rlast = s_axi_rlast_packed[3];
    assign s3_axi_rvalid = s_axi_rvalid_packed[3];
    assign s_axi_rready_packed[3] = s3_axi_rready;

    // =========================================================================
    // AXI Arbiter Instance
    // =========================================================================
    
    axi_arbiter #(
        .NUM_PORTS      (NUM_AXI_PORTS),
        .DATA_WIDTH     (AXI_DATA_WIDTH),
        .ADDR_WIDTH     (AXI_ADDR_WIDTH),
        .ID_WIDTH       (AXI_ID_WIDTH),
        .LEN_WIDTH      (8)
    ) u_axi_arbiter (
        .clk            (clk_core),
        .rst_n          (rst_n),
        
        // Slave ports (from masters)
        .s_axi_awid     (s_axi_awid_packed),
        .s_axi_awaddr   (s_axi_awaddr_packed),
        .s_axi_awlen    (s_axi_awlen_packed),
        .s_axi_awsize   (s_axi_awsize_packed),
        .s_axi_awburst  (s_axi_awburst_packed),
        .s_axi_awvalid  (s_axi_awvalid_packed),
        .s_axi_awready  (s_axi_awready_packed),
        
        .s_axi_wdata    (s_axi_wdata_packed),
        .s_axi_wstrb    (s_axi_wstrb_packed),
        .s_axi_wlast    (s_axi_wlast_packed),
        .s_axi_wvalid   (s_axi_wvalid_packed),
        .s_axi_wready   (s_axi_wready_packed),
        
        .s_axi_bid      (s_axi_bid_packed),
        .s_axi_bresp    (s_axi_bresp_packed),
        .s_axi_bvalid   (s_axi_bvalid_packed),
        .s_axi_bready   (s_axi_bready_packed),
        
        .s_axi_arid     (s_axi_arid_packed),
        .s_axi_araddr   (s_axi_araddr_packed),
        .s_axi_arlen    (s_axi_arlen_packed),
        .s_axi_arsize   (s_axi_arsize_packed),
        .s_axi_arburst  (s_axi_arburst_packed),
        .s_axi_arvalid  (s_axi_arvalid_packed),
        .s_axi_arready  (s_axi_arready_packed),
        
        .s_axi_rid      (s_axi_rid_packed),
        .s_axi_rdata    (s_axi_rdata_packed),
        .s_axi_rresp    (s_axi_rresp_packed),
        .s_axi_rlast    (s_axi_rlast_packed),
        .s_axi_rvalid   (s_axi_rvalid_packed),
        .s_axi_rready   (s_axi_rready_packed),
        
        // Master port (to DDR3 controller)
        .m_axi_awid     (arb_axi_awid),
        .m_axi_awaddr   (arb_axi_awaddr),
        .m_axi_awlen    (arb_axi_awlen),
        .m_axi_awsize   (arb_axi_awsize),
        .m_axi_awburst  (arb_axi_awburst),
        .m_axi_awvalid  (arb_axi_awvalid),
        .m_axi_awready  (arb_axi_awready),
        
        .m_axi_wdata    (arb_axi_wdata),
        .m_axi_wstrb    (arb_axi_wstrb),
        .m_axi_wlast    (arb_axi_wlast),
        .m_axi_wvalid   (arb_axi_wvalid),
        .m_axi_wready   (arb_axi_wready),
        
        .m_axi_bid      (arb_axi_bid),
        .m_axi_bresp    (arb_axi_bresp),
        .m_axi_bvalid   (arb_axi_bvalid),
        .m_axi_bready   (arb_axi_bready),
        
        .m_axi_arid     (arb_axi_arid),
        .m_axi_araddr   (arb_axi_araddr),
        .m_axi_arlen    (arb_axi_arlen),
        .m_axi_arsize   (arb_axi_arsize),
        .m_axi_arburst  (arb_axi_arburst),
        .m_axi_arvalid  (arb_axi_arvalid),
        .m_axi_arready  (arb_axi_arready),
        
        .m_axi_rid      (arb_axi_rid),
        .m_axi_rdata    (arb_axi_rdata),
        .m_axi_rresp    (arb_axi_rresp),
        .m_axi_rlast    (arb_axi_rlast),
        .m_axi_rvalid   (arb_axi_rvalid),
        .m_axi_rready   (arb_axi_rready),
        
        // Status
        .port_active        (port_active),
        .current_read_port  (current_rd_port),
        .current_write_port (current_wr_port)
    );

    // =========================================================================
    // DDR3 Controller - AXI to DDR3 Command Bridge
    // =========================================================================
    
    // Controller state machine
    localparam CTRL_IDLE      = 4'd0;
    localparam CTRL_ACTIVATE  = 4'd1;
    localparam CTRL_READ      = 4'd2;
    localparam CTRL_READ_WAIT = 4'd3;
    localparam CTRL_WRITE     = 4'd4;
    localparam CTRL_WRITE_DATA= 4'd5;
    localparam CTRL_PRECHARGE = 4'd6;
    localparam CTRL_REFRESH   = 4'd7;
    
    reg [3:0] ctrl_state;
    reg [15:0] ctrl_wait_cnt;
    reg [15:0] refresh_cnt;
    reg refresh_pending;
    
    // Address decoding - Bank-Row-Column interleaving
    // AXI addr: [31:0] -> DDR: bank[2:0], row[13:0], col[9:0]
    wire [DDR_BANK_WIDTH-1:0] ddr_bank = arb_axi_arvalid ? arb_axi_araddr[12:10] : arb_axi_awaddr[12:10];
    wire [13:0] ddr_row = arb_axi_arvalid ? arb_axi_araddr[26:13] : arb_axi_awaddr[26:13];
    wire [9:0]  ddr_col = arb_axi_arvalid ? arb_axi_araddr[9:0] : arb_axi_awaddr[9:0];
    
    // Bank tracking
    reg [7:0]  bank_active;
    reg [13:0] active_row [0:7];
    
    // PHY command encoding
    localparam CMD_NOP   = 3'b111;
    localparam CMD_ACT   = 3'b011;
    localparam CMD_READ  = 3'b101;
    localparam CMD_WRITE = 3'b100;
    localparam CMD_PRE   = 3'b010;
    localparam CMD_REF   = 3'b001;
    
    reg [2:0]               phy_cmd_reg;
    reg                     phy_cmd_valid_reg;
    reg [DDR_ADDR_WIDTH-1:0] phy_addr_reg;
    reg [DDR_BANK_WIDTH-1:0] phy_bank_reg;
    
    assign phy_cmd = phy_cmd_reg;
    assign phy_cmd_valid = phy_cmd_valid_reg;
    assign phy_addr = phy_addr_reg;
    assign phy_bank = phy_bank_reg;

    // AXI response handling
    reg [AXI_ID_WIDTH-1:0]   pending_arid;
    reg [AXI_ID_WIDTH-1:0]   pending_awid;
    reg [7:0]                pending_arlen;
    reg [7:0]                rd_beat_cnt;
    reg [7:0]                wr_beat_cnt;
    
    // AXI master port ready/valid registers
    reg arb_axi_arready_reg;
    reg arb_axi_awready_reg;
    reg arb_axi_wready_reg;
    reg arb_axi_rvalid_reg;
    reg arb_axi_rlast_reg;
    reg [AXI_DATA_WIDTH-1:0] arb_axi_rdata_reg;
    reg arb_axi_bvalid_reg;
    
    assign arb_axi_arready = arb_axi_arready_reg;
    assign arb_axi_awready = arb_axi_awready_reg;
    assign arb_axi_wready = arb_axi_wready_reg;
    assign arb_axi_rvalid = arb_axi_rvalid_reg;
    assign arb_axi_rlast = arb_axi_rlast_reg;
    assign arb_axi_rdata = arb_axi_rdata_reg;
    assign arb_axi_rid = pending_arid;
    assign arb_axi_rresp = 2'b00;  // OKAY
    assign arb_axi_bvalid = arb_axi_bvalid_reg;
    assign arb_axi_bid = pending_awid;
    assign arb_axi_bresp = 2'b00;  // OKAY
    
    // Write data path
    reg [DDR_DATA_WIDTH*2-1:0] phy_wdata_reg;
    reg [DDR_DATA_WIDTH/4-1:0] phy_wdata_mask_reg;
    reg                        phy_wdata_valid_reg;
    
    assign phy_wdata = phy_wdata_reg;
    assign phy_wdata_mask = phy_wdata_mask_reg;
    assign phy_wdata_valid = phy_wdata_valid_reg;
    
    integer j;

    // Controller state machine
    always @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state <= CTRL_IDLE;
            ctrl_wait_cnt <= 16'd0;
            refresh_cnt <= 16'd0;
            refresh_pending <= 1'b0;
            
            phy_cmd_reg <= CMD_NOP;
            phy_cmd_valid_reg <= 1'b0;
            phy_addr_reg <= {DDR_ADDR_WIDTH{1'b0}};
            phy_bank_reg <= {DDR_BANK_WIDTH{1'b0}};
            
            bank_active <= 8'd0;
            for (j = 0; j < 8; j = j + 1) begin
                active_row[j] <= 14'd0;
            end
            
            arb_axi_arready_reg <= 1'b0;
            arb_axi_awready_reg <= 1'b0;
            arb_axi_wready_reg <= 1'b0;
            arb_axi_rvalid_reg <= 1'b0;
            arb_axi_rlast_reg <= 1'b0;
            arb_axi_rdata_reg <= {AXI_DATA_WIDTH{1'b0}};
            arb_axi_bvalid_reg <= 1'b0;
            
            pending_arid <= {AXI_ID_WIDTH{1'b0}};
            pending_awid <= {AXI_ID_WIDTH{1'b0}};
            pending_arlen <= 8'd0;
            rd_beat_cnt <= 8'd0;
            wr_beat_cnt <= 8'd0;
            
            phy_wdata_reg <= {(DDR_DATA_WIDTH*2){1'b0}};
            phy_wdata_mask_reg <= {(DDR_DATA_WIDTH/4){1'b0}};
            phy_wdata_valid_reg <= 1'b0;
            
        end else begin
            // Default: clear command
            phy_cmd_valid_reg <= 1'b0;
            arb_axi_arready_reg <= 1'b0;
            arb_axi_awready_reg <= 1'b0;
            phy_wdata_valid_reg <= 1'b0;
            
            // Clear response valid when accepted
            if (arb_axi_rvalid_reg && arb_axi_rready) begin
                arb_axi_rvalid_reg <= 1'b0;
                arb_axi_rlast_reg <= 1'b0;
            end
            if (arb_axi_bvalid_reg && arb_axi_bready) begin
                arb_axi_bvalid_reg <= 1'b0;
            end
            
            // Refresh counter
            if (phy_ready) begin
                if (refresh_cnt >= tREFI) begin
                    refresh_pending <= 1'b1;
                    refresh_cnt <= 16'd0;
                end else begin
                    refresh_cnt <= refresh_cnt + 1'b1;
                end
            end
            
            case (ctrl_state)
                CTRL_IDLE: begin
                    if (phy_ready) begin
                        // Priority: Refresh > Read > Write
                        if (refresh_pending) begin
                            // Need to precharge all banks first
                            if (|bank_active) begin
                                phy_cmd_reg <= CMD_PRE;
                                phy_cmd_valid_reg <= 1'b1;
                                phy_addr_reg[10] <= 1'b1;  // All banks
                                bank_active <= 8'd0;
                                ctrl_wait_cnt <= tRP;
                                ctrl_state <= CTRL_PRECHARGE;
                            end else begin
                                ctrl_state <= CTRL_REFRESH;
                            end
                            
                        end else if (arb_axi_arvalid) begin
                            // Read request
                            pending_arid <= arb_axi_arid;
                            pending_arlen <= arb_axi_arlen;
                            rd_beat_cnt <= arb_axi_arlen;
                            arb_axi_arready_reg <= 1'b1;
                            
                            // Check if row already active
                            if (bank_active[ddr_bank] && active_row[ddr_bank] == ddr_row) begin
                                // Row hit
                                ctrl_state <= CTRL_READ;
                            end else if (bank_active[ddr_bank]) begin
                                // Row miss - precharge first
                                phy_cmd_reg <= CMD_PRE;
                                phy_cmd_valid_reg <= 1'b1;
                                phy_bank_reg <= ddr_bank;
                                phy_addr_reg[10] <= 1'b0;
                                bank_active[ddr_bank] <= 1'b0;
                                ctrl_wait_cnt <= tRP;
                                ctrl_state <= CTRL_PRECHARGE;
                            end else begin
                                ctrl_state <= CTRL_ACTIVATE;
                            end
                            
                        end else if (arb_axi_awvalid) begin
                            // Write request
                            pending_awid <= arb_axi_awid;
                            arb_axi_awready_reg <= 1'b1;
                            wr_beat_cnt <= arb_axi_awlen;
                            
                            if (bank_active[ddr_bank] && active_row[ddr_bank] == ddr_row) begin
                                ctrl_state <= CTRL_WRITE;
                            end else if (bank_active[ddr_bank]) begin
                                phy_cmd_reg <= CMD_PRE;
                                phy_cmd_valid_reg <= 1'b1;
                                phy_bank_reg <= ddr_bank;
                                phy_addr_reg[10] <= 1'b0;
                                bank_active[ddr_bank] <= 1'b0;
                                ctrl_wait_cnt <= tRP;
                                ctrl_state <= CTRL_PRECHARGE;
                            end else begin
                                ctrl_state <= CTRL_ACTIVATE;
                            end
                        end
                    end
                end
                
                CTRL_ACTIVATE: begin
                    if (ctrl_wait_cnt > 0) begin
                        ctrl_wait_cnt <= ctrl_wait_cnt - 1'b1;
                    end else if (phy_cmd_ready) begin
                        phy_cmd_reg <= CMD_ACT;
                        phy_cmd_valid_reg <= 1'b1;
                        phy_bank_reg <= ddr_bank;
                        phy_addr_reg <= {{(DDR_ADDR_WIDTH-14){1'b0}}, ddr_row};
                        
                        bank_active[ddr_bank] <= 1'b1;
                        active_row[ddr_bank] <= ddr_row;
                        
                        ctrl_wait_cnt <= tRCD;
                        if (|rd_beat_cnt || rd_beat_cnt == 8'd0 && pending_arlen != 8'd0) begin
                            ctrl_state <= CTRL_READ;
                        end else begin
                            ctrl_state <= CTRL_WRITE;
                        end
                    end
                end
                
                CTRL_READ: begin
                    if (ctrl_wait_cnt > 0) begin
                        ctrl_wait_cnt <= ctrl_wait_cnt - 1'b1;
                    end else if (phy_cmd_ready) begin
                        phy_cmd_reg <= CMD_READ;
                        phy_cmd_valid_reg <= 1'b1;
                        phy_bank_reg <= ddr_bank;
                        phy_addr_reg <= {{(DDR_ADDR_WIDTH-10){1'b0}}, ddr_col};
                        
                        ctrl_wait_cnt <= CL + 4;  // CL + burst
                        ctrl_state <= CTRL_READ_WAIT;
                    end
                end
                
                CTRL_READ_WAIT: begin
                    if (ctrl_wait_cnt > 0) begin
                        ctrl_wait_cnt <= ctrl_wait_cnt - 1'b1;
                    end
                    
                    // Capture read data from PHY
                    if (phy_rdata_valid) begin
                        arb_axi_rdata_reg <= phy_rdata[AXI_DATA_WIDTH-1:0];
                        arb_axi_rvalid_reg <= 1'b1;
                        
                        if (rd_beat_cnt == 0) begin
                            arb_axi_rlast_reg <= 1'b1;
                            ctrl_state <= CTRL_IDLE;
                        end else begin
                            rd_beat_cnt <= rd_beat_cnt - 1'b1;
                        end
                    end
                end
                
                CTRL_WRITE: begin
                    if (ctrl_wait_cnt > 0) begin
                        ctrl_wait_cnt <= ctrl_wait_cnt - 1'b1;
                    end else if (phy_cmd_ready) begin
                        phy_cmd_reg <= CMD_WRITE;
                        phy_cmd_valid_reg <= 1'b1;
                        phy_bank_reg <= ddr_bank;
                        phy_addr_reg <= {{(DDR_ADDR_WIDTH-10){1'b0}}, ddr_col};
                        
                        arb_axi_wready_reg <= 1'b1;
                        ctrl_state <= CTRL_WRITE_DATA;
                    end
                end
                
                CTRL_WRITE_DATA: begin
                    if (arb_axi_wvalid && arb_axi_wready_reg) begin
                        // Forward write data to PHY
                        phy_wdata_reg <= {arb_axi_wdata, arb_axi_wdata[DDR_DATA_WIDTH-1:0]};
                        phy_wdata_mask_reg <= ~arb_axi_wstrb[DDR_DATA_WIDTH/8-1:0];
                        phy_wdata_valid_reg <= 1'b1;
                        
                        if (arb_axi_wlast) begin
                            arb_axi_wready_reg <= 1'b0;
                            arb_axi_bvalid_reg <= 1'b1;
                            ctrl_state <= CTRL_IDLE;
                        end else begin
                            wr_beat_cnt <= wr_beat_cnt - 1'b1;
                        end
                    end
                end
                
                CTRL_PRECHARGE: begin
                    if (ctrl_wait_cnt > 0) begin
                        ctrl_wait_cnt <= ctrl_wait_cnt - 1'b1;
                    end else begin
                        if (refresh_pending) begin
                            ctrl_state <= CTRL_REFRESH;
                        end else begin
                            ctrl_state <= CTRL_ACTIVATE;
                        end
                    end
                end
                
                CTRL_REFRESH: begin
                    if (phy_cmd_ready) begin
                        phy_cmd_reg <= CMD_REF;
                        phy_cmd_valid_reg <= 1'b1;
                        refresh_pending <= 1'b0;
                        ctrl_wait_cnt <= tRFC;
                        ctrl_state <= CTRL_IDLE;
                    end
                end
                
                default: ctrl_state <= CTRL_IDLE;
            endcase
        end
    end

    // =========================================================================
    // DDR3 PHY Instance
    // =========================================================================
    
    wire cal_done_w;
    wire cal_error_w;
    
    ddr3_phy #(
        .DATA_WIDTH     (DDR_DATA_WIDTH),
        .ADDR_WIDTH     (DDR_ADDR_WIDTH),
        .BANK_WIDTH     (DDR_BANK_WIDTH),
        .NUM_RANKS      (1),
        .BURST_LEN      (8),
        .CL             (CL),
        .CWL            (CWL)
    ) u_ddr3_phy (
        .clk_core       (clk_core),
        .clk_ddr        (clk_ddr),
        .clk_ddr_90     (clk_ddr_90),
        .clk_ddr_180    (clk_ddr_180),
        .clk_ref        (clk_ref),
        .rst_n          (rst_n),
        
        // Controller interface
        .cmd            (phy_cmd),
        .cmd_valid      (phy_cmd_valid),
        .cmd_ready      (phy_cmd_ready),
        .addr           (phy_addr),
        .bank           (phy_bank),
        .wdata          (phy_wdata),
        .wdata_mask     (phy_wdata_mask),
        .wdata_valid    (phy_wdata_valid),
        .wdata_ready    (phy_wdata_ready),
        .rdata          (phy_rdata),
        .rdata_valid    (phy_rdata_valid),
        
        // DDR3 physical signals
        .ddr3_ck_p      (ddr3_ck_p),
        .ddr3_ck_n      (ddr3_ck_n),
        .ddr3_cs_n      (ddr3_cs_n),
        .ddr3_ras_n     (ddr3_ras_n),
        .ddr3_cas_n     (ddr3_cas_n),
        .ddr3_we_n      (ddr3_we_n),
        .ddr3_cke       (ddr3_cke),
        .ddr3_reset_n   (ddr3_reset_n),
        .ddr3_ba        (ddr3_ba),
        .ddr3_addr      (ddr3_addr),
        .ddr3_dq        (ddr3_dq),
        .ddr3_dqs_p     (ddr3_dqs_p),
        .ddr3_dqs_n     (ddr3_dqs_n),
        .ddr3_dm        (ddr3_dm),
        .ddr3_odt       (ddr3_odt),
        
        // Calibration
        .cal_start      (1'b1),
        .cal_done       (cal_done_w),
        .cal_error      (cal_error_w),
        .rd_dqs_delay   (),
        .wr_dqs_delay   (),
        .phy_ready      (phy_ready)
    );
    
    // =========================================================================
    // Status Outputs
    // =========================================================================
    
    assign init_done = phy_ready;
    assign cal_complete = cal_done_w;
    assign debug_state = {4'b0, ctrl_state};

endmodule

`default_nettype wire
