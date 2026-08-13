// =============================================================================
// SiLens FPGA Wrapper for Xilinx FPGAs
// =============================================================================
// Top-level FPGA wrapper with clock management (MMCM/PLL), I/O buffers,
// and FPGA-specific infrastructure.
//
// Target: Xilinx Artix-7 (XC7A200T) or Kintex-7 (XC7K325T)
//
// Features:
//   - MMCM for clock generation (100MHz core, 250MHz PCIe)
//   - Differential clock input
//   - IDELAYCTRL for high-speed I/O
//   - Debug JTAG/ILA interface
//
// License: Apache 2.0
// =============================================================================

module silens_fpga_wrapper #(
    // Clock parameters
    parameter INPUT_CLK_FREQ_MHZ  = 200,           // Input differential clock
    parameter CORE_CLK_FREQ_MHZ   = 100,           // Core clock
    parameter PCIE_CLK_FREQ_MHZ   = 250,           // PCIe interface clock
    
    // Model parameters (inherited from silens_top)
    parameter VISION_DIM   = 768,
    parameter LLM_DIM      = 576,
    parameter VOCAB_SIZE   = 49152,
    parameter MAX_SEQ_LEN  = 8192,
    parameter IMG_SIZE     = 384,
    parameter PATCH_SIZE   = 16,
    parameter ACT_WIDTH    = 8
)(
    // Differential clock input
    input  wire         clk_p,
    input  wire         clk_n,
    
    // System reset (active low)
    input  wire         sys_rst_n,
    
    // PCIe interface
    input  wire         pcie_refclk_p,
    input  wire         pcie_refclk_n,
    input  wire         pcie_rst_n,
    input  wire [3:0]   pcie_rx_p,
    input  wire [3:0]   pcie_rx_n,
    output wire [3:0]   pcie_tx_p,
    output wire [3:0]   pcie_tx_n,
    
    // User I/O
    input  wire [3:0]   btn,
    input  wire [3:0]   sw,
    output wire [7:0]   led,
    
    // UART (debug)
    input  wire         uart_rx,
    output wire         uart_tx,
    
    // DDR3 interface (optional external memory)
    output wire [14:0]  ddr3_addr,
    output wire [2:0]   ddr3_ba,
    output wire         ddr3_ras_n,
    output wire         ddr3_cas_n,
    output wire         ddr3_we_n,
    output wire         ddr3_reset_n,
    output wire [0:0]   ddr3_ck_p,
    output wire [0:0]   ddr3_ck_n,
    output wire [0:0]   ddr3_cke,
    output wire [0:0]   ddr3_cs_n,
    output wire [3:0]   ddr3_dm,
    output wire [0:0]   ddr3_odt,
    inout  wire [31:0]  ddr3_dq,
    inout  wire [3:0]   ddr3_dqs_p,
    inout  wire [3:0]   ddr3_dqs_n
);

    // =========================================================================
    // Clock generation (MMCM)
    // =========================================================================
    
    wire clk_in_buf;
    wire core_clk;
    wire core_clk_buf;
    wire pcie_clk_internal;
    wire mmcm_locked;
    wire mmcm_feedback;
    
    // Differential input buffer
    IBUFDS #(
        .DIFF_TERM("FALSE"),
        .IBUF_LOW_PWR("FALSE"),
        .IOSTANDARD("LVDS_25")
    ) ibufds_clk (
        .O(clk_in_buf),
        .I(clk_p),
        .IB(clk_n)
    );
    
    // MMCM for clock generation
    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKFBOUT_MULT_F(5.0),                     // VCO = 200MHz * 5 = 1000MHz
        .CLKFBOUT_PHASE(0.0),
        .CLKIN1_PERIOD(5.0),                       // 200MHz input
        .CLKOUT0_DIVIDE_F(10.0),                   // 100MHz core clock
        .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT0_PHASE(0.0),
        .CLKOUT1_DIVIDE(4),                        // 250MHz PCIe clock
        .CLKOUT1_DUTY_CYCLE(0.5),
        .CLKOUT1_PHASE(0.0),
        .CLKOUT2_DIVIDE(8),                        // 125MHz aux clock
        .CLKOUT2_DUTY_CYCLE(0.5),
        .CLKOUT2_PHASE(0.0),
        .DIVCLK_DIVIDE(1),
        .REF_JITTER1(0.01),
        .STARTUP_WAIT("FALSE")
    ) mmcm_inst (
        .CLKOUT0(core_clk),
        .CLKOUT1(pcie_clk_internal),
        .CLKOUT2(),                                // Unused
        .CLKOUT3(),
        .CLKOUT4(),
        .CLKOUT5(),
        .CLKOUT6(),
        .CLKFBOUT(mmcm_feedback),
        .LOCKED(mmcm_locked),
        .CLKIN1(clk_in_buf),
        .PWRDWN(1'b0),
        .RST(~sys_rst_n),
        .CLKFBIN(mmcm_feedback)
    );
    
    // Global clock buffers
    BUFG bufg_core (
        .I(core_clk),
        .O(core_clk_buf)
    );
    
    // =========================================================================
    // Reset synchronization
    // =========================================================================
    
    reg [3:0] rst_sync_r;
    wire      rst_n_sync;
    
    always @(posedge core_clk_buf or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            rst_sync_r <= 4'b0000;
        end else if (mmcm_locked) begin
            rst_sync_r <= {rst_sync_r[2:0], 1'b1};
        end else begin
            rst_sync_r <= 4'b0000;
        end
    end
    
    assign rst_n_sync = rst_sync_r[3];
    
    // =========================================================================
    // IDELAYCTRL for high-speed I/O (required for IDELAY elements)
    // =========================================================================
    
    wire idelay_rdy;
    
    (* IODELAY_GROUP = "silens_delay_grp" *)
    IDELAYCTRL idelayctrl_inst (
        .RDY(idelay_rdy),
        .REFCLK(core_clk_buf),                     // 200MHz reference
        .RST(~rst_n_sync)
    );
    
    // =========================================================================
    // Internal signals
    // =========================================================================
    
    localparam IN_CHANNELS = 3;
    localparam NUM_PATCHES = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE);
    
    // PCIe signals (directly connected to hard IP)
    wire [127:0] pcie_rx_data;
    wire         pcie_rx_valid;
    wire         pcie_rx_ready;
    wire [127:0] pcie_tx_data;
    wire         pcie_tx_valid;
    wire         pcie_tx_ready;
    
    // Control/status signals
    wire frame_start;
    wire seq_start;
    wire generate_start;
    wire [7:0] status_leds;
    wire heartbeat;
    wire error_flag;
    wire vision_busy;
    wire llm_busy;
    
    // Token interfaces
    wire [IN_CHANNELS*ACT_WIDTH-1:0] pixel_in;
    wire pixel_valid;
    wire pixel_ready;
    
    wire [$clog2(VOCAB_SIZE)-1:0] token_in;
    wire token_in_valid;
    wire token_in_ready;
    
    wire [$clog2(VOCAB_SIZE)-1:0] token_out;
    wire token_out_valid;
    wire token_out_ready;
    
    // =========================================================================
    // SiLens core instantiation
    // =========================================================================
    
    silens_top #(
        .CLK_FREQ_MHZ(CORE_CLK_FREQ_MHZ),
        .VISION_DIM(VISION_DIM),
        .LLM_DIM(LLM_DIM),
        .VOCAB_SIZE(VOCAB_SIZE),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .IMG_SIZE(IMG_SIZE),
        .PATCH_SIZE(PATCH_SIZE),
        .ACT_WIDTH(ACT_WIDTH)
    ) u_silens_top (
        .clk(core_clk_buf),
        .rst_n(rst_n_sync),
        
        // PCIe interface
        .pcie_clk(pcie_clk_internal),
        .pcie_rst_n(pcie_rst_n),
        .pcie_rx_data(pcie_rx_data),
        .pcie_rx_valid(pcie_rx_valid),
        .pcie_rx_ready(pcie_rx_ready),
        .pcie_tx_data(pcie_tx_data),
        .pcie_tx_valid(pcie_tx_valid),
        .pcie_tx_ready(pcie_tx_ready),
        
        // Pixel input
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .pixel_ready(pixel_ready),
        
        // Token I/O
        .token_in(token_in),
        .token_in_valid(token_in_valid),
        .token_in_ready(token_in_ready),
        
        .token_out(token_out),
        .token_out_valid(token_out_valid),
        .token_out_ready(token_out_ready),
        
        // Control
        .frame_start(frame_start),
        .seq_start(seq_start),
        .generate(generate_start),
        
        // Status
        .status_leds(status_leds),
        .heartbeat(heartbeat),
        .error_flag(error_flag),
        .vision_busy(vision_busy),
        .llm_busy(llm_busy)
    );
    
    // =========================================================================
    // Debug interface (directly controlled via buttons/switches)
    // =========================================================================
    
    // Buttons trigger operations
    assign frame_start    = btn[0];
    assign seq_start      = btn[1];
    assign generate_start = btn[2];
    
    // LEDs show status
    assign led[7]   = heartbeat;
    assign led[6]   = mmcm_locked;
    assign led[5]   = idelay_rdy;
    assign led[4]   = error_flag;
    assign led[3:0] = status_leds[3:0];
    
    // =========================================================================
    // UART debug interface (directly controlled)
    // =========================================================================
    
    assign uart_tx = 1'b1;  // Idle high (no transmission)
    
    // =========================================================================
    // DDR3 interface placeholder
    // =========================================================================
    
    // TODO: Instantiate MIG IP for DDR3 access
    assign ddr3_addr    = 15'b0;
    assign ddr3_ba      = 3'b0;
    assign ddr3_ras_n   = 1'b1;
    assign ddr3_cas_n   = 1'b1;
    assign ddr3_we_n    = 1'b1;
    assign ddr3_reset_n = 1'b1;
    assign ddr3_ck_p    = 1'b0;
    assign ddr3_ck_n    = 1'b1;
    assign ddr3_cke     = 1'b0;
    assign ddr3_cs_n    = 1'b1;
    assign ddr3_dm      = 4'b0;
    assign ddr3_odt     = 1'b0;
    
    // =========================================================================
    // PCIe placeholder (vendor hard IP integration required)
    // =========================================================================
    
    // PCIe connections would go to Xilinx PCIe hard IP
    assign pcie_tx_p = 4'b0;
    assign pcie_tx_n = 4'b1;
    
    // Tie off internal PCIe signals for now
    assign pcie_rx_data  = 128'b0;
    assign pcie_rx_valid = 1'b0;
    assign pcie_tx_ready = 1'b1;
    
    // Tie off unused interfaces
    assign pixel_in       = {(IN_CHANNELS*ACT_WIDTH){1'b0}};
    assign pixel_valid    = 1'b0;
    assign token_in       = {$clog2(VOCAB_SIZE){1'b0}};
    assign token_in_valid = 1'b0;
    assign token_out_ready = 1'b1;

endmodule
