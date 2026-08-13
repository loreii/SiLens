// =============================================================================
// SiLens FPGA Wrapper for Intel FPGAs
// =============================================================================
// Top-level FPGA wrapper with PLL clock management and Intel-specific I/O.
//
// Target: Intel Arria 10 GX or Cyclone 10 GX
//
// License: Apache 2.0
// =============================================================================

module silens_fpga_wrapper_intel #(
    parameter INPUT_CLK_FREQ_MHZ = 100,
    parameter CORE_CLK_FREQ_MHZ  = 100,
    parameter VISION_DIM  = 768,
    parameter LLM_DIM     = 576,
    parameter VOCAB_SIZE  = 49152,
    parameter MAX_SEQ_LEN = 8192,
    parameter IMG_SIZE    = 384,
    parameter PATCH_SIZE  = 16,
    parameter ACT_WIDTH   = 8
)(
    // Clock input (single-ended or differential)
    input  wire         clk_osc,
    input  wire         clk_osc_p,
    input  wire         clk_osc_n,
    
    // System reset
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
    output wire [7:0]   led
);

    // =========================================================================
    // Clock generation (Intel IOPLL)
    // =========================================================================
    
    wire core_clk;
    wire fast_clk;
    wire pll_locked;

    // Intel IOPLL instantiation (platform-specific)
    // In actual implementation, use Platform Designer generated PLL
    `ifdef ARRIA10
    // Arria 10 IOPLL
    iopll pll_inst (
        .refclk(clk_osc_p),
        .rst(~sys_rst_n),
        .outclk_0(core_clk),
        .outclk_1(fast_clk),
        .locked(pll_locked)
    );
    `else
    // Cyclone 10 GX IOPLL
    iopll pll_inst (
        .refclk(clk_osc),
        .rst(~sys_rst_n),
        .outclk_0(core_clk),
        .outclk_1(fast_clk),
        .locked(pll_locked)
    );
    `endif
    
    // =========================================================================
    // Reset synchronization
    // =========================================================================
    
    reg [3:0] rst_sync_r;
    wire      rst_n_sync;
    
    always @(posedge core_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            rst_sync_r <= 4'b0000;
        end else if (pll_locked) begin
            rst_sync_r <= {rst_sync_r[2:0], 1'b1};
        end else begin
            rst_sync_r <= 4'b0000;
        end
    end
    
    assign rst_n_sync = rst_sync_r[3];
    
    // =========================================================================
    // Internal signals
    // =========================================================================
    
    localparam IN_CHANNELS = 3;
    localparam NUM_PATCHES = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE);
    
    wire [127:0] pcie_rx_data;
    wire         pcie_rx_valid;
    wire         pcie_rx_ready;
    wire [127:0] pcie_tx_data;
    wire         pcie_tx_valid;
    wire         pcie_tx_ready;
    
    wire frame_start;
    wire seq_start;
    wire generate_start;
    wire [7:0] status_leds;
    wire heartbeat;
    wire error_flag;
    wire vision_busy;
    wire llm_busy;
    
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
        .clk(core_clk),
        .rst_n(rst_n_sync),
        
        .pcie_clk(core_clk),
        .pcie_rst_n(pcie_rst_n),
        .pcie_rx_data(pcie_rx_data),
        .pcie_rx_valid(pcie_rx_valid),
        .pcie_rx_ready(pcie_rx_ready),
        .pcie_tx_data(pcie_tx_data),
        .pcie_tx_valid(pcie_tx_valid),
        .pcie_tx_ready(pcie_tx_ready),
        
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .pixel_ready(pixel_ready),
        
        .token_in(token_in),
        .token_in_valid(token_in_valid),
        .token_in_ready(token_in_ready),
        
        .token_out(token_out),
        .token_out_valid(token_out_valid),
        .token_out_ready(token_out_ready),
        
        .frame_start(frame_start),
        .seq_start(seq_start),
        .generate(generate_start),
        
        .status_leds(status_leds),
        .heartbeat(heartbeat),
        .error_flag(error_flag),
        .vision_busy(vision_busy),
        .llm_busy(llm_busy)
    );
    
    // =========================================================================
    // Debug interface
    // =========================================================================
    
    assign frame_start    = btn[0];
    assign seq_start      = btn[1];
    assign generate_start = btn[2];
    
    assign led[7]   = heartbeat;
    assign led[6]   = pll_locked;
    assign led[5]   = rst_n_sync;
    assign led[4]   = error_flag;
    assign led[3:0] = status_leds[3:0];
    
    // =========================================================================
    // Tie-offs
    // =========================================================================
    
    assign pcie_tx_p = 4'b0;
    assign pcie_tx_n = 4'b1;
    
    assign pcie_rx_data  = 128'b0;
    assign pcie_rx_valid = 1'b0;
    assign pcie_tx_ready = 1'b1;
    
    assign pixel_in       = {(IN_CHANNELS*ACT_WIDTH){1'b0}};
    assign pixel_valid    = 1'b0;
    assign token_in       = {$clog2(VOCAB_SIZE){1'b0}};
    assign token_in_valid = 1'b0;
    assign token_out_ready = 1'b1;

endmodule
