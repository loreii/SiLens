// =============================================================================
// SiLens - Configuration and Status Register File
// =============================================================================
// Memory-mapped register file accessible via PCIe BAR0.
//
// Register Map:
//   0x000 - Control register
//   0x004 - Status register
//   0x008 - Version register
//   0x00C - Interrupt enable
//   0x010 - Interrupt status
//   0x020 - Image buffer address (low)
//   0x024 - Image buffer address (high)
//   0x028 - Image dimensions
//   0x030 - Output buffer address (low)
//   0x034 - Output buffer address (high)
//   0x038 - Output length
//   0x100 - DMA control
//   0x104 - DMA status
//   0x110 - DMA H2D host address (low)
//   0x114 - DMA H2D host address (high)
//   0x118 - DMA H2D local address
//   0x11C - DMA H2D length
//   0x120 - DMA D2H host address (low)
//   0x124 - DMA D2H host address (high)
//   0x128 - DMA D2H local address
//   0x12C - DMA D2H length
//   0x200 - Debug register 0
//   0x204 - Debug register 1
//   ...
//
// License: Apache 2.0
// =============================================================================

module register_file #(
    parameter ADDR_WIDTH = 12,                      // 4KB address space
    parameter DATA_WIDTH = 32,                      // Register width
    parameter VERSION    = 32'h0001_0000            // Version 1.0.0
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Register interface (from PCIe)
    input  wire [ADDR_WIDTH-1:0]        reg_addr,
    input  wire [DATA_WIDTH-1:0]        reg_wr_data,
    input  wire                         reg_wr_en,
    input  wire                         reg_rd_en,
    output reg  [DATA_WIDTH-1:0]        reg_rd_data,
    output reg                          reg_rd_valid,
    
    // Control outputs
    output wire                         soft_reset,
    output wire                         inference_start,
    output wire                         frame_start,
    output wire                         seq_start,
    output wire                         generate_start,
    output wire [31:0]                  max_tokens,
    
    // DMA control outputs
    output wire                         dma_enable,
    output wire                         dma_h2d_start,
    output wire                         dma_d2h_start,
    output wire [63:0]                  dma_h2d_host_addr,
    output wire [31:0]                  dma_h2d_local_addr,
    output wire [23:0]                  dma_h2d_length,
    output wire [63:0]                  dma_d2h_host_addr,
    output wire [31:0]                  dma_d2h_local_addr,
    output wire [23:0]                  dma_d2h_length,
    
    // Status inputs
    input  wire                         vision_busy,
    input  wire                         llm_busy,
    input  wire                         dma_h2d_busy,
    input  wire                         dma_d2h_busy,
    input  wire                         dma_h2d_done,
    input  wire                         dma_d2h_done,
    input  wire                         error_flag,
    input  wire [15:0]                  tokens_generated,
    
    // Interrupt outputs
    output wire                         interrupt_out,
    input  wire                         interrupt_ack
);


    // =========================================================================
    // Register addresses
    // =========================================================================
    
    localparam ADDR_CTRL         = 12'h000;
    localparam ADDR_STATUS       = 12'h004;
    localparam ADDR_VERSION      = 12'h008;
    localparam ADDR_INT_ENABLE   = 12'h00C;
    localparam ADDR_INT_STATUS   = 12'h010;
    localparam ADDR_IMG_ADDR_LO  = 12'h020;
    localparam ADDR_IMG_ADDR_HI  = 12'h024;
    localparam ADDR_IMG_DIM      = 12'h028;
    localparam ADDR_OUT_ADDR_LO  = 12'h030;
    localparam ADDR_OUT_ADDR_HI  = 12'h034;
    localparam ADDR_OUT_LEN      = 12'h038;
    localparam ADDR_MAX_TOKENS   = 12'h03C;
    localparam ADDR_DMA_CTRL     = 12'h100;
    localparam ADDR_DMA_STATUS   = 12'h104;
    localparam ADDR_H2D_ADDR_LO  = 12'h110;
    localparam ADDR_H2D_ADDR_HI  = 12'h114;
    localparam ADDR_H2D_LOCAL    = 12'h118;
    localparam ADDR_H2D_LEN      = 12'h11C;
    localparam ADDR_D2H_ADDR_LO  = 12'h120;
    localparam ADDR_D2H_ADDR_HI  = 12'h124;
    localparam ADDR_D2H_LOCAL    = 12'h128;
    localparam ADDR_D2H_LEN      = 12'h12C;
    localparam ADDR_DEBUG0       = 12'h200;
    localparam ADDR_DEBUG1       = 12'h204;
    localparam ADDR_TOK_COUNT    = 12'h208;
    
    // =========================================================================
    // Register storage
    // =========================================================================
    
    reg [31:0] r_ctrl;
    reg [31:0] r_int_enable;
    reg [31:0] r_int_status;
    reg [31:0] r_img_addr_lo;
    reg [31:0] r_img_addr_hi;
    reg [31:0] r_img_dim;
    reg [31:0] r_out_addr_lo;
    reg [31:0] r_out_addr_hi;
    reg [31:0] r_out_len;
    reg [31:0] r_max_tokens;
    reg [31:0] r_dma_ctrl;
    reg [31:0] r_h2d_addr_lo;
    reg [31:0] r_h2d_addr_hi;
    reg [31:0] r_h2d_local;
    reg [31:0] r_h2d_len;
    reg [31:0] r_d2h_addr_lo;
    reg [31:0] r_d2h_addr_hi;
    reg [31:0] r_d2h_local;
    reg [31:0] r_d2h_len;
    reg [31:0] r_debug0;
    reg [31:0] r_debug1;
    
    // =========================================================================
    // Control register bits
    // =========================================================================
    // r_ctrl[0] - soft reset
    // r_ctrl[1] - inference start (auto-clear)
    // r_ctrl[2] - frame start (auto-clear)
    // r_ctrl[3] - seq start (auto-clear)
    // r_ctrl[4] - generate start (auto-clear)
    
    assign soft_reset      = r_ctrl[0];
    assign inference_start = r_ctrl[1];
    assign frame_start     = r_ctrl[2];
    assign seq_start       = r_ctrl[3];
    assign generate_start  = r_ctrl[4];
    assign max_tokens      = r_max_tokens;
    
    // DMA control bits
    // r_dma_ctrl[0] - DMA enable
    // r_dma_ctrl[1] - H2D start (auto-clear)
    // r_dma_ctrl[2] - D2H start (auto-clear)
    
    assign dma_enable     = r_dma_ctrl[0];
    assign dma_h2d_start  = r_dma_ctrl[1];
    assign dma_d2h_start  = r_dma_ctrl[2];
    
    assign dma_h2d_host_addr  = {r_h2d_addr_hi, r_h2d_addr_lo};
    assign dma_h2d_local_addr = r_h2d_local;
    assign dma_h2d_length     = r_h2d_len[23:0];
    assign dma_d2h_host_addr  = {r_d2h_addr_hi, r_d2h_addr_lo};
    assign dma_d2h_local_addr = r_d2h_local;
    assign dma_d2h_length     = r_d2h_len[23:0];

    
    // =========================================================================
    // Write logic
    // =========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            r_ctrl        <= 32'b0;
            r_int_enable  <= 32'b0;
            r_img_addr_lo <= 32'b0;
            r_img_addr_hi <= 32'b0;
            r_img_dim     <= 32'h0180_0180;  // Default 384x384
            r_out_addr_lo <= 32'b0;
            r_out_addr_hi <= 32'b0;
            r_out_len     <= 32'b0;
            r_max_tokens  <= 32'd256;
            r_dma_ctrl    <= 32'b0;
            r_h2d_addr_lo <= 32'b0;
            r_h2d_addr_hi <= 32'b0;
            r_h2d_local   <= 32'b0;
            r_h2d_len     <= 32'b0;
            r_d2h_addr_lo <= 32'b0;
            r_d2h_addr_hi <= 32'b0;
            r_d2h_local   <= 32'b0;
            r_d2h_len     <= 32'b0;
            r_debug0      <= 32'b0;
            r_debug1      <= 32'b0;
        end else begin
            // Auto-clear pulse bits
            r_ctrl[4:1]     <= 4'b0;
            r_dma_ctrl[2:1] <= 2'b0;
            
            if (reg_wr_en) begin
                case (reg_addr)
                    ADDR_CTRL:        r_ctrl        <= reg_wr_data;
                    ADDR_INT_ENABLE:  r_int_enable  <= reg_wr_data;
                    ADDR_IMG_ADDR_LO: r_img_addr_lo <= reg_wr_data;
                    ADDR_IMG_ADDR_HI: r_img_addr_hi <= reg_wr_data;
                    ADDR_IMG_DIM:     r_img_dim     <= reg_wr_data;
                    ADDR_OUT_ADDR_LO: r_out_addr_lo <= reg_wr_data;
                    ADDR_OUT_ADDR_HI: r_out_addr_hi <= reg_wr_data;
                    ADDR_OUT_LEN:     r_out_len     <= reg_wr_data;
                    ADDR_MAX_TOKENS:  r_max_tokens  <= reg_wr_data;
                    ADDR_DMA_CTRL:    r_dma_ctrl    <= reg_wr_data;
                    ADDR_H2D_ADDR_LO: r_h2d_addr_lo <= reg_wr_data;
                    ADDR_H2D_ADDR_HI: r_h2d_addr_hi <= reg_wr_data;
                    ADDR_H2D_LOCAL:   r_h2d_local   <= reg_wr_data;
                    ADDR_H2D_LEN:     r_h2d_len     <= reg_wr_data;
                    ADDR_D2H_ADDR_LO: r_d2h_addr_lo <= reg_wr_data;
                    ADDR_D2H_ADDR_HI: r_d2h_addr_hi <= reg_wr_data;
                    ADDR_D2H_LOCAL:   r_d2h_local   <= reg_wr_data;
                    ADDR_D2H_LEN:     r_d2h_len     <= reg_wr_data;
                    ADDR_DEBUG0:      r_debug0      <= reg_wr_data;
                    ADDR_DEBUG1:      r_debug1      <= reg_wr_data;
                    default: ;
                endcase
            end
        end
    end
    
    // =========================================================================
    // Read logic
    // =========================================================================
    
    wire [31:0] status_reg;
    assign status_reg = {16'b0,
                         4'b0,
                         error_flag,
                         dma_d2h_busy,
                         dma_h2d_busy,
                         1'b0,
                         4'b0,
                         llm_busy,
                         vision_busy,
                         1'b0,
                         1'b1};  // Ready bit
    
    wire [31:0] dma_status_reg;
    assign dma_status_reg = {28'b0,
                             dma_d2h_done,
                             dma_h2d_done,
                             dma_d2h_busy,
                             dma_h2d_busy};
    
    always @(posedge clk) begin
        if (!rst_n) begin
            reg_rd_data  <= 32'b0;
            reg_rd_valid <= 1'b0;
        end else begin
            reg_rd_valid <= reg_rd_en;
            
            if (reg_rd_en) begin
                case (reg_addr)
                    ADDR_CTRL:        reg_rd_data <= r_ctrl;
                    ADDR_STATUS:      reg_rd_data <= status_reg;
                    ADDR_VERSION:     reg_rd_data <= VERSION;
                    ADDR_INT_ENABLE:  reg_rd_data <= r_int_enable;
                    ADDR_INT_STATUS:  reg_rd_data <= r_int_status;
                    ADDR_IMG_ADDR_LO: reg_rd_data <= r_img_addr_lo;
                    ADDR_IMG_ADDR_HI: reg_rd_data <= r_img_addr_hi;
                    ADDR_IMG_DIM:     reg_rd_data <= r_img_dim;
                    ADDR_OUT_ADDR_LO: reg_rd_data <= r_out_addr_lo;
                    ADDR_OUT_ADDR_HI: reg_rd_data <= r_out_addr_hi;
                    ADDR_OUT_LEN:     reg_rd_data <= r_out_len;
                    ADDR_MAX_TOKENS:  reg_rd_data <= r_max_tokens;
                    ADDR_DMA_CTRL:    reg_rd_data <= r_dma_ctrl;
                    ADDR_DMA_STATUS:  reg_rd_data <= dma_status_reg;
                    ADDR_H2D_ADDR_LO: reg_rd_data <= r_h2d_addr_lo;
                    ADDR_H2D_ADDR_HI: reg_rd_data <= r_h2d_addr_hi;
                    ADDR_H2D_LOCAL:   reg_rd_data <= r_h2d_local;
                    ADDR_H2D_LEN:     reg_rd_data <= r_h2d_len;
                    ADDR_D2H_ADDR_LO: reg_rd_data <= r_d2h_addr_lo;
                    ADDR_D2H_ADDR_HI: reg_rd_data <= r_d2h_addr_hi;
                    ADDR_D2H_LOCAL:   reg_rd_data <= r_d2h_local;
                    ADDR_D2H_LEN:     reg_rd_data <= r_d2h_len;
                    ADDR_DEBUG0:      reg_rd_data <= r_debug0;
                    ADDR_DEBUG1:      reg_rd_data <= r_debug1;
                    ADDR_TOK_COUNT:   reg_rd_data <= {16'b0, tokens_generated};
                    default:          reg_rd_data <= 32'hDEAD_BEEF;
                endcase
            end
        end
    end

    
    // =========================================================================
    // Interrupt logic
    // =========================================================================
    
    // Interrupt sources:
    // [0] - Inference complete
    // [1] - DMA H2D complete
    // [2] - DMA D2H complete
    // [3] - Error
    
    wire [3:0] int_sources;
    assign int_sources = {error_flag, dma_d2h_done, dma_h2d_done, ~(vision_busy | llm_busy)};
    
    always @(posedge clk) begin
        if (!rst_n) begin
            r_int_status <= 32'b0;
        end else begin
            // Set interrupt status bits
            r_int_status <= r_int_status | {28'b0, int_sources};
            
            // Clear on write
            if (reg_wr_en && reg_addr == ADDR_INT_STATUS) begin
                r_int_status <= r_int_status & ~reg_wr_data;
            end
            
            // Clear on acknowledge
            if (interrupt_ack) begin
                r_int_status <= 32'b0;
            end
        end
    end
    
    assign interrupt_out = |(r_int_status & r_int_enable);

endmodule
