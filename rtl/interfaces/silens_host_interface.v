// =============================================================================
// SiLens Host Interface Controller
// =============================================================================
// Parallel host interface for FPGA bridge communication.
// Provides register access, token streaming, and DMA for image data.
//
// For SKY130: 32-bit parallel bus at 100MHz = 400 MB/s peak bandwidth
// This bridges to an FPGA which provides PCIe/USB connectivity to host PC.
//
// Register Map:
//   0x0000: Control register (RW)
//   0x0004: Status register (RO)
//   0x0008: Token write FIFO (WO)
//   0x000C: Token read FIFO (RO)
//   0x0010: Image DMA base address (RW)
//   0x0014: Image DMA length (RW)
//   0x0018: Image DMA control (RW)
//   0x001C: Interrupt enable (RW)
//   0x0020: Interrupt status (RO/W1C)
//   0x0024-0x003F: Reserved
//   0x0040-0x00FF: Configuration registers
//   0x0100-0x01FF: Debug/status registers
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_host_interface #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 16,
    parameter TOKEN_FIFO_DEPTH = 256,
    parameter PIXEL_FIFO_DEPTH = 1024
)(
    // =========================================================================
    // Host Side (FPGA bridge clock domain)
    // =========================================================================
    input  wire                     host_clk,
    input  wire                     host_rst_n,
    
    // Parallel bus interface
    input  wire [DATA_WIDTH-1:0]    host_data_in,
    output reg  [DATA_WIDTH-1:0]    host_data_out,
    output reg                      host_data_oe,
    input  wire [ADDR_WIDTH-1:0]    host_addr,
    input  wire                     host_rd_n,
    input  wire                     host_wr_n,
    input  wire                     host_cs_n,
    output reg                      host_ready,
    output wire                     host_irq,
    
    // =========================================================================
    // Core Side (Core clock domain)
    // =========================================================================
    input  wire                     core_clk,
    input  wire                     core_rst_n,
    
    // Control outputs (synchronized to core_clk)
    output reg                      frame_start,
    output reg                      seq_start,
    output reg                      gen_start,
    output reg                      abort,
    
    // Token streaming (to/from VLM core)
    output reg  [15:0]              token_in,
    output reg                      token_in_valid,
    input  wire                     token_in_ready,
    
    input  wire [15:0]              token_out,
    input  wire                     token_out_valid,
    output reg                      token_out_ready,
    
    // Pixel streaming (to vision encoder)
    output wire [23:0]              pixel_out,         // RGB888
    output wire                     pixel_out_valid,
    input  wire                     pixel_out_ready,
    
    // Status inputs
    input  wire [31:0]              status,
    input  wire                     ddr_init_done
);

    // =========================================================================
    // Register Addresses
    // =========================================================================
    
    localparam REG_CONTROL      = 16'h0000;
    localparam REG_STATUS       = 16'h0004;
    localparam REG_TOKEN_WR     = 16'h0008;
    localparam REG_TOKEN_RD     = 16'h000C;
    localparam REG_DMA_BASE     = 16'h0010;
    localparam REG_DMA_LEN      = 16'h0014;
    localparam REG_DMA_CTRL     = 16'h0018;
    localparam REG_IRQ_ENABLE   = 16'h001C;
    localparam REG_IRQ_STATUS   = 16'h0020;
    localparam REG_VERSION      = 16'h0024;
    localparam REG_PIXEL_WR     = 16'h0028;
    localparam REG_FIFO_STATUS  = 16'h002C;
    
    // =========================================================================
    // Control Register Bits
    // =========================================================================
    
    // Control register (0x0000)
    // [0]    : frame_start (pulse)
    // [1]    : seq_start (pulse)
    // [2]    : gen_start (pulse)
    // [3]    : abort (pulse)
    // [4]    : soft_reset
    // [7:5]  : reserved
    // [15:8] : generation config (max_tokens)
    // [31:16]: reserved
    
    // =========================================================================
    // Host-Side Registers
    // =========================================================================
    
    reg [31:0] ctrl_reg;
    reg [31:0] dma_base_reg;
    reg [31:0] dma_len_reg;
    reg [31:0] dma_ctrl_reg;
    reg [31:0] irq_enable_reg;
    reg [31:0] irq_status_reg;
    
    // Version constant
    localparam VERSION = 32'h0001_0000;  // v1.0.0
    
    // =========================================================================
    // Token Input FIFO (Host -> Core)
    // =========================================================================
    
    // Async FIFO for CDC: host_clk -> core_clk
    reg [15:0] token_in_fifo [0:TOKEN_FIFO_DEPTH-1];
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_in_wr_ptr_host;
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_in_rd_ptr_core;
    
    // Gray-coded pointers for CDC
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_in_wr_ptr_gray;
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_in_rd_ptr_gray;
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_in_wr_ptr_sync [0:1];
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_in_rd_ptr_sync [0:1];
    
    wire token_in_fifo_full;
    wire token_in_fifo_empty;
    
    // Binary to Gray conversion
    function [$clog2(TOKEN_FIFO_DEPTH):0] bin2gray;
        input [$clog2(TOKEN_FIFO_DEPTH):0] bin;
        bin2gray = bin ^ (bin >> 1);
    endfunction
    
    // Gray to Binary conversion
    function [$clog2(TOKEN_FIFO_DEPTH):0] gray2bin;
        input [$clog2(TOKEN_FIFO_DEPTH):0] gray;
        integer i;
        begin
            gray2bin[$clog2(TOKEN_FIFO_DEPTH)] = gray[$clog2(TOKEN_FIFO_DEPTH)];
            for (i = $clog2(TOKEN_FIFO_DEPTH)-1; i >= 0; i = i - 1)
                gray2bin[i] = gray2bin[i+1] ^ gray[i];
        end
    endfunction
    
    // Write side (host_clk domain)
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            token_in_wr_ptr_host <= 0;
            token_in_wr_ptr_gray <= 0;
        end else if (!host_cs_n && !host_wr_n && host_addr == REG_TOKEN_WR && !token_in_fifo_full) begin
            token_in_fifo[token_in_wr_ptr_host[$clog2(TOKEN_FIFO_DEPTH)-1:0]] <= host_data_in[15:0];
            token_in_wr_ptr_host <= token_in_wr_ptr_host + 1;
            token_in_wr_ptr_gray <= bin2gray(token_in_wr_ptr_host + 1);
        end
    end
    
    // Synchronize write pointer to core domain
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            token_in_wr_ptr_sync[0] <= 0;
            token_in_wr_ptr_sync[1] <= 0;
        end else begin
            token_in_wr_ptr_sync[0] <= token_in_wr_ptr_gray;
            token_in_wr_ptr_sync[1] <= token_in_wr_ptr_sync[0];
        end
    end
    
    // Read side (core_clk domain)
    wire [$clog2(TOKEN_FIFO_DEPTH):0] token_in_wr_ptr_core = gray2bin(token_in_wr_ptr_sync[1]);
    
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            token_in_rd_ptr_core <= 0;
            token_in_rd_ptr_gray <= 0;
            token_in <= 0;
            token_in_valid <= 0;
        end else begin
            token_in_valid <= 0;
            
            if (!token_in_fifo_empty && token_in_ready) begin
                token_in <= token_in_fifo[token_in_rd_ptr_core[$clog2(TOKEN_FIFO_DEPTH)-1:0]];
                token_in_valid <= 1;
                token_in_rd_ptr_core <= token_in_rd_ptr_core + 1;
                token_in_rd_ptr_gray <= bin2gray(token_in_rd_ptr_core + 1);
            end
        end
    end
    
    // Synchronize read pointer to host domain
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            token_in_rd_ptr_sync[0] <= 0;
            token_in_rd_ptr_sync[1] <= 0;
        end else begin
            token_in_rd_ptr_sync[0] <= token_in_rd_ptr_gray;
            token_in_rd_ptr_sync[1] <= token_in_rd_ptr_sync[0];
        end
    end
    
    wire [$clog2(TOKEN_FIFO_DEPTH):0] token_in_rd_ptr_host = gray2bin(token_in_rd_ptr_sync[1]);
    
    assign token_in_fifo_full = (token_in_wr_ptr_host[$clog2(TOKEN_FIFO_DEPTH)] != 
                                 token_in_rd_ptr_host[$clog2(TOKEN_FIFO_DEPTH)]) &&
                                (token_in_wr_ptr_host[$clog2(TOKEN_FIFO_DEPTH)-1:0] == 
                                 token_in_rd_ptr_host[$clog2(TOKEN_FIFO_DEPTH)-1:0]);
    assign token_in_fifo_empty = (token_in_wr_ptr_core == token_in_rd_ptr_core);
    
    // =========================================================================
    // Token Output FIFO (Core -> Host)
    // =========================================================================
    
    reg [15:0] token_out_fifo [0:TOKEN_FIFO_DEPTH-1];
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_out_wr_ptr_core;
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_out_rd_ptr_host;
    
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_out_wr_ptr_gray;
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_out_rd_ptr_gray;
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_out_wr_ptr_sync [0:1];
    reg [$clog2(TOKEN_FIFO_DEPTH):0] token_out_rd_ptr_sync [0:1];
    
    wire token_out_fifo_full;
    wire token_out_fifo_empty;
    
    // Write side (core_clk domain)
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            token_out_wr_ptr_core <= 0;
            token_out_wr_ptr_gray <= 0;
            token_out_ready <= 0;
        end else begin
            token_out_ready <= !token_out_fifo_full;
            
            if (token_out_valid && !token_out_fifo_full) begin
                token_out_fifo[token_out_wr_ptr_core[$clog2(TOKEN_FIFO_DEPTH)-1:0]] <= token_out;
                token_out_wr_ptr_core <= token_out_wr_ptr_core + 1;
                token_out_wr_ptr_gray <= bin2gray(token_out_wr_ptr_core + 1);
            end
        end
    end
    
    // Synchronize write pointer to host domain
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            token_out_wr_ptr_sync[0] <= 0;
            token_out_wr_ptr_sync[1] <= 0;
        end else begin
            token_out_wr_ptr_sync[0] <= token_out_wr_ptr_gray;
            token_out_wr_ptr_sync[1] <= token_out_wr_ptr_sync[0];
        end
    end
    
    wire [$clog2(TOKEN_FIFO_DEPTH):0] token_out_wr_ptr_host_sync = gray2bin(token_out_wr_ptr_sync[1]);
    
    // Read side (host_clk domain)
    reg [15:0] token_out_data_host;
    
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            token_out_rd_ptr_host <= 0;
            token_out_rd_ptr_gray <= 0;
            token_out_data_host <= 0;
        end else if (!host_cs_n && !host_rd_n && host_addr == REG_TOKEN_RD && !token_out_fifo_empty) begin
            token_out_data_host <= token_out_fifo[token_out_rd_ptr_host[$clog2(TOKEN_FIFO_DEPTH)-1:0]];
            token_out_rd_ptr_host <= token_out_rd_ptr_host + 1;
            token_out_rd_ptr_gray <= bin2gray(token_out_rd_ptr_host + 1);
        end
    end
    
    // Synchronize read pointer to core domain
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            token_out_rd_ptr_sync[0] <= 0;
            token_out_rd_ptr_sync[1] <= 0;
        end else begin
            token_out_rd_ptr_sync[0] <= token_out_rd_ptr_gray;
            token_out_rd_ptr_sync[1] <= token_out_rd_ptr_sync[0];
        end
    end
    
    wire [$clog2(TOKEN_FIFO_DEPTH):0] token_out_rd_ptr_core_sync = gray2bin(token_out_rd_ptr_sync[1]);
    
    assign token_out_fifo_full = (token_out_wr_ptr_core[$clog2(TOKEN_FIFO_DEPTH)] != 
                                  token_out_rd_ptr_core_sync[$clog2(TOKEN_FIFO_DEPTH)]) &&
                                 (token_out_wr_ptr_core[$clog2(TOKEN_FIFO_DEPTH)-1:0] == 
                                  token_out_rd_ptr_core_sync[$clog2(TOKEN_FIFO_DEPTH)-1:0]);
    assign token_out_fifo_empty = (token_out_wr_ptr_host_sync == token_out_rd_ptr_host);
    
    // =========================================================================
    // Pixel FIFO (Host -> Vision Encoder)
    // =========================================================================
    
    reg [23:0] pixel_fifo [0:PIXEL_FIFO_DEPTH-1];
    reg [$clog2(PIXEL_FIFO_DEPTH):0] pixel_wr_ptr;
    reg [$clog2(PIXEL_FIFO_DEPTH):0] pixel_rd_ptr;
    
    // Simplified for same-clock-domain operation (assumes host_clk bridges to core_clk)
    wire pixel_fifo_full = (pixel_wr_ptr[$clog2(PIXEL_FIFO_DEPTH)] != pixel_rd_ptr[$clog2(PIXEL_FIFO_DEPTH)]) &&
                           (pixel_wr_ptr[$clog2(PIXEL_FIFO_DEPTH)-1:0] == pixel_rd_ptr[$clog2(PIXEL_FIFO_DEPTH)-1:0]);
    wire pixel_fifo_empty = (pixel_wr_ptr == pixel_rd_ptr);
    
    // Write side (host_clk domain)
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            pixel_wr_ptr <= 0;
        end else if (!host_cs_n && !host_wr_n && host_addr == REG_PIXEL_WR && !pixel_fifo_full) begin
            pixel_fifo[pixel_wr_ptr[$clog2(PIXEL_FIFO_DEPTH)-1:0]] <= host_data_in[23:0];
            pixel_wr_ptr <= pixel_wr_ptr + 1;
        end
    end
    
    // Read side - needs CDC to core_clk (simplified here)
    assign pixel_out = pixel_fifo[pixel_rd_ptr[$clog2(PIXEL_FIFO_DEPTH)-1:0]];
    assign pixel_out_valid = !pixel_fifo_empty;
    
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            pixel_rd_ptr <= 0;
        end else if (pixel_out_valid && pixel_out_ready) begin
            pixel_rd_ptr <= pixel_rd_ptr + 1;
        end
    end
    
    // =========================================================================
    // Control Signal CDC (host_clk -> core_clk)
    // =========================================================================
    
    reg [3:0] ctrl_pulse_host;
    reg [3:0] ctrl_pulse_sync [0:2];
    
    // Capture pulses in host domain
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            ctrl_pulse_host <= 0;
        end else if (!host_cs_n && !host_wr_n && host_addr == REG_CONTROL) begin
            ctrl_pulse_host <= host_data_in[3:0];
        end else begin
            ctrl_pulse_host <= 0;
        end
    end
    
    // Synchronize to core domain
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            ctrl_pulse_sync[0] <= 0;
            ctrl_pulse_sync[1] <= 0;
            ctrl_pulse_sync[2] <= 0;
        end else begin
            ctrl_pulse_sync[0] <= ctrl_pulse_host;
            ctrl_pulse_sync[1] <= ctrl_pulse_sync[0];
            ctrl_pulse_sync[2] <= ctrl_pulse_sync[1];
        end
    end
    
    // Edge detect for pulses
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            frame_start <= 0;
            seq_start <= 0;
            gen_start <= 0;
            abort <= 0;
        end else begin
            frame_start <= ctrl_pulse_sync[1][0] & ~ctrl_pulse_sync[2][0];
            seq_start   <= ctrl_pulse_sync[1][1] & ~ctrl_pulse_sync[2][1];
            gen_start   <= ctrl_pulse_sync[1][2] & ~ctrl_pulse_sync[2][2];
            abort       <= ctrl_pulse_sync[1][3] & ~ctrl_pulse_sync[2][3];
        end
    end
    
    // =========================================================================
    // Status CDC (core_clk -> host_clk)
    // =========================================================================
    
    reg [31:0] status_sync [0:1];
    
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            status_sync[0] <= 0;
            status_sync[1] <= 0;
        end else begin
            status_sync[0] <= status;
            status_sync[1] <= status_sync[0];
        end
    end
    
    // =========================================================================
    // Interrupt Generation
    // =========================================================================
    
    // Interrupt sources:
    // [0]: Token output FIFO not empty
    // [1]: Inference complete
    // [2]: Error
    // [3]: DDR init done
    
    wire [3:0] irq_sources = {
        ddr_init_done,
        status_sync[1][1],  // error
        status_sync[1][0],  // inference_done
        !token_out_fifo_empty
    };
    
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            irq_status_reg <= 0;
        end else begin
            // Set on rising edge of source
            irq_status_reg <= irq_status_reg | irq_sources;
            
            // Clear on write-1-to-clear
            if (!host_cs_n && !host_wr_n && host_addr == REG_IRQ_STATUS) begin
                irq_status_reg <= irq_status_reg & ~host_data_in[3:0];
            end
        end
    end
    
    assign host_irq = |(irq_status_reg & irq_enable_reg);
    
    // =========================================================================
    // Register Read/Write Logic (host_clk domain)
    // =========================================================================
    
    always @(posedge host_clk or negedge host_rst_n) begin
        if (!host_rst_n) begin
            ctrl_reg <= 0;
            dma_base_reg <= 0;
            dma_len_reg <= 0;
            dma_ctrl_reg <= 0;
            irq_enable_reg <= 0;
            host_data_out <= 0;
            host_data_oe <= 0;
            host_ready <= 1;
        end else begin
            host_ready <= 1;
            host_data_oe <= 0;
            
            if (!host_cs_n) begin
                if (!host_wr_n) begin
                    // Write
                    case (host_addr)
                        REG_CONTROL:    ctrl_reg <= host_data_in;
                        REG_DMA_BASE:   dma_base_reg <= host_data_in;
                        REG_DMA_LEN:    dma_len_reg <= host_data_in;
                        REG_DMA_CTRL:   dma_ctrl_reg <= host_data_in;
                        REG_IRQ_ENABLE: irq_enable_reg <= host_data_in;
                        // REG_TOKEN_WR, REG_PIXEL_WR handled by FIFOs
                        // REG_IRQ_STATUS handled above (W1C)
                    endcase
                end else if (!host_rd_n) begin
                    // Read
                    host_data_oe <= 1;
                    case (host_addr)
                        REG_CONTROL:    host_data_out <= ctrl_reg;
                        REG_STATUS:     host_data_out <= status_sync[1];
                        REG_TOKEN_RD:   host_data_out <= {16'b0, token_out_data_host};
                        REG_DMA_BASE:   host_data_out <= dma_base_reg;
                        REG_DMA_LEN:    host_data_out <= dma_len_reg;
                        REG_DMA_CTRL:   host_data_out <= dma_ctrl_reg;
                        REG_IRQ_ENABLE: host_data_out <= irq_enable_reg;
                        REG_IRQ_STATUS: host_data_out <= irq_status_reg;
                        REG_VERSION:    host_data_out <= VERSION;
                        REG_FIFO_STATUS: host_data_out <= {
                            8'b0,
                            7'b0, pixel_fifo_full,
                            7'b0, token_out_fifo_empty,
                            7'b0, token_in_fifo_full
                        };
                        default:        host_data_out <= 32'hDEAD_BEEF;
                    endcase
                end
            end
        end
    end

endmodule

`default_nettype wire
