// =============================================================================
// SiLens - Double-Buffered Activation Storage
// =============================================================================
// Provides double-buffered activation storage for pipelined inference.
// While one buffer is being read for processing, the other can be written
// with new data.
//
// Features:
//   - Double buffering for concurrent read/write
//   - Configurable depth and width
//   - Bank switching for seamless pipeline operation
//   - BRAM inference friendly
//
// License: Apache 2.0
// =============================================================================

module activation_buffer #(
    parameter DEPTH      = 576,                     // Number of tokens/vectors
    parameter WIDTH      = 768,                     // Dimension per token
    parameter ACT_WIDTH  = 8,                       // Bits per element
    parameter ADDR_WIDTH = $clog2(DEPTH)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Write interface (producer side)
    input  wire [WIDTH*ACT_WIDTH-1:0]   wr_data,
    input  wire [ADDR_WIDTH-1:0]        wr_addr,
    input  wire                         wr_en,
    output wire                         wr_ready,
    
    // Read interface (consumer side)
    input  wire [ADDR_WIDTH-1:0]        rd_addr,
    input  wire                         rd_en,
    output reg  [WIDTH*ACT_WIDTH-1:0]   rd_data,
    output reg                          rd_valid,
    
    // Buffer control
    input  wire                         swap_buffers,   // Swap read/write banks
    output wire                         buffer_id,      // Current write buffer ID
    
    // Status
    output wire                         full,
    output wire                         empty
);

    // =========================================================================
    // Buffer selection
    // =========================================================================
    
    reg active_write_bank;  // 0 or 1
    
    always @(posedge clk) begin
        if (!rst_n) begin
            active_write_bank <= 1'b0;
        end else if (swap_buffers) begin
            active_write_bank <= ~active_write_bank;
        end
    end
    
    assign buffer_id = active_write_bank;
    
    // =========================================================================
    // Write tracking
    // =========================================================================
    
    reg [ADDR_WIDTH:0] wr_count [0:1];
    
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_count[0] <= 0;
            wr_count[1] <= 0;
        end else begin
            if (swap_buffers) begin
                // Clear the new write buffer count
                wr_count[~active_write_bank] <= 0;
            end else if (wr_en && wr_ready) begin
                wr_count[active_write_bank] <= wr_count[active_write_bank] + 1;
            end
        end
    end
    
    assign wr_ready = (wr_count[active_write_bank] < DEPTH);
    assign full     = (wr_count[active_write_bank] >= DEPTH);
    assign empty    = (wr_count[~active_write_bank] == 0);

    
    // =========================================================================
    // Memory banks (BRAM inference)
    // =========================================================================
    
    // Split into smaller chunks for better BRAM inference
    localparam CHUNK_WIDTH = 72;  // Match BRAM primitive width
    localparam NUM_CHUNKS = (WIDTH * ACT_WIDTH + CHUNK_WIDTH - 1) / CHUNK_WIDTH;
    localparam PADDED_WIDTH = NUM_CHUNKS * CHUNK_WIDTH;
    
    // Bank 0
    (* ram_style = "block" *)
    reg [PADDED_WIDTH-1:0] bank0 [0:DEPTH-1];
    
    // Bank 1
    (* ram_style = "block" *)
    reg [PADDED_WIDTH-1:0] bank1 [0:DEPTH-1];
    
    // =========================================================================
    // Write logic
    // =========================================================================
    
    wire [PADDED_WIDTH-1:0] wr_data_padded;
    assign wr_data_padded = {{(PADDED_WIDTH - WIDTH*ACT_WIDTH){1'b0}}, wr_data};
    
    always @(posedge clk) begin
        if (wr_en && wr_ready) begin
            if (active_write_bank == 1'b0) begin
                bank0[wr_addr] <= wr_data_padded;
            end else begin
                bank1[wr_addr] <= wr_data_padded;
            end
        end
    end
    
    // =========================================================================
    // Read logic (from opposite bank)
    // =========================================================================
    
    reg [PADDED_WIDTH-1:0] rd_data_padded;
    reg rd_en_r;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_data_padded <= 0;
            rd_en_r <= 1'b0;
        end else begin
            rd_en_r <= rd_en;
            
            if (rd_en) begin
                // Read from opposite bank
                if (active_write_bank == 1'b0) begin
                    rd_data_padded <= bank1[rd_addr];
                end else begin
                    rd_data_padded <= bank0[rd_addr];
                end
            end
        end
    end
    
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_data <= 0;
            rd_valid <= 1'b0;
        end else begin
            rd_data <= rd_data_padded[WIDTH*ACT_WIDTH-1:0];
            rd_valid <= rd_en_r;
        end
    end

endmodule
