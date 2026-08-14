// =============================================================================
// SiLens - Key-Value Cache for Autoregressive Decoding
// =============================================================================
// Stores key and value vectors for autoregressive token generation.
// Supports efficient append and sequential read operations.
//
// Architecture:
//   - Separate K and V memories for parallel access
//   - Circular buffer for context length management
//   - Per-layer storage (instantiate one per decoder layer)
//
// Memory requirements per layer (576 dim, 9 heads, 8K context):
//   K: 8192 x 576 x 8 bits = 37.75 MB
//   V: 8192 x 576 x 8 bits = 37.75 MB
//   Total per layer: ~75.5 MB
//   30 layers: ~2.27 GB (requires external memory for full context)
//
// This module handles a smaller on-chip working set.
//
// License: Apache 2.0
// =============================================================================

module kv_cache #(
    parameter DIM         = 576,                    // Total KV dimension
    parameter NUM_HEADS   = 9,                      // Number of attention heads
    parameter HEAD_DIM    = 64,                     // DIM / NUM_HEADS
    parameter MAX_SEQ_LEN = 2048,                   // On-chip cache depth
    parameter ACT_WIDTH   = 8,                      // Activation width
    parameter ADDR_WIDTH  = $clog2(MAX_SEQ_LEN)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Cache control
    input  wire                         cache_clear,        // Clear cache (new sequence)
    output wire [ADDR_WIDTH:0]          cache_len,          // Current cached length
    
    // Write interface (append new KV pair)
    input  wire [DIM*ACT_WIDTH-1:0]     k_in,               // New key vector
    input  wire [DIM*ACT_WIDTH-1:0]     v_in,               // New value vector
    input  wire                         kv_write_en,
    output wire                         kv_write_ready,
    
    // Read interface (for attention computation)
    input  wire [ADDR_WIDTH-1:0]        rd_pos,             // Position to read
    input  wire [$clog2(NUM_HEADS)-1:0] rd_head,            // Head to read
    input  wire                         rd_en,
    output reg  [HEAD_DIM*ACT_WIDTH-1:0] k_out,             // Key for position
    output reg  [HEAD_DIM*ACT_WIDTH-1:0] v_out,             // Value for position
    output reg                          kv_read_valid,
    
    // Status
    output wire                         cache_full
);

    // =========================================================================
    // Cache position tracking
    // =========================================================================
    
    reg [ADDR_WIDTH:0] write_ptr;
    
    always @(posedge clk) begin
        if (!rst_n || cache_clear) begin
            write_ptr <= 0;
        end else if (kv_write_en && kv_write_ready) begin
            write_ptr <= write_ptr + 1;
        end
    end
    
    assign cache_len      = write_ptr;
    assign cache_full     = (write_ptr >= MAX_SEQ_LEN);
    assign kv_write_ready = ~cache_full;

    
    // =========================================================================
    // Key cache memory (BRAM inference)
    // =========================================================================
    // Organized as: MAX_SEQ_LEN x NUM_HEADS memories, each HEAD_DIM wide
    
    genvar h;
    generate
        for (h = 0; h < NUM_HEADS; h = h + 1) begin : gen_head_cache
            
            // Key memory for this head
            (* ram_style = "block" *)
            reg [HEAD_DIM*ACT_WIDTH-1:0] k_mem [0:MAX_SEQ_LEN-1];
            
            // Value memory for this head
            (* ram_style = "block" *)
            reg [HEAD_DIM*ACT_WIDTH-1:0] v_mem [0:MAX_SEQ_LEN-1];
            
            // Write logic
            always @(posedge clk) begin
                if (kv_write_en && kv_write_ready) begin
                    k_mem[write_ptr[ADDR_WIDTH-1:0]] <= 
                        k_in[h*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH];
                    v_mem[write_ptr[ADDR_WIDTH-1:0]] <= 
                        v_in[h*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH];
                end
            end
            
        end
    endgenerate
    
    // =========================================================================
    // Read logic with head selection
    // =========================================================================
    
    reg [ADDR_WIDTH-1:0] rd_pos_r;
    reg [$clog2(NUM_HEADS)-1:0] rd_head_r;
    reg rd_en_r;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_pos_r  <= 0;
            rd_head_r <= 0;
            rd_en_r   <= 1'b0;
        end else begin
            rd_pos_r  <= rd_pos;
            rd_head_r <= rd_head;
            rd_en_r   <= rd_en;
        end
    end
    
    // Read mux (selects output from appropriate head)
    integer read_h;
    always @(posedge clk) begin
        if (!rst_n) begin
            k_out <= 0;
            v_out <= 0;
            kv_read_valid <= 1'b0;
        end else begin
            kv_read_valid <= rd_en_r;
            
            // Default
            k_out <= 0;
            v_out <= 0;
            
            // Select from appropriate head memory
            for (read_h = 0; read_h < NUM_HEADS; read_h = read_h + 1) begin
                if (rd_head_r == read_h) begin
                    k_out <= gen_head_cache[read_h].k_mem[rd_pos_r];
                    v_out <= gen_head_cache[read_h].v_mem[rd_pos_r];
                end
            end
        end
    end

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module kv_cache_tb;
    parameter DIM = 64;
    parameter NUM_HEADS = 4;
    parameter HEAD_DIM = 16;
    parameter MAX_SEQ_LEN = 32;
    parameter ACT_WIDTH = 8;
    
    reg clk, rst_n;
    reg cache_clear;
    wire [$clog2(MAX_SEQ_LEN):0] cache_len;
    reg [DIM*ACT_WIDTH-1:0] k_in, v_in;
    reg kv_write_en;
    wire kv_write_ready;
    reg [$clog2(MAX_SEQ_LEN)-1:0] rd_pos;
    reg [$clog2(NUM_HEADS)-1:0] rd_head;
    reg rd_en;
    wire [HEAD_DIM*ACT_WIDTH-1:0] k_out, v_out;
    wire kv_read_valid;
    wire cache_full;
    
    kv_cache #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .ACT_WIDTH(ACT_WIDTH)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i;
    
    initial begin
        $display("KV Cache Testbench");
        
        clk = 0; rst_n = 0;
        cache_clear = 0;
        k_in = 0; v_in = 0;
        kv_write_en = 0;
        rd_pos = 0; rd_head = 0; rd_en = 0;
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Write some KV pairs
        for (i = 0; i < 10; i = i + 1) begin
            k_in = {DIM{1'b0}} | i[7:0];  // Fill lower bits with i
            v_in = {DIM{1'b0}} | ((i + 128) & 8'hFF);  // Fill lower bits with i+128
            kv_write_en = 1;
            @(posedge clk);
        end
        kv_write_en = 0;
        
        $display("Cache length: %0d", cache_len);
        
        // Read back
        for (i = 0; i < 10; i = i + 1) begin
            rd_pos = i;
            rd_head = i % NUM_HEADS;
            rd_en = 1;
            @(posedge clk);
            @(posedge clk);
            $display("Pos %0d Head %0d: K=%h V=%h", i, rd_head, k_out[7:0], v_out[7:0]);
        end
        
        $display("Testbench complete");
        $finish;
    end
endmodule

`endif
