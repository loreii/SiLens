// =============================================================================
// SiLens - Token Embedding Lookup Table
// =============================================================================
// ROM-based token embedding table for vocabulary lookup.
// Supports both input embeddings (token -> vector) and tied output embeddings.
//
// Memory requirements:
//   49,152 tokens x 576 dimensions x 8 bits = ~27 MB
//   This requires external memory or compression for full vocabulary.
//   
// This module implements a smaller working subset or uses external memory.
//
// License: Apache 2.0
// =============================================================================

module embedding_rom #(
    parameter VOCAB_SIZE  = 49152,                  // Vocabulary size
    parameter EMBED_DIM   = 576,                    // Embedding dimension
    parameter ACT_WIDTH   = 8,                      // Bits per element
    parameter ADDR_WIDTH  = $clog2(VOCAB_SIZE),
    
    // For FPGA: use smaller on-chip subset
    parameter ONCHIP_SIZE = 4096,                   // On-chip vocabulary subset
    parameter ONCHIP_ADDR_WIDTH = $clog2(ONCHIP_SIZE)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Lookup interface
    input  wire [ADDR_WIDTH-1:0]        token_id,
    input  wire                         lookup_en,
    output reg  [EMBED_DIM*ACT_WIDTH-1:0] embedding,
    output reg                          valid_out,
    output wire                         cache_hit,
    
    // External memory interface (for out-of-cache tokens)
    output reg  [ADDR_WIDTH-1:0]        ext_addr,
    output reg                          ext_rd_en,
    input  wire [EMBED_DIM*ACT_WIDTH-1:0] ext_data,
    input  wire                         ext_valid,
    
    // ROM initialization (for loading weights)
    input  wire [ONCHIP_ADDR_WIDTH-1:0] init_addr,
    input  wire [EMBED_DIM*ACT_WIDTH-1:0] init_data,
    input  wire                         init_we
);

    // =========================================================================
    // On-chip embedding ROM (BRAM)
    // =========================================================================
    // Stores most common tokens (first ONCHIP_SIZE entries)
    
    (* ram_style = "block" *)
    reg [EMBED_DIM*ACT_WIDTH-1:0] embed_mem [0:ONCHIP_SIZE-1];
    
    // =========================================================================
    // Cache hit detection
    // =========================================================================
    
    wire is_onchip;
    assign is_onchip = (token_id < ONCHIP_SIZE);
    assign cache_hit = is_onchip;
    
    // =========================================================================
    // Lookup FSM
    // =========================================================================
    
    localparam STATE_IDLE    = 2'b00;
    localparam STATE_ONCHIP  = 2'b01;
    localparam STATE_OFFCHIP = 2'b10;
    
    reg [1:0] state;
    reg [ADDR_WIDTH-1:0] token_id_r;

    
    // =========================================================================
    // Initialization logic
    // =========================================================================
    
    always @(posedge clk) begin
        if (init_we) begin
            embed_mem[init_addr] <= init_data;
        end
    end
    
    // =========================================================================
    // Main lookup FSM
    // =========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            token_id_r <= 0;
            embedding <= 0;
            valid_out <= 1'b0;
            ext_addr <= 0;
            ext_rd_en <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    ext_rd_en <= 1'b0;
                    
                    if (lookup_en) begin
                        token_id_r <= token_id;
                        
                        if (is_onchip) begin
                            state <= STATE_ONCHIP;
                        end else begin
                            // Request from external memory
                            ext_addr <= token_id;
                            ext_rd_en <= 1'b1;
                            state <= STATE_OFFCHIP;
                        end
                    end
                end
                
                STATE_ONCHIP: begin
                    // BRAM read latency (1 cycle)
                    embedding <= embed_mem[token_id_r[ONCHIP_ADDR_WIDTH-1:0]];
                    valid_out <= 1'b1;
                    state <= STATE_IDLE;
                end
                
                STATE_OFFCHIP: begin
                    ext_rd_en <= 1'b0;
                    
                    if (ext_valid) begin
                        embedding <= ext_data;
                        valid_out <= 1'b1;
                        state <= STATE_IDLE;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module embedding_rom_tb;
    parameter VOCAB_SIZE = 256;
    parameter EMBED_DIM = 32;
    parameter ACT_WIDTH = 8;
    parameter ONCHIP_SIZE = 64;
    
    reg clk, rst_n;
    reg [$clog2(VOCAB_SIZE)-1:0] token_id;
    reg lookup_en;
    wire [EMBED_DIM*ACT_WIDTH-1:0] embedding;
    wire valid_out, cache_hit;
    wire [$clog2(VOCAB_SIZE)-1:0] ext_addr;
    wire ext_rd_en;
    reg [EMBED_DIM*ACT_WIDTH-1:0] ext_data;
    reg ext_valid;
    reg [$clog2(ONCHIP_SIZE)-1:0] init_addr;
    reg [EMBED_DIM*ACT_WIDTH-1:0] init_data;
    reg init_we;
    
    embedding_rom #(
        .VOCAB_SIZE(VOCAB_SIZE),
        .EMBED_DIM(EMBED_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ONCHIP_SIZE(ONCHIP_SIZE)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i;
    
    initial begin
        $display("Embedding ROM Testbench");
        
        clk = 0; rst_n = 0;
        token_id = 0; lookup_en = 0;
        ext_data = 0; ext_valid = 0;
        init_addr = 0; init_data = 0; init_we = 0;
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Initialize embeddings
        for (i = 0; i < ONCHIP_SIZE; i = i + 1) begin
            init_addr = i;
            init_data = {(EMBED_DIM){i[7:0]}};
            init_we = 1;
            @(posedge clk);
        end
        init_we = 0;
        @(posedge clk);
        
        // Test on-chip lookup
        token_id = 10;
        lookup_en = 1;
        @(posedge clk);
        lookup_en = 0;
        
        while (!valid_out) @(posedge clk);
        $display("Token %0d (on-chip): hit=%b, embed[0]=%h", token_id, cache_hit, embedding[7:0]);
        
        @(posedge clk);
        
        // Test off-chip lookup
        token_id = 100;
        lookup_en = 1;
        @(posedge clk);
        lookup_en = 0;
        
        // Simulate external memory response
        repeat(3) @(posedge clk);
        ext_data = {(EMBED_DIM){8'hAB}};
        ext_valid = 1;
        @(posedge clk);
        ext_valid = 0;
        
        while (!valid_out) @(posedge clk);
        $display("Token %0d (off-chip): hit=%b, embed[0]=%h", 100, cache_hit, embedding[7:0]);
        
        $display("Testbench complete");
        $finish;
    end
endmodule

`endif
