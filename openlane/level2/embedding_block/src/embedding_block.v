// =============================================================================
// Embedding Block - Level 2 Synthesis Block
// =============================================================================
// Token and Position Embeddings for LLM decoder.
// Combines: output = token_embedding + position_embedding
//
// Architecture:
// - Token embedding: Vocabulary (49152) → 576-dim vectors (27MB external DRAM)
// - Position embedding: Max 2048 positions → 576-dim vectors (1.1MB on-chip)
//
// Features:
// - Pipelined token and position lookups
// - Saturating addition of embeddings
// - Batch processing support
// - External memory interface for token embeddings (DDR3/DRAM)
// - On-chip ROM/SRAM for position embeddings
// - Streaming output interface
//
// Target: ~15mm² on SKY130 (3900µm × 3900µm)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module embedding_block #(
    parameter VOCAB_SIZE    = 49152,     // Vocabulary size (tokens 0-49151)
    parameter MAX_POS       = 2048,      // Maximum sequence positions
    parameter EMBED_DIM     = 576,       // Embedding dimension
    parameter DATA_WIDTH    = 8,         // Embedding element width (8-bit)
    parameter TOKEN_BITS    = 16,        // Bits for token index
    parameter POS_BITS      = 12,        // Bits for position (log2(2048) = 11, use 12)
    parameter BATCH_MAX     = 32,        // Maximum batch size
    parameter MEM_BURST_LEN = 8          // Memory burst length (bytes per beat)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // =========================================================================
    // Control interface
    // =========================================================================
    input  wire                         start,          // Start embedding lookup
    input  wire [$clog2(BATCH_MAX):0]   batch_size,     // Number of tokens to process
    output reg                          busy,           // Processing in progress
    output reg                          done,           // Batch complete
    
    // =========================================================================
    // Token input interface
    // =========================================================================
    input  wire                         token_valid,    // Token input valid
    input  wire [TOKEN_BITS-1:0]        token_id,       // Token index (0 to VOCAB_SIZE-1)
    input  wire [POS_BITS-1:0]          token_pos,      // Position in sequence
    output wire                         token_ready,    // Ready to accept token
    
    // =========================================================================
    // Token embedding memory interface (external DRAM)
    // =========================================================================
    // Address = token_id * EMBED_DIM (byte address)
    // Reads 576 bytes = 72 bursts of 8 bytes
    output reg                          tok_mem_req,    // Memory request
    output reg  [31:0]                  tok_mem_addr,   // Byte address
    output reg  [7:0]                   tok_mem_len,    // Burst length (beats - 1)
    input  wire                         tok_mem_grant,  // Request granted
    input  wire                         tok_mem_valid,  // Read data valid
    input  wire [MEM_BURST_LEN*8-1:0]   tok_mem_data,   // Read data (64 bits)
    output wire                         tok_mem_ready,  // Ready to accept data
    
    // =========================================================================
    // Position embedding interface (on-chip SRAM or ROM)
    // =========================================================================
    // Smaller (1.1MB) - can fit in on-chip memory
    output reg  [POS_BITS-1:0]          pos_mem_addr,   // Position index
    output reg  [$clog2(EMBED_DIM)-1:0] pos_mem_elem,   // Element index within vector
    output reg                          pos_mem_rd,     // Read enable
    input  wire [DATA_WIDTH-1:0]        pos_mem_data,   // Read data (one element)
    input  wire                         pos_mem_valid,  // Data valid
    
    // =========================================================================
    // Output streaming interface
    // =========================================================================
    output reg                          out_valid,      // Output valid
    output reg  [DATA_WIDTH-1:0]        out_data,       // One element at a time
    output reg  [$clog2(EMBED_DIM)-1:0] out_elem_idx,   // Current element index
    output reg                          out_last_elem,  // Last element of vector
    output reg                          out_last_token, // Last token of batch
    input  wire                         out_ready       // Downstream ready
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam ELEM_BITS     = $clog2(EMBED_DIM);
    localparam BATCH_BITS    = $clog2(BATCH_MAX) + 1;
    localparam BURST_BYTES   = MEM_BURST_LEN;  // 8 bytes per burst beat
    localparam BURSTS_PER_VEC = (EMBED_DIM + BURST_BYTES - 1) / BURST_BYTES;  // 72 bursts
    
    // =========================================================================
    // State machine
    // =========================================================================
    localparam [3:0]
        S_IDLE          = 4'd0,
        S_FETCH_TOKEN   = 4'd1,     // Accept token input
        S_REQ_TOK_MEM   = 4'd2,     // Request token embedding from DRAM
        S_WAIT_TOK_MEM  = 4'd3,     // Wait for memory grant
        S_READ_TOK_EMB  = 4'd4,     // Read token embedding data
        S_READ_POS_EMB  = 4'd5,     // Read position embedding (pipelined)
        S_COMBINE       = 4'd6,     // Add token + position embeddings
        S_OUTPUT        = 4'd7,     // Stream output
        S_NEXT_TOKEN    = 4'd8,     // Move to next token in batch
        S_DONE          = 4'd9;
    
    reg [3:0] state, next_state;
    
    // =========================================================================
    // Token/Position registers
    // =========================================================================
    reg [TOKEN_BITS-1:0]    current_token;
    reg [POS_BITS-1:0]      current_pos;
    reg [BATCH_BITS-1:0]    batch_cnt;          // Current token in batch
    reg [BATCH_BITS-1:0]    batch_size_reg;     // Registered batch size
    
    // =========================================================================
    // Embedding buffers
    // =========================================================================
    // Token embedding buffer (filled from external memory)
    reg [DATA_WIDTH-1:0] tok_emb_buf [0:EMBED_DIM-1];
    
    // Position embedding buffer (filled from on-chip memory)
    reg [DATA_WIDTH-1:0] pos_emb_buf [0:EMBED_DIM-1];
    
    // Combined embedding buffer
    reg [DATA_WIDTH-1:0] combined_buf [0:EMBED_DIM-1];
    
    // =========================================================================
    // Memory interface counters
    // =========================================================================
    reg [7:0]  burst_cnt;       // Current burst within token embedding read
    reg [ELEM_BITS-1:0] elem_cnt;  // Element counter
    reg [2:0]  byte_cnt;        // Byte within burst
    
    // =========================================================================
    // Pipeline registers for position embedding read
    // =========================================================================
    reg pos_read_active;
    reg [ELEM_BITS-1:0] pos_read_elem;
    
    // =========================================================================
    // Saturating addition
    // =========================================================================
    function automatic [DATA_WIDTH-1:0] sat_add;
        input signed [DATA_WIDTH-1:0] a;
        input signed [DATA_WIDTH-1:0] b;
        reg signed [DATA_WIDTH:0] sum;
        begin
            sum = {a[DATA_WIDTH-1], a} + {b[DATA_WIDTH-1], b};
            // Saturate on overflow
            if (sum > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}}))
                sat_add = {1'b0, {(DATA_WIDTH-1){1'b1}}};  // Max positive
            else if (sum < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}}))
                sat_add = {1'b1, {(DATA_WIDTH-1){1'b0}}};  // Max negative
            else
                sat_add = sum[DATA_WIDTH-1:0];
        end
    endfunction
    
    // =========================================================================
    // State machine transitions
    // =========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: begin
                if (start && batch_size > 0)
                    next_state = S_FETCH_TOKEN;
            end
            
            S_FETCH_TOKEN: begin
                if (token_valid)
                    next_state = S_REQ_TOK_MEM;
            end
            
            S_REQ_TOK_MEM: begin
                next_state = S_WAIT_TOK_MEM;
            end
            
            S_WAIT_TOK_MEM: begin
                if (tok_mem_grant)
                    next_state = S_READ_TOK_EMB;
            end
            
            S_READ_TOK_EMB: begin
                // Read all bursts for token embedding
                if (elem_cnt >= EMBED_DIM - 1 && tok_mem_valid)
                    next_state = S_READ_POS_EMB;
            end
            
            S_READ_POS_EMB: begin
                // Read all position embedding elements
                if (elem_cnt >= EMBED_DIM - 1 && pos_mem_valid)
                    next_state = S_COMBINE;
            end
            
            S_COMBINE: begin
                // Combine in one cycle (combinational, stored next cycle)
                next_state = S_OUTPUT;
            end
            
            S_OUTPUT: begin
                // Stream all elements
                if (out_ready && elem_cnt >= EMBED_DIM - 1)
                    next_state = S_NEXT_TOKEN;
            end
            
            S_NEXT_TOKEN: begin
                if (batch_cnt >= batch_size_reg - 1)
                    next_state = S_DONE;
                else
                    next_state = S_FETCH_TOKEN;
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end
    
    // =========================================================================
    // Control signals
    // =========================================================================
    assign token_ready = (state == S_FETCH_TOKEN);
    assign tok_mem_ready = (state == S_READ_TOK_EMB);
    
    // =========================================================================
    // Main state machine logic
    // =========================================================================
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            out_valid <= 1'b0;
            out_data <= {DATA_WIDTH{1'b0}};
            out_elem_idx <= {ELEM_BITS{1'b0}};
            out_last_elem <= 1'b0;
            out_last_token <= 1'b0;
            
            tok_mem_req <= 1'b0;
            tok_mem_addr <= 32'd0;
            tok_mem_len <= 8'd0;
            
            pos_mem_addr <= {POS_BITS{1'b0}};
            pos_mem_elem <= {ELEM_BITS{1'b0}};
            pos_mem_rd <= 1'b0;
            
            current_token <= {TOKEN_BITS{1'b0}};
            current_pos <= {POS_BITS{1'b0}};
            batch_cnt <= {BATCH_BITS{1'b0}};
            batch_size_reg <= {BATCH_BITS{1'b0}};
            
            burst_cnt <= 8'd0;
            elem_cnt <= {ELEM_BITS{1'b0}};
            byte_cnt <= 3'd0;
            
            pos_read_active <= 1'b0;
            pos_read_elem <= {ELEM_BITS{1'b0}};
            
            for (i = 0; i < EMBED_DIM; i = i + 1) begin
                tok_emb_buf[i] <= {DATA_WIDTH{1'b0}};
                pos_emb_buf[i] <= {DATA_WIDTH{1'b0}};
                combined_buf[i] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            state <= next_state;
            
            // Default deassertions
            tok_mem_req <= 1'b0;
            pos_mem_rd <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    done <= 1'b0;
                    out_valid <= 1'b0;
                    if (start && batch_size > 0) begin
                        busy <= 1'b1;
                        batch_size_reg <= batch_size;
                        batch_cnt <= {BATCH_BITS{1'b0}};
                    end
                end
                
                S_FETCH_TOKEN: begin
                    if (token_valid) begin
                        current_token <= token_id;
                        current_pos <= token_pos;
                        elem_cnt <= {ELEM_BITS{1'b0}};
                        burst_cnt <= 8'd0;
                    end
                end
                
                S_REQ_TOK_MEM: begin
                    // Request token embedding from external memory
                    // Address = token_id * EMBED_DIM (576 bytes per token)
                    tok_mem_req <= 1'b1;
                    tok_mem_addr <= {current_token, 9'd0} + {current_token, 6'd0}; 
                    // token_id * 576 = token_id * 512 + token_id * 64
                    tok_mem_len <= BURSTS_PER_VEC - 1;  // 72 bursts - 1
                end
                
                S_WAIT_TOK_MEM: begin
                    tok_mem_req <= 1'b1;  // Keep request asserted
                end
                
                S_READ_TOK_EMB: begin
                    if (tok_mem_valid) begin
                        // Unpack burst data into token embedding buffer
                        // Each burst = 8 bytes = 8 elements
                        for (i = 0; i < MEM_BURST_LEN; i = i + 1) begin
                            if (elem_cnt + i < EMBED_DIM) begin
                                tok_emb_buf[elem_cnt + i] <= tok_mem_data[i*DATA_WIDTH +: DATA_WIDTH];
                            end
                        end
                        elem_cnt <= elem_cnt + MEM_BURST_LEN;
                        burst_cnt <= burst_cnt + 1'b1;
                    end
                    
                    // Prepare for position embedding read
                    if (elem_cnt >= EMBED_DIM - MEM_BURST_LEN) begin
                        elem_cnt <= {ELEM_BITS{1'b0}};
                    end
                end
                
                S_READ_POS_EMB: begin
                    // Read position embedding elements sequentially
                    // On-chip memory, so we can read one element per cycle
                    pos_mem_rd <= 1'b1;
                    pos_mem_addr <= current_pos;
                    pos_mem_elem <= elem_cnt;
                    
                    // Pipeline: capture previous read
                    if (pos_mem_valid) begin
                        pos_emb_buf[pos_read_elem] <= pos_mem_data;
                    end
                    
                    pos_read_active <= 1'b1;
                    pos_read_elem <= elem_cnt;
                    
                    if (elem_cnt < EMBED_DIM - 1) begin
                        elem_cnt <= elem_cnt + 1'b1;
                    end
                    
                    // Capture last element and reset
                    if (elem_cnt >= EMBED_DIM - 1 && pos_mem_valid) begin
                        pos_emb_buf[pos_read_elem] <= pos_mem_data;
                        elem_cnt <= {ELEM_BITS{1'b0}};
                        pos_read_active <= 1'b0;
                    end
                end
                
                S_COMBINE: begin
                    // Combine token + position embeddings with saturating addition
                    for (i = 0; i < EMBED_DIM; i = i + 1) begin
                        combined_buf[i] <= sat_add(
                            $signed(tok_emb_buf[i]),
                            $signed(pos_emb_buf[i])
                        );
                    end
                    elem_cnt <= {ELEM_BITS{1'b0}};
                end
                
                S_OUTPUT: begin
                    out_valid <= 1'b1;
                    out_data <= combined_buf[elem_cnt];
                    out_elem_idx <= elem_cnt;
                    out_last_elem <= (elem_cnt == EMBED_DIM - 1);
                    out_last_token <= (batch_cnt == batch_size_reg - 1) && 
                                      (elem_cnt == EMBED_DIM - 1);
                    
                    if (out_ready) begin
                        if (elem_cnt < EMBED_DIM - 1) begin
                            elem_cnt <= elem_cnt + 1'b1;
                        end
                    end
                    
                    // Deassert valid on last element accepted
                    if (out_ready && elem_cnt >= EMBED_DIM - 1) begin
                        out_valid <= 1'b0;
                    end
                end
                
                S_NEXT_TOKEN: begin
                    out_valid <= 1'b0;
                    out_last_elem <= 1'b0;
                    out_last_token <= 1'b0;
                    batch_cnt <= batch_cnt + 1'b1;
                    elem_cnt <= {ELEM_BITS{1'b0}};
                end
                
                S_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    batch_cnt <= {BATCH_BITS{1'b0}};
                end
                
                default: begin
                    // Do nothing
                end
            endcase
        end
    end

endmodule

`default_nettype wire
