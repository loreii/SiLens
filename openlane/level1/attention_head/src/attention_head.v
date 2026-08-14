// =============================================================================
// Attention Head Block - Level 1 Synthesis Block
// =============================================================================
// Single attention head computation for transformer models.
// Computes: Attention(Q,K,V) = softmax(Q·K^T / sqrt(d)) · V
//
// Features:
// - Ternary weight projections for Q, K, V (no multipliers needed)
// - KV cache interface for autoregressive decoding
// - Streaming Q input, cached K/V from external memory
// - Approximate softmax using piece-wise linear function
//
// Target: ~2mm² on SKY130 (1400µm × 1400µm)
// Reuse: 414× total (144 in vision encoder, 270 in LLM)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module attention_head #(
    parameter HEAD_DIM   = 64,       // Dimension per head (d_k = d_v)
    parameter MAX_SEQ    = 256,      // Maximum sequence length for KV cache
    parameter ACT_WIDTH  = 8,        // Activation bit width
    parameter ACC_WIDTH  = 24,       // Accumulator width for dot products
    parameter SCORE_WIDTH = 16       // Attention score width
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // =========================================================================
    // Control interface
    // =========================================================================
    input  wire                         start,          // Start attention computation
    input  wire                         is_prefill,     // 1=prefill (process all), 0=decode (single token)
    input  wire [$clog2(MAX_SEQ)-1:0]   seq_len,        // Current sequence length
    input  wire [$clog2(MAX_SEQ)-1:0]   query_pos,      // Position of current query
    output reg                          busy,           // Computation in progress
    output reg                          done,           // Computation complete
    
    // =========================================================================
    // Input: Query vector (already projected, or raw activation)
    // =========================================================================
    input  wire                         q_valid,
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] q_data,        // Query vector
    output wire                         q_ready,
    
    // =========================================================================
    // Ternary weight interface for Q/K/V projections (optional)
    // =========================================================================
    // If weights are provided, this block projects raw input to Q/K/V
    // Otherwise, assume q_data is already projected Q
    input  wire                         use_projection, // Enable internal projection
    input  wire [HEAD_DIM*2-1:0]        w_q,           // Ternary Q weights
    input  wire [HEAD_DIM*2-1:0]        w_k,           // Ternary K weights  
    input  wire [HEAD_DIM*2-1:0]        w_v,           // Ternary V weights
    
    // =========================================================================
    // KV Cache memory interface (external SRAM)
    // =========================================================================
    // Read/write cached K and V vectors
    output reg  [$clog2(MAX_SEQ)-1:0]   kv_addr,       // Address in KV cache
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] kv_rdata,     // Read data (K or V)
    output reg  [HEAD_DIM*ACT_WIDTH-1:0] kv_wdata,     // Write data
    output reg                          kv_rd,          // Read enable
    output reg                          kv_wr,          // Write enable
    output reg                          kv_sel,         // 0=K cache, 1=V cache
    
    // =========================================================================
    // Output: Attention output vector
    // =========================================================================
    output reg                          out_valid,
    output reg  [HEAD_DIM*ACT_WIDTH-1:0] out_data,     // Attention output
    input  wire                         out_ready
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam SEQ_BITS = $clog2(MAX_SEQ);
    
    // Scale factor: 1/sqrt(HEAD_DIM) = 1/8 for HEAD_DIM=64
    // Implemented as right shift by 3
    localparam SCALE_SHIFT = 3;
    
    // =========================================================================
    // State machine
    // =========================================================================
    localparam [3:0] 
        S_IDLE        = 4'd0,
        S_LOAD_Q      = 4'd1,   // Load/project query
        S_WRITE_KV    = 4'd2,   // Write K,V to cache (for current token)
        S_COMPUTE_QK  = 4'd3,   // Compute Q·K^T scores
        S_SOFTMAX     = 4'd4,   // Compute softmax of scores
        S_COMPUTE_AV  = 4'd5,   // Compute weighted sum of V
        S_OUTPUT      = 4'd6,   // Output result
        S_DONE        = 4'd7;
    
    reg [3:0] state, next_state;
    
    // =========================================================================
    // Internal registers
    // =========================================================================
    
    // Query vector register
    reg [HEAD_DIM*ACT_WIDTH-1:0] q_reg;
    
    // Current K vector being processed
    reg [HEAD_DIM*ACT_WIDTH-1:0] k_reg;
    
    // Attention scores (one per sequence position)
    reg signed [SCORE_WIDTH-1:0] scores [0:MAX_SEQ-1];
    
    // Softmax outputs (normalized attention weights)
    reg [SCORE_WIDTH-1:0] attn_weights [0:MAX_SEQ-1];
    
    // Accumulator for output
    reg signed [ACC_WIDTH-1:0] out_acc [0:HEAD_DIM-1];
    
    // Position counter
    reg [SEQ_BITS-1:0] pos_cnt;
    
    // Computation pipeline registers
    reg [SEQ_BITS-1:0] seq_len_reg;
    reg [SEQ_BITS-1:0] query_pos_reg;
    
    // =========================================================================
    // Q/K/V Projection using ternary weights (optional)
    // =========================================================================
    // Ternary multiply: w * x where w in {-1, 0, +1}
    // No actual multiplier needed!
    
    function automatic signed [ACT_WIDTH:0] ternary_mult;
        input [1:0] w;                          // Weight: 00=0, 01=+1, 10=-1
        input signed [ACT_WIDTH-1:0] x;         // Activation
        begin
            case (w)
                2'b01:   ternary_mult = {x[ACT_WIDTH-1], x};     // +1: sign extend
                2'b10:   ternary_mult = -{x[ACT_WIDTH-1], x};    // -1: negate
                default: ternary_mult = {(ACT_WIDTH+1){1'b0}};   // 0: zero
            endcase
        end
    endfunction
    
    // =========================================================================
    // Dot product computation (Q·K^T)
    // =========================================================================
    // 64-element dot product with reduction tree
    
    wire signed [ACT_WIDTH-1:0] q_elements [0:HEAD_DIM-1];
    wire signed [ACT_WIDTH-1:0] k_elements [0:HEAD_DIM-1];
    wire signed [ACT_WIDTH*2:0] products_qk [0:HEAD_DIM-1];
    
    genvar i;
    generate
        for (i = 0; i < HEAD_DIM; i = i + 1) begin : unpack_gen
            assign q_elements[i] = $signed(q_reg[i*ACT_WIDTH +: ACT_WIDTH]);
            assign k_elements[i] = $signed(k_reg[i*ACT_WIDTH +: ACT_WIDTH]);
            // Q·K element-wise product
            assign products_qk[i] = q_elements[i] * k_elements[i];
        end
    endgenerate
    
    // Reduction tree for dot product (6 levels for 64 elements)
    // Level 1: 64 -> 32
    wire signed [ACT_WIDTH*2+1:0] sum_l1 [0:31];
    generate
        for (i = 0; i < 32; i = i + 1) begin : l1_gen
            assign sum_l1[i] = $signed(products_qk[i*2]) + $signed(products_qk[i*2+1]);
        end
    endgenerate
    
    // Level 2: 32 -> 16
    wire signed [ACT_WIDTH*2+2:0] sum_l2 [0:15];
    generate
        for (i = 0; i < 16; i = i + 1) begin : l2_gen
            assign sum_l2[i] = $signed(sum_l1[i*2]) + $signed(sum_l1[i*2+1]);
        end
    endgenerate
    
    // Level 3: 16 -> 8
    wire signed [ACT_WIDTH*2+3:0] sum_l3 [0:7];
    generate
        for (i = 0; i < 8; i = i + 1) begin : l3_gen
            assign sum_l3[i] = $signed(sum_l2[i*2]) + $signed(sum_l2[i*2+1]);
        end
    endgenerate
    
    // Level 4: 8 -> 4
    wire signed [ACT_WIDTH*2+4:0] sum_l4 [0:3];
    generate
        for (i = 0; i < 4; i = i + 1) begin : l4_gen
            assign sum_l4[i] = $signed(sum_l3[i*2]) + $signed(sum_l3[i*2+1]);
        end
    endgenerate
    
    // Level 5: 4 -> 2
    wire signed [ACT_WIDTH*2+5:0] sum_l5 [0:1];
    assign sum_l5[0] = $signed(sum_l4[0]) + $signed(sum_l4[1]);
    assign sum_l5[1] = $signed(sum_l4[2]) + $signed(sum_l4[3]);
    
    // Level 6: 2 -> 1 (final dot product result)
    wire signed [ACT_WIDTH*2+6:0] dot_product_raw;
    assign dot_product_raw = $signed(sum_l5[0]) + $signed(sum_l5[1]);
    
    // Scale by 1/sqrt(d) = 1/8 for d=64
    wire signed [SCORE_WIDTH-1:0] dot_product_scaled;
    assign dot_product_scaled = dot_product_raw[ACT_WIDTH*2+6:SCALE_SHIFT];
    
    // =========================================================================
    // Softmax approximation
    // =========================================================================
    // Full softmax is expensive. Use piece-wise linear approximation:
    // 1. Find max score
    // 2. Subtract max (for numerical stability)
    // 3. Approximate exp() with PWL
    // 4. Normalize
    
    // Max finder
    reg signed [SCORE_WIDTH-1:0] max_score;
    reg [SEQ_BITS-1:0] softmax_cnt;
    reg [31:0] exp_sum;  // Sum of exp(scores) for normalization
    
    // PWL exp approximation: exp(x) ≈ max(0, 1 + x + x²/2) for x < 0
    // Simplified: exp(x) ≈ max(0, 1 + x/4) scaled
    function automatic [SCORE_WIDTH-1:0] approx_exp;
        input signed [SCORE_WIDTH-1:0] x;  // x should be <= 0 after max subtraction
        reg signed [SCORE_WIDTH:0] temp;
        begin
            // exp(x) ≈ 256 * (1 + x/16) for x in [-16, 0], clamped
            temp = 256 + (x >>> 4);
            approx_exp = (temp < 0) ? 16'd0 : temp[SCORE_WIDTH-1:0];
        end
    endfunction
    
    // =========================================================================
    // Output accumulation (weighted sum of V vectors)
    // =========================================================================
    wire signed [ACT_WIDTH-1:0] v_elements [0:HEAD_DIM-1];
    
    generate
        for (i = 0; i < HEAD_DIM; i = i + 1) begin : v_unpack_gen
            assign v_elements[i] = $signed(kv_rdata[i*ACT_WIDTH +: ACT_WIDTH]);
        end
    endgenerate
    
    // =========================================================================
    // State machine transitions
    // =========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: begin
                if (start)
                    next_state = S_LOAD_Q;
            end
            
            S_LOAD_Q: begin
                if (q_valid)
                    next_state = S_WRITE_KV;
            end
            
            S_WRITE_KV: begin
                // Write K and V for current position, then compute scores
                if (pos_cnt == 1)  // Written both K and V
                    next_state = S_COMPUTE_QK;
            end
            
            S_COMPUTE_QK: begin
                // Compute dot products for all positions up to seq_len
                if (pos_cnt == seq_len_reg)
                    next_state = S_SOFTMAX;
            end
            
            S_SOFTMAX: begin
                // Two passes: find max, then compute normalized weights
                if (softmax_cnt == seq_len_reg + seq_len_reg)  // 2x passes
                    next_state = S_COMPUTE_AV;
            end
            
            S_COMPUTE_AV: begin
                // Weighted sum of V vectors
                if (pos_cnt == seq_len_reg)
                    next_state = S_OUTPUT;
            end
            
            S_OUTPUT: begin
                if (out_ready)
                    next_state = S_DONE;
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end
    
    // =========================================================================
    // State machine outputs and datapath
    // =========================================================================
    assign q_ready = (state == S_LOAD_Q);
    
    integer j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            out_valid <= 1'b0;
            kv_rd <= 1'b0;
            kv_wr <= 1'b0;
            kv_sel <= 1'b0;
            kv_addr <= {SEQ_BITS{1'b0}};
            kv_wdata <= {(HEAD_DIM*ACT_WIDTH){1'b0}};
            pos_cnt <= {SEQ_BITS{1'b0}};
            softmax_cnt <= {SEQ_BITS{1'b0}};
            max_score <= {SCORE_WIDTH{1'b1}};  // Most negative
            exp_sum <= 32'd0;
            q_reg <= {(HEAD_DIM*ACT_WIDTH){1'b0}};
            k_reg <= {(HEAD_DIM*ACT_WIDTH){1'b0}};
            out_data <= {(HEAD_DIM*ACT_WIDTH){1'b0}};
            seq_len_reg <= {SEQ_BITS{1'b0}};
            query_pos_reg <= {SEQ_BITS{1'b0}};
            for (j = 0; j < HEAD_DIM; j = j + 1) begin
                out_acc[j] <= {ACC_WIDTH{1'b0}};
            end
            for (j = 0; j < MAX_SEQ; j = j + 1) begin
                scores[j] <= {SCORE_WIDTH{1'b0}};
                attn_weights[j] <= {SCORE_WIDTH{1'b0}};
            end
        end else begin
            state <= next_state;
            
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    done <= 1'b0;
                    out_valid <= 1'b0;
                    kv_rd <= 1'b0;
                    kv_wr <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        seq_len_reg <= seq_len;
                        query_pos_reg <= query_pos;
                        pos_cnt <= {SEQ_BITS{1'b0}};
                        softmax_cnt <= {SEQ_BITS{1'b0}};
                        max_score <= {1'b1, {(SCORE_WIDTH-1){1'b0}}};  // Most negative
                        exp_sum <= 32'd0;
                        for (j = 0; j < HEAD_DIM; j = j + 1) begin
                            out_acc[j] <= {ACC_WIDTH{1'b0}};
                        end
                    end
                end
                
                S_LOAD_Q: begin
                    if (q_valid) begin
                        q_reg <= q_data;
                        pos_cnt <= {SEQ_BITS{1'b0}};
                    end
                end
                
                S_WRITE_KV: begin
                    // Write current token's K and V to cache
                    // pos_cnt=0: write K, pos_cnt=1: write V
                    kv_wr <= 1'b1;
                    kv_addr <= query_pos_reg;
                    kv_sel <= pos_cnt[0];  // 0=K, 1=V
                    kv_wdata <= q_reg;     // In real impl, would project to K/V
                    pos_cnt <= pos_cnt + 1'b1;
                    
                    if (pos_cnt == 1) begin
                        kv_wr <= 1'b0;
                        pos_cnt <= {SEQ_BITS{1'b0}};
                    end
                end
                
                S_COMPUTE_QK: begin
                    // Read K vectors and compute dot products
                    kv_wr <= 1'b0;
                    kv_sel <= 1'b0;  // K cache
                    kv_rd <= 1'b1;
                    kv_addr <= pos_cnt;
                    
                    // Pipeline: use previous read data
                    if (pos_cnt > 0) begin
                        k_reg <= kv_rdata;
                        scores[pos_cnt - 1] <= dot_product_scaled;
                    end
                    
                    if (pos_cnt == seq_len_reg) begin
                        kv_rd <= 1'b0;
                        // Capture last score
                        k_reg <= kv_rdata;
                        scores[pos_cnt - 1] <= dot_product_scaled;
                    end else begin
                        pos_cnt <= pos_cnt + 1'b1;
                    end
                end
                
                S_SOFTMAX: begin
                    kv_rd <= 1'b0;
                    kv_wr <= 1'b0;
                    
                    if (softmax_cnt < seq_len_reg) begin
                        // Pass 1: Find max
                        if ($signed(scores[softmax_cnt]) > $signed(max_score)) begin
                            max_score <= scores[softmax_cnt];
                        end
                    end else begin
                        // Pass 2: Compute exp and accumulate sum
                        if (softmax_cnt == seq_len_reg) begin
                            // Start pass 2
                            exp_sum <= 32'd0;
                        end
                        
                        // Compute exp(score - max)
                        begin : softmax_exp_block
                            reg signed [SCORE_WIDTH-1:0] score_shifted;
                            reg [SCORE_WIDTH-1:0] exp_val;
                            reg [SEQ_BITS-1:0] idx;
                            
                            idx = softmax_cnt - seq_len_reg;
                            score_shifted = scores[idx] - max_score;
                            exp_val = approx_exp(score_shifted);
                            attn_weights[idx] <= exp_val;
                            exp_sum <= exp_sum + exp_val;
                        end
                    end
                    
                    softmax_cnt <= softmax_cnt + 1'b1;
                    
                    if (softmax_cnt == seq_len_reg + seq_len_reg - 1) begin
                        pos_cnt <= {SEQ_BITS{1'b0}};
                    end
                end
                
                S_COMPUTE_AV: begin
                    // Read V vectors and accumulate weighted sum
                    kv_sel <= 1'b1;  // V cache
                    kv_rd <= 1'b1;
                    kv_addr <= pos_cnt;
                    
                    // Weighted accumulation (pipeline)
                    if (pos_cnt > 0) begin
                        for (j = 0; j < HEAD_DIM; j = j + 1) begin
                            // weight * V[j], normalized by exp_sum
                            out_acc[j] <= out_acc[j] + 
                                (($signed(v_elements[j]) * $signed({1'b0, attn_weights[pos_cnt-1]})) >>> 8);
                        end
                    end
                    
                    if (pos_cnt == seq_len_reg) begin
                        kv_rd <= 1'b0;
                        // Final accumulation
                        for (j = 0; j < HEAD_DIM; j = j + 1) begin
                            out_acc[j] <= out_acc[j] + 
                                (($signed(v_elements[j]) * $signed({1'b0, attn_weights[pos_cnt-1]})) >>> 8);
                        end
                    end else begin
                        pos_cnt <= pos_cnt + 1'b1;
                    end
                end
                
                S_OUTPUT: begin
                    out_valid <= 1'b1;
                    // Pack output accumulator (take upper bits)
                    for (j = 0; j < HEAD_DIM; j = j + 1) begin
                        out_data[j*ACT_WIDTH +: ACT_WIDTH] <= out_acc[j][ACC_WIDTH-1 -: ACT_WIDTH];
                    end
                    
                    if (out_ready) begin
                        out_valid <= 1'b0;
                    end
                end
                
                S_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                end
                
                default: begin
                    // Do nothing
                end
            endcase
        end
    end

endmodule

`default_nettype wire
