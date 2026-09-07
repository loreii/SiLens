// =============================================================================
// Attention Head Block (EDGE) - Level 1 Synthesis Block - Edge Vision Variant
// =============================================================================
// Edge-optimized attention head for SiLens Edge vision synthesis.
// Computes: Attention(Q,K,V) = softmax(Q·K^T / sqrt(d)) · V
//
// EDGE VARIANT SPECIFICATIONS:
// - HEAD_DIM: 64 (192-dim hidden / 3 attention heads)
// - MAX_SEQ: 64 (vision processing in chunks, 14×14=196 patches chunked)
// - Target clock: 200MHz (5ns period)
// - Estimated area: ~1.5mm² on SKY130
//
// Features:
// - Ternary weight projections for Q, K, V (no multipliers needed)
// - KV cache interface for autoregressive decoding
// - Streaming Q input, cached K/V from external memory
// - Approximate softmax using piece-wise linear function
//
// Target: ~1.5mm² on SKY130 (1225µm × 1225µm)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module attention_head_edge #(
    parameter HEAD_DIM   = 64,       // Dimension per head (192/3 = 64)
    parameter MAX_SEQ    = 64,       // Maximum sequence length for KV cache
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
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] q_data,        // Query vector (512 bits)
    output wire                         q_ready,
    
    // =========================================================================
    // Ternary weight interface for Q/K/V projections (optional)
    // =========================================================================
    input  wire                         use_projection, // Enable internal projection
    input  wire [HEAD_DIM*2-1:0]        w_q,           // Ternary Q weights (128 bits)
    input  wire [HEAD_DIM*2-1:0]        w_k,           // Ternary K weights (128 bits)
    input  wire [HEAD_DIM*2-1:0]        w_v,           // Ternary V weights (128 bits)
    
    // =========================================================================
    // KV Cache memory interface (external SRAM)
    // =========================================================================
    output reg  [$clog2(MAX_SEQ)-1:0]   kv_addr,       // Address in KV cache (6 bits)
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] kv_rdata,     // Read data (K or V) (512 bits)
    output reg  [HEAD_DIM*ACT_WIDTH-1:0] kv_wdata,     // Write data (512 bits)
    output reg                          kv_rd,          // Read enable
    output reg                          kv_wr,          // Write enable
    output reg                          kv_sel,         // 0=K cache, 1=V cache
    
    // =========================================================================
    // Output: Attention output vector
    // =========================================================================
    output reg                          out_valid,
    output reg  [HEAD_DIM*ACT_WIDTH-1:0] out_data,     // Attention output (512 bits)
    input  wire                         out_ready
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam SEQ_BITS = $clog2(MAX_SEQ);  // 6 bits for MAX_SEQ=64
    
    // Scale factor: 1/sqrt(HEAD_DIM) = 1/8 for HEAD_DIM=64
    localparam SCALE_SHIFT = 3;
    
    // =========================================================================
    // State machine
    // =========================================================================
    localparam [3:0] 
        S_IDLE        = 4'd0,
        S_LOAD_Q      = 4'd1,
        S_WRITE_KV    = 4'd2,
        S_COMPUTE_QK  = 4'd3,
        S_SOFTMAX     = 4'd4,
        S_COMPUTE_AV  = 4'd5,
        S_OUTPUT      = 4'd6,
        S_DONE        = 4'd7;
    
    reg [3:0] state, next_state;
    
    // =========================================================================
    // Internal registers
    // =========================================================================
    reg [HEAD_DIM*ACT_WIDTH-1:0] q_reg;
    reg [HEAD_DIM*ACT_WIDTH-1:0] k_reg;
    reg signed [SCORE_WIDTH-1:0] scores [0:MAX_SEQ-1];
    reg [SCORE_WIDTH-1:0] attn_weights [0:MAX_SEQ-1];
    reg signed [ACC_WIDTH-1:0] out_acc [0:HEAD_DIM-1];
    reg [SEQ_BITS-1:0] pos_cnt;
    reg [SEQ_BITS-1:0] seq_len_reg;
    reg [SEQ_BITS-1:0] query_pos_reg;
    reg signed [SCORE_WIDTH-1:0] max_score;
    reg [SEQ_BITS:0] softmax_cnt;  // Extra bit for counting up to 2*MAX_SEQ
    reg [31:0] exp_sum;
    
    // =========================================================================
    // Ternary multiply function
    // =========================================================================
    function automatic signed [ACT_WIDTH:0] ternary_mult;
        input [1:0] w;
        input signed [ACT_WIDTH-1:0] x;
        begin
            case (w)
                2'b01:   ternary_mult = {x[ACT_WIDTH-1], x};
                2'b10:   ternary_mult = -{x[ACT_WIDTH-1], x};
                default: ternary_mult = {(ACT_WIDTH+1){1'b0}};
            endcase
        end
    endfunction
    
    // =========================================================================
    // Dot product computation (Q·K^T) - Optimized for HEAD_DIM=64
    // =========================================================================
    wire signed [ACT_WIDTH-1:0] q_elements [0:HEAD_DIM-1];
    wire signed [ACT_WIDTH-1:0] k_elements [0:HEAD_DIM-1];
    wire signed [ACT_WIDTH*2:0] products_qk [0:HEAD_DIM-1];
    
    genvar gi;
    generate
        for (gi = 0; gi < HEAD_DIM; gi = gi + 1) begin : unpack_gen
            assign q_elements[gi] = $signed(q_reg[gi*ACT_WIDTH +: ACT_WIDTH]);
            assign k_elements[gi] = $signed(k_reg[gi*ACT_WIDTH +: ACT_WIDTH]);
            assign products_qk[gi] = q_elements[gi] * k_elements[gi];
        end
    endgenerate
    
    // Reduction tree for dot product (6 levels for HEAD_DIM=64)
    // Level 1: 64 -> 32
    wire signed [ACT_WIDTH*2+1:0] sum_l1 [0:31];
    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : l1_gen
            assign sum_l1[gi] = $signed(products_qk[gi*2]) + $signed(products_qk[gi*2+1]);
        end
    endgenerate
    
    // Level 2: 32 -> 16
    wire signed [ACT_WIDTH*2+2:0] sum_l2 [0:15];
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : l2_gen
            assign sum_l2[gi] = $signed(sum_l1[gi*2]) + $signed(sum_l1[gi*2+1]);
        end
    endgenerate
    
    // Level 3: 16 -> 8
    wire signed [ACT_WIDTH*2+3:0] sum_l3 [0:7];
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : l3_gen
            assign sum_l3[gi] = $signed(sum_l2[gi*2]) + $signed(sum_l2[gi*2+1]);
        end
    endgenerate
    
    // Level 4: 8 -> 4
    wire signed [ACT_WIDTH*2+4:0] sum_l4 [0:3];
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : l4_gen
            assign sum_l4[gi] = $signed(sum_l3[gi*2]) + $signed(sum_l3[gi*2+1]);
        end
    endgenerate
    
    // Level 5: 4 -> 2
    wire signed [ACT_WIDTH*2+5:0] sum_l5 [0:1];
    assign sum_l5[0] = $signed(sum_l4[0]) + $signed(sum_l4[1]);
    assign sum_l5[1] = $signed(sum_l4[2]) + $signed(sum_l4[3]);
    
    // Level 6: 2 -> 1 (final sum)
    wire signed [ACT_WIDTH*2+6:0] dot_product_raw;
    assign dot_product_raw = $signed(sum_l5[0]) + $signed(sum_l5[1]);
    
    wire signed [SCORE_WIDTH-1:0] dot_product_scaled;
    assign dot_product_scaled = dot_product_raw[ACT_WIDTH*2+6:SCALE_SHIFT];
    
    // =========================================================================
    // PWL exp approximation for softmax
    // =========================================================================
    function automatic [SCORE_WIDTH-1:0] approx_exp;
        input signed [SCORE_WIDTH-1:0] x;
        reg signed [SCORE_WIDTH:0] temp;
        begin
            temp = 256 + (x >>> 4);
            approx_exp = (temp < 0) ? 16'd0 : temp[SCORE_WIDTH-1:0];
        end
    endfunction
    
    // =========================================================================
    // V vector elements
    // =========================================================================
    wire signed [ACT_WIDTH-1:0] v_elements [0:HEAD_DIM-1];
    generate
        for (gi = 0; gi < HEAD_DIM; gi = gi + 1) begin : v_unpack_gen
            assign v_elements[gi] = $signed(kv_rdata[gi*ACT_WIDTH +: ACT_WIDTH]);
        end
    endgenerate
    
    // =========================================================================
    // State machine transitions (combinational)
    // =========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:       if (start) next_state = S_LOAD_Q;
            S_LOAD_Q:     if (q_valid) next_state = S_WRITE_KV;
            S_WRITE_KV:   if (pos_cnt == 1) next_state = S_COMPUTE_QK;
            S_COMPUTE_QK: if (pos_cnt == seq_len_reg) next_state = S_SOFTMAX;
            S_SOFTMAX:    if (softmax_cnt == {1'b0, seq_len_reg} + {1'b0, seq_len_reg}) next_state = S_COMPUTE_AV;
            S_COMPUTE_AV: if (pos_cnt == seq_len_reg) next_state = S_OUTPUT;
            S_OUTPUT:     if (out_ready) next_state = S_DONE;
            S_DONE:       next_state = S_IDLE;
            default:      next_state = S_IDLE;
        endcase
    end
    
    assign q_ready = (state == S_LOAD_Q);
    
    // =========================================================================
    // Scores array - individual always blocks via generate
    // =========================================================================
    generate
        for (gi = 0; gi < MAX_SEQ; gi = gi + 1) begin : gen_scores
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    scores[gi] <= {SCORE_WIDTH{1'b0}};
                end else if (state == S_COMPUTE_QK && pos_cnt > 0 && (pos_cnt - 1) == gi[SEQ_BITS-1:0]) begin
                    scores[gi] <= dot_product_scaled;
                end
            end
        end
    endgenerate
    
    // =========================================================================
    // Attn_weights array - individual always blocks via generate
    // =========================================================================
    generate
        for (gi = 0; gi < MAX_SEQ; gi = gi + 1) begin : gen_attn_weights
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    attn_weights[gi] <= {SCORE_WIDTH{1'b0}};
                end else if (state == S_SOFTMAX && softmax_cnt >= {1'b0, seq_len_reg} && (softmax_cnt - {1'b0, seq_len_reg}) == gi[SEQ_BITS:0]) begin
                    attn_weights[gi] <= approx_exp(scores[gi] - max_score);
                end
            end
        end
    endgenerate
    
    // =========================================================================
    // Out_acc array - individual always blocks via generate
    // =========================================================================
    generate
        for (gi = 0; gi < HEAD_DIM; gi = gi + 1) begin : gen_out_acc
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    out_acc[gi] <= {ACC_WIDTH{1'b0}};
                end else if (state == S_IDLE && start) begin
                    out_acc[gi] <= {ACC_WIDTH{1'b0}};
                end else if (state == S_COMPUTE_AV && pos_cnt > 0) begin
                    out_acc[gi] <= out_acc[gi] + 
                        (($signed(v_elements[gi]) * $signed({1'b0, attn_weights[pos_cnt-1]})) >>> 8);
                end
            end
        end
    endgenerate
    
    // =========================================================================
    // Out_data array - individual always blocks via generate
    // =========================================================================
    generate
        for (gi = 0; gi < HEAD_DIM; gi = gi + 1) begin : gen_out_data
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    out_data[gi*ACT_WIDTH +: ACT_WIDTH] <= {ACT_WIDTH{1'b0}};
                end else if (state == S_OUTPUT) begin
                    out_data[gi*ACT_WIDTH +: ACT_WIDTH] <= out_acc[gi][ACC_WIDTH-1 -: ACT_WIDTH];
                end
            end
        end
    endgenerate
    
    // =========================================================================
    // Main state machine - scalar signals only
    // =========================================================================
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
            softmax_cnt <= {(SEQ_BITS+1){1'b0}};
            max_score <= {SCORE_WIDTH{1'b1}};
            exp_sum <= 32'd0;
            q_reg <= {(HEAD_DIM*ACT_WIDTH){1'b0}};
            k_reg <= {(HEAD_DIM*ACT_WIDTH){1'b0}};
            seq_len_reg <= {SEQ_BITS{1'b0}};
            query_pos_reg <= {SEQ_BITS{1'b0}};
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
                        softmax_cnt <= {(SEQ_BITS+1){1'b0}};
                        max_score <= {1'b1, {(SCORE_WIDTH-1){1'b0}}};
                        exp_sum <= 32'd0;
                    end
                end
                
                S_LOAD_Q: begin
                    if (q_valid) begin
                        q_reg <= q_data;
                        pos_cnt <= {SEQ_BITS{1'b0}};
                    end
                end
                
                S_WRITE_KV: begin
                    kv_wr <= 1'b1;
                    kv_addr <= query_pos_reg;
                    kv_sel <= pos_cnt[0];
                    kv_wdata <= q_reg;
                    pos_cnt <= pos_cnt + 1'b1;
                    
                    if (pos_cnt == 1) begin
                        kv_wr <= 1'b0;
                        pos_cnt <= {SEQ_BITS{1'b0}};
                    end
                end
                
                S_COMPUTE_QK: begin
                    kv_wr <= 1'b0;
                    kv_sel <= 1'b0;
                    kv_rd <= 1'b1;
                    kv_addr <= pos_cnt;
                    
                    if (pos_cnt > 0) begin
                        k_reg <= kv_rdata;
                    end
                    
                    if (pos_cnt == seq_len_reg) begin
                        kv_rd <= 1'b0;
                        k_reg <= kv_rdata;
                    end else begin
                        pos_cnt <= pos_cnt + 1'b1;
                    end
                end
                
                S_SOFTMAX: begin
                    kv_rd <= 1'b0;
                    kv_wr <= 1'b0;
                    
                    if (softmax_cnt < {1'b0, seq_len_reg}) begin
                        // Pass 1: Find max
                        if ($signed(scores[softmax_cnt[SEQ_BITS-1:0]]) > $signed(max_score)) begin
                            max_score <= scores[softmax_cnt[SEQ_BITS-1:0]];
                        end
                    end else begin
                        // Pass 2: Accumulate exp_sum
                        if (softmax_cnt == {1'b0, seq_len_reg}) begin
                            exp_sum <= 32'd0;
                        end
                        exp_sum <= exp_sum + approx_exp(scores[softmax_cnt[SEQ_BITS-1:0] - seq_len_reg] - max_score);
                    end
                    
                    softmax_cnt <= softmax_cnt + 1'b1;
                    
                    if (softmax_cnt == {1'b0, seq_len_reg} + {1'b0, seq_len_reg} - 1) begin
                        pos_cnt <= {SEQ_BITS{1'b0}};
                    end
                end
                
                S_COMPUTE_AV: begin
                    kv_sel <= 1'b1;
                    kv_rd <= 1'b1;
                    kv_addr <= pos_cnt;
                    
                    if (pos_cnt == seq_len_reg) begin
                        kv_rd <= 1'b0;
                    end else begin
                        pos_cnt <= pos_cnt + 1'b1;
                    end
                end
                
                S_OUTPUT: begin
                    out_valid <= 1'b1;
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
