// =============================================================================
// SiLens - Language Model Grouped-Query Attention
// =============================================================================
// Implements grouped-query attention (GQA) for the language model decoder.
//
// Architecture:
//   - 9 attention heads
//   - 576 total dimensions (64 per head)
//   - Key-value cache for autoregressive generation
//   - Rotary Position Embeddings (RoPE)
//
// GQA (Grouped-Query Attention):
//   - Queries: 9 heads
//   - Key-Value: fewer heads (can share KV across query heads for efficiency)
//   - Reduces KV cache size
//
// RoPE (Rotary Position Embeddings):
//   - Applies rotation to queries and keys based on position
//   - Enables relative position encoding without additional parameters
//   - q' = q * cos(θ) + rotate(q) * sin(θ)
//
// License: Apache 2.0
// =============================================================================

module llm_attention #(
    parameter DIM         = 576,                    // Model dimension
    parameter NUM_HEADS   = 9,                      // Number of attention heads
    parameter HEAD_DIM    = 64,                     // Dimension per head
    parameter MAX_SEQ_LEN = 8192,                   // Maximum sequence length (for KV cache)
    parameter KV_HEADS    = 9,                      // Number of KV heads (for GQA)
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,               // Input token embedding
    input  wire [$clog2(MAX_SEQ_LEN)-1:0] position,         // Current position (for RoPE)
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Cache control
    input  wire                         cache_clear,        // Clear KV cache (new sequence)
    
    // Hardwired ternary weights
    input  wire [DIM*DIM*2-1:0]         w_q,                // Query weights
    input  wire [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] w_k,        // Key weights
    input  wire [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] w_v,        // Value weights
    input  wire [DIM*DIM*2-1:0]         w_o,                // Output weights
    
    // RoPE frequencies (precomputed cos/sin)
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] rope_cos,          // cos(θ) per head dim
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] rope_sin,          // sin(θ) per head dim
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0]     y_out,
    output reg                          valid_out,
    input  wire                         ready_out
);

    // =========================================================================
    // Weight encoding
    // =========================================================================
    
    localparam W_ZERO = 2'b00;
    localparam W_POS  = 2'b01;
    localparam W_NEG  = 2'b10;
    
    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE       = 4'd0;
    localparam STATE_PROJ_Q     = 4'd1;   // Query projection
    localparam STATE_PROJ_KV    = 4'd2;   // Key/Value projection
    localparam STATE_ROPE       = 4'd3;   // Apply RoPE
    localparam STATE_CACHE_KV   = 4'd4;   // Store K/V in cache
    localparam STATE_ATTENTION  = 4'd5;   // Compute attention scores
    localparam STATE_SOFTMAX    = 4'd6;   // Softmax
    localparam STATE_WEIGHTED   = 4'd7;   // Weighted sum of V
    localparam STATE_PROJ_OUT   = 4'd8;   // Output projection
    localparam STATE_OUTPUT     = 4'd9;
    
    reg [3:0] state;
    
    // =========================================================================
    // KV Cache
    // =========================================================================
    
    // For autoregressive generation, we cache past K and V values
    // K cache: MAX_SEQ_LEN x KV_HEADS x HEAD_DIM
    // V cache: MAX_SEQ_LEN x KV_HEADS x HEAD_DIM
    reg signed [ACT_WIDTH-1:0] k_cache [0:MAX_SEQ_LEN-1][0:KV_HEADS*HEAD_DIM-1];
    reg signed [ACT_WIDTH-1:0] v_cache [0:MAX_SEQ_LEN-1][0:KV_HEADS*HEAD_DIM-1];
    
    // Current cache length
    reg [$clog2(MAX_SEQ_LEN)-1:0] cache_len;
    
    // =========================================================================
    // Processing buffers
    // =========================================================================
    
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];
    reg signed [ACT_WIDTH-1:0] q_buf [0:DIM-1];           // Query after RoPE
    reg signed [ACT_WIDTH-1:0] k_new [0:KV_HEADS*HEAD_DIM-1];  // New key
    reg signed [ACT_WIDTH-1:0] v_new [0:KV_HEADS*HEAD_DIM-1];  // New value
    reg signed [ACT_WIDTH-1:0] attn_out [0:DIM-1];        // Attention output
    
    // Attention scores (for current query against all cached keys)
    reg signed [ACT_WIDTH-1:0] attn_scores [0:MAX_SEQ_LEN-1];
    
    // Processing indices
    reg [$clog2(DIM)-1:0] dim_idx;
    reg [$clog2(NUM_HEADS)-1:0] head_idx;
    reg [$clog2(MAX_SEQ_LEN)-1:0] cache_idx;
    reg [$clog2(DIM/PARALLEL+1)-1:0] proj_iter;
    
    localparam NUM_PROJ_ITERS = (DIM + PARALLEL - 1) / PARALLEL;
    
    // =========================================================================
    // Accumulator
    // =========================================================================
    
    reg signed [ACC_WIDTH-1:0] mac_accum;
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    // =========================================================================
    // Input loading
    // =========================================================================
    
    integer load_i;
    
    always @(posedge clk) begin
        if (state == STATE_IDLE && valid_in) begin
            for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                x_buf[load_i] <= $signed(x_in[load_i*ACT_WIDTH +: ACT_WIDTH]);
            end
        end
    end
    
    // =========================================================================
    // Ternary MAC helper
    // =========================================================================
    
    function signed [ACC_WIDTH-1:0] ternary_mac_single;
        input [ACT_WIDTH-1:0] act;
        input [1:0] weight;
        begin
            case (weight)
                W_POS:   ternary_mac_single = $signed({1'b0, act});
                W_NEG:   ternary_mac_single = -$signed({1'b0, act});
                default: ternary_mac_single = 0;
            endcase
        end
    endfunction
    
    // =========================================================================
    // RoPE application
    // =========================================================================
    // RoPE formula: x' = x * cos(θ) + rotate(x) * sin(θ)
    // Where rotate swaps pairs and negates alternately
    
    function signed [ACT_WIDTH-1:0] apply_rope;
        input signed [ACT_WIDTH-1:0] x_even;
        input signed [ACT_WIDTH-1:0] x_odd;
        input signed [ACT_WIDTH-1:0] cos_val;
        input signed [ACT_WIDTH-1:0] sin_val;
        input integer is_odd;
        reg signed [ACC_WIDTH-1:0] result;
        begin
            if (is_odd) begin
                // Odd positions: x*cos + x_prev*sin
                result = ($signed(x_odd) * $signed(cos_val) + 
                         $signed(x_even) * $signed(sin_val)) >>> FRAC_BITS;
            end else begin
                // Even positions: x*cos - x_next*sin
                result = ($signed(x_even) * $signed(cos_val) - 
                         $signed(x_odd) * $signed(sin_val)) >>> FRAC_BITS;
            end
            apply_rope = saturate(result);
        end
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer init_i, init_j;
    reg signed [ACC_WIDTH-1:0] score_accum;
    reg signed [ACC_WIDTH-1:0] weighted_accum;
    reg signed [ACC_WIDTH-1:0] proj_accum;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            cache_len <= 0;
            dim_idx <= 0;
            head_idx <= 0;
            cache_idx <= 0;
            proj_iter <= 0;
            mac_accum <= 0;
            valid_out <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    
                    if (cache_clear) begin
                        cache_len <= 0;
                    end
                    
                    if (valid_in) begin
                        state <= STATE_PROJ_Q;
                        dim_idx <= 0;
                        proj_iter <= 0;
                        mac_accum <= 0;
                    end
                end
                
                STATE_PROJ_Q: begin
                    // Compute query projection for current dimension
                    // Q[dim_idx] = sum(x[i] * W_q[dim_idx][i])
                    begin : proj_q_block
                        reg signed [ACC_WIDTH-1:0] q_sum;
                        integer qi;
                        q_sum = 0;
                        for (qi = 0; qi < DIM; qi = qi + 1) begin
                            q_sum = q_sum + ternary_mac_single(
                                x_buf[qi],
                                w_q[(dim_idx * DIM + qi) * 2 +: 2]
                            );
                        end
                        q_buf[dim_idx] <= saturate(q_sum);
                    end
                    
                    if (dim_idx >= DIM - 1) begin
                        dim_idx <= 0;
                        state <= STATE_PROJ_KV;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                STATE_PROJ_KV: begin
                    // Compute K and V projections
                    begin : proj_kv_block
                        reg signed [ACC_WIDTH-1:0] k_sum, v_sum;
                        integer kvi;
                        k_sum = 0;
                        v_sum = 0;
                        for (kvi = 0; kvi < DIM; kvi = kvi + 1) begin
                            k_sum = k_sum + ternary_mac_single(
                                x_buf[kvi],
                                w_k[(dim_idx * DIM + kvi) * 2 +: 2]
                            );
                            v_sum = v_sum + ternary_mac_single(
                                x_buf[kvi],
                                w_v[(dim_idx * DIM + kvi) * 2 +: 2]
                            );
                        end
                        k_new[dim_idx] <= saturate(k_sum);
                        v_new[dim_idx] <= saturate(v_sum);
                    end
                    
                    if (dim_idx >= KV_HEADS * HEAD_DIM - 1) begin
                        dim_idx <= 0;
                        state <= STATE_ROPE;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                STATE_ROPE: begin
                    // Apply RoPE to queries and keys
                    // Simplified: just store without full RoPE for now
                    // Full implementation would apply rotation per head
                    state <= STATE_CACHE_KV;
                end
                
                STATE_CACHE_KV: begin
                    // Store new K and V in cache at current position
                    for (init_j = 0; init_j < KV_HEADS * HEAD_DIM; init_j = init_j + 1) begin
                        k_cache[cache_len][init_j] <= k_new[init_j];
                        v_cache[cache_len][init_j] <= v_new[init_j];
                    end
                    
                    cache_len <= cache_len + 1;
                    head_idx <= 0;
                    cache_idx <= 0;
                    state <= STATE_ATTENTION;
                end
                
                STATE_ATTENTION: begin
                    // Compute attention scores for current head
                    // score[cache_idx] = Q[head_idx] · K[cache_idx][head_idx] / sqrt(d)
                    score_accum = 0;
                    for (init_i = 0; init_i < HEAD_DIM; init_i = init_i + 1) begin
                        score_accum = score_accum + 
                            $signed(q_buf[head_idx * HEAD_DIM + init_i]) *
                            $signed(k_cache[cache_idx][head_idx * HEAD_DIM + init_i]);
                    end
                    // Scale by 1/sqrt(64) = 1/8 (shift right by 3)
                    attn_scores[cache_idx] <= saturate(score_accum >>> 3);
                    
                    if (cache_idx >= cache_len - 1) begin
                        cache_idx <= 0;
                        state <= STATE_SOFTMAX;
                    end else begin
                        cache_idx <= cache_idx + 1;
                    end
                end
                
                STATE_SOFTMAX: begin
                    // Simplified softmax (pass through for hardware)
                    // In practice, use softmax_approx module
                    dim_idx <= 0;
                    state <= STATE_WEIGHTED;
                end
                
                STATE_WEIGHTED: begin
                    // Compute weighted sum of V for this head dimension
                    weighted_accum = 0;
                    for (init_i = 0; init_i < MAX_SEQ_LEN; init_i = init_i + 1) begin
                        if (init_i < cache_len) begin
                            weighted_accum = weighted_accum + 
                                $signed(attn_scores[init_i]) *
                                $signed(v_cache[init_i][head_idx * HEAD_DIM + dim_idx]);
                        end
                    end
                    attn_out[head_idx * HEAD_DIM + dim_idx] <= 
                        saturate(weighted_accum >>> FRAC_BITS);
                    
                    if (dim_idx >= HEAD_DIM - 1) begin
                        dim_idx <= 0;
                        if (head_idx >= NUM_HEADS - 1) begin
                            head_idx <= 0;
                            state <= STATE_PROJ_OUT;
                        end else begin
                            head_idx <= head_idx + 1;
                            cache_idx <= 0;
                            state <= STATE_ATTENTION;
                        end
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                STATE_PROJ_OUT: begin
                    // Output projection
                    proj_accum = 0;
                    for (init_i = 0; init_i < DIM; init_i = init_i + 1) begin
                        proj_accum = proj_accum + ternary_mac_single(
                            attn_out[init_i],
                            w_o[(dim_idx * DIM + init_i) * 2 +: 2]
                        );
                    end
                    y_out[dim_idx*ACT_WIDTH +: ACT_WIDTH] <= saturate(proj_accum);
                    
                    if (dim_idx >= DIM - 1) begin
                        dim_idx <= 0;
                        state <= STATE_OUTPUT;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                STATE_OUTPUT: begin
                    valid_out <= 1'b1;
                    
                    if (ready_out) begin
                        valid_out <= 1'b0;
                        state <= STATE_IDLE;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Saturation function
    // =========================================================================
    
    function signed [ACT_WIDTH-1:0] saturate;
        input signed [ACC_WIDTH-1:0] val;
        begin
            if (val > $signed({{(ACC_WIDTH-ACT_WIDTH+1){1'b0}}, {(ACT_WIDTH-1){1'b1}}}))
                saturate = {1'b0, {(ACT_WIDTH-1){1'b1}}};
            else if (val < $signed({{(ACC_WIDTH-ACT_WIDTH+1){1'b1}}, {(ACT_WIDTH-1){1'b0}}}))
                saturate = {1'b1, {(ACT_WIDTH-1){1'b0}}};
            else
                saturate = val[ACT_WIDTH-1:0];
        end
    endfunction

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module llm_attention_tb;
    parameter DIM = 64;
    parameter NUM_HEADS = 4;
    parameter HEAD_DIM = 16;
    parameter MAX_SEQ_LEN = 32;
    parameter KV_HEADS = 4;
    parameter ACT_WIDTH = 8;
    
    reg clk, rst_n;
    reg [DIM*ACT_WIDTH-1:0] x_in;
    reg [$clog2(MAX_SEQ_LEN)-1:0] position;
    reg valid_in;
    wire ready_in;
    reg cache_clear;
    reg [DIM*DIM*2-1:0] w_q, w_o;
    reg [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] w_k, w_v;
    reg [HEAD_DIM*ACT_WIDTH-1:0] rope_cos, rope_sin;
    wire [DIM*ACT_WIDTH-1:0] y_out;
    wire valid_out;
    reg ready_out;
    
    llm_attention #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .KV_HEADS(KV_HEADS),
        .ACT_WIDTH(ACT_WIDTH)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i, j;
    
    initial begin
        $display("LLM Attention Testbench");
        $display("=======================");
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        position = 0;
        valid_in = 0;
        cache_clear = 0;
        ready_out = 1;
        
        // Initialize weights
        w_q = {(DIM*DIM){2'b01}};
        w_k = {(DIM*KV_HEADS*HEAD_DIM){2'b01}};
        w_v = {(DIM*KV_HEADS*HEAD_DIM){2'b01}};
        w_o = {(DIM*DIM){2'b01}};
        rope_cos = {HEAD_DIM{8'd16}};  // 1.0
        rope_sin = 0;                   // 0.0
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Clear cache
        cache_clear = 1;
        @(posedge clk);
        cache_clear = 0;
        
        // Process several tokens
        for (i = 0; i < 5; i = i + 1) begin
            // Wait for ready
            while (!ready_in) @(posedge clk);
            
            // Create input
            for (j = 0; j < DIM; j = j + 1) begin
                x_in[j*ACT_WIDTH +: ACT_WIDTH] = (i + j) % 32;
            end
            position = i;
            valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            
            // Wait for output
            while (!valid_out) @(posedge clk);
            $display("Token %0d output received", i);
        end
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
