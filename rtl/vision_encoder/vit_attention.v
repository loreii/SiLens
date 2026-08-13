// =============================================================================
// SiLens - Vision Transformer Multi-Head Self-Attention Module
// =============================================================================
// Implements multi-head self-attention for Vision Transformer (ViT).
//
// Architecture:
//   - 12 attention heads
//   - 768 total dimensions (64 per head)
//   - Q, K, V projections with hardwired ternary weights
//   - Scaled dot-product attention
//   - Output projection
//
// Attention formula:
//   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// For each head h (64 dims):
//   Q_h = X * W_q_h,  K_h = X * W_k_h,  V_h = X * W_v_h
//   attn_h = softmax(Q_h * K_h^T / 8) * V_h
//   output = concat(attn_heads) * W_o
//
// License: Apache 2.0
// =============================================================================

module vit_attention #(
    parameter DIM         = 768,                    // Model dimension
    parameter NUM_HEADS   = 12,                     // Number of attention heads
    parameter HEAD_DIM    = 64,                     // Dimension per head (DIM/NUM_HEADS)
    parameter SEQ_LEN     = 576,                    // Sequence length (number of patches)
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits for fixed-point
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface (one token at a time)
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,               // Input token
    input  wire [$clog2(SEQ_LEN)-1:0]   token_idx,          // Token position
    input  wire                         token_valid,
    output wire                         token_ready,
    
    // Control signals
    input  wire                         seq_start,          // Start of sequence
    input  wire                         seq_done,           // All tokens received
    
    // Hardwired ternary weights for Q, K, V, O projections
    // Each projection: DIM x DIM = 768 x 768 = 589,824 weights x 2 bits
    input  wire [DIM*DIM*2-1:0]         w_q,                // Query weights
    input  wire [DIM*DIM*2-1:0]         w_k,                // Key weights  
    input  wire [DIM*DIM*2-1:0]         w_v,                // Value weights
    input  wire [DIM*DIM*2-1:0]         w_o,                // Output weights
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0]     y_out,              // Attention output
    output reg  [$clog2(SEQ_LEN)-1:0]   out_token_idx,
    output reg                          out_valid,
    input  wire                         out_ready
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
    localparam STATE_LOAD       = 4'd1;   // Load input tokens
    localparam STATE_PROJ_QKV   = 4'd2;   // Compute Q, K, V projections
    localparam STATE_ATTENTION  = 4'd3;   // Compute attention scores
    localparam STATE_SOFTMAX    = 4'd4;   // Apply softmax
    localparam STATE_WEIGHTED   = 4'd5;   // Compute weighted sum of V
    localparam STATE_PROJ_OUT   = 4'd6;   // Output projection
    localparam STATE_OUTPUT     = 4'd7;   // Output results
    
    reg [3:0] state;
    
    // =========================================================================
    // Token buffers
    // =========================================================================
    
    // Input token buffer
    reg [ACT_WIDTH-1:0] x_buffer [0:SEQ_LEN-1][0:DIM-1];
    reg [$clog2(SEQ_LEN)-1:0] load_count;
    
    // Q, K, V buffers (per head)
    reg signed [ACT_WIDTH-1:0] q_buffer [0:SEQ_LEN-1][0:DIM-1];
    reg signed [ACT_WIDTH-1:0] k_buffer [0:SEQ_LEN-1][0:DIM-1];
    reg signed [ACT_WIDTH-1:0] v_buffer [0:SEQ_LEN-1][0:DIM-1];
    
    // Attention scores buffer (per head, SEQ_LEN x SEQ_LEN per head)
    // This is the largest buffer - we compute one head at a time to save area
    reg signed [ACT_WIDTH-1:0] attn_scores [0:SEQ_LEN-1];
    
    // Output buffer
    reg signed [ACT_WIDTH-1:0] attn_out [0:SEQ_LEN-1][0:DIM-1];
    
    // Processing indices
    reg [$clog2(SEQ_LEN)-1:0] seq_idx;      // Current sequence position
    reg [$clog2(NUM_HEADS)-1:0] head_idx;   // Current head being processed
    reg [$clog2(DIM)-1:0] dim_idx;          // Dimension index
    reg [$clog2(SEQ_LEN)-1:0] k_idx;        // Key index for attention
    
    // Projection iteration counter
    reg [$clog2(DIM/PARALLEL+1)-1:0] proj_iter;
    localparam NUM_PROJ_ITERS = (DIM + PARALLEL - 1) / PARALLEL;
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign token_ready = (state == STATE_LOAD);
    
    // =========================================================================
    // Ternary MAC for projections
    // =========================================================================
    
    // Compute one output dimension at a time, accumulating over input dimensions
    reg signed [ACC_WIDTH-1:0] proj_accum;
    wire signed [ACC_WIDTH-1:0] proj_partial;
    
    // Parallel MAC for projection
    reg signed [ACC_WIDTH-1:0] mac_results [0:PARALLEL-1];
    integer mac_i;
    
    always @(*) begin
        for (mac_i = 0; mac_i < PARALLEL; mac_i = mac_i + 1) begin
            mac_results[mac_i] = 0;
            if (proj_iter * PARALLEL + mac_i < DIM) begin
                // Determine which weights to use based on current projection
                // Weight index: dim_idx * DIM + (proj_iter * PARALLEL + mac_i)
            end
        end
    end
    
    // =========================================================================
    // Scale factor for attention
    // =========================================================================
    // sqrt(HEAD_DIM) = sqrt(64) = 8
    // Division by 8 = right shift by 3
    localparam SCALE_SHIFT = 3;
    
    // =========================================================================
    // Input loading
    // =========================================================================
    
    integer load_d;
    
    always @(posedge clk) begin
        if (state == STATE_LOAD && token_valid) begin
            for (load_d = 0; load_d < DIM; load_d = load_d + 1) begin
                x_buffer[token_idx][load_d] <= x_in[load_d*ACT_WIDTH +: ACT_WIDTH];
            end
        end
    end
    
    // =========================================================================
    // Ternary projection computation
    // =========================================================================
    
    // Single ternary MAC unit for computing projections
    function signed [ACC_WIDTH-1:0] ternary_mac_one;
        input [ACT_WIDTH-1:0] act;
        input [1:0] weight;
        begin
            case (weight)
                W_POS:   ternary_mac_one = $signed({1'b0, act});
                W_NEG:   ternary_mac_one = -$signed({1'b0, act});
                default: ternary_mac_one = 0;
            endcase
        end
    endfunction
    
    // Compute dot product of one row of input with one column of weights
    function signed [ACC_WIDTH-1:0] compute_projection;
        input [$clog2(SEQ_LEN)-1:0] seq;
        input [$clog2(DIM)-1:0] out_dim;
        input [DIM*DIM*2-1:0] weights;
        integer cp_i;
        reg signed [ACC_WIDTH-1:0] sum;
        begin
            sum = 0;
            for (cp_i = 0; cp_i < DIM; cp_i = cp_i + 1) begin
                sum = sum + ternary_mac_one(
                    x_buffer[seq][cp_i],
                    weights[(out_dim * DIM + cp_i) * 2 +: 2]
                );
            end
            compute_projection = sum;
        end
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer init_s, init_d;
    reg signed [ACC_WIDTH-1:0] attn_score_acc;
    reg signed [ACC_WIDTH-1:0] weighted_sum;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            load_count <= 0;
            seq_idx <= 0;
            head_idx <= 0;
            dim_idx <= 0;
            k_idx <= 0;
            proj_iter <= 0;
            out_valid <= 1'b0;
            out_token_idx <= 0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    out_valid <= 1'b0;
                    if (seq_start) begin
                        state <= STATE_LOAD;
                        load_count <= 0;
                    end
                end
                
                STATE_LOAD: begin
                    // Load tokens until sequence is done
                    if (token_valid) begin
                        load_count <= load_count + 1;
                    end
                    
                    if (seq_done) begin
                        state <= STATE_PROJ_QKV;
                        seq_idx <= 0;
                        dim_idx <= 0;
                    end
                end
                
                STATE_PROJ_QKV: begin
                    // Compute Q, K, V for current token and dimension
                    q_buffer[seq_idx][dim_idx] <= saturate(
                        compute_projection(seq_idx, dim_idx, w_q)
                    );
                    k_buffer[seq_idx][dim_idx] <= saturate(
                        compute_projection(seq_idx, dim_idx, w_k)
                    );
                    v_buffer[seq_idx][dim_idx] <= saturate(
                        compute_projection(seq_idx, dim_idx, w_v)
                    );
                    
                    // Update indices
                    if (dim_idx >= DIM - 1) begin
                        dim_idx <= 0;
                        if (seq_idx >= load_count - 1) begin
                            seq_idx <= 0;
                            head_idx <= 0;
                            k_idx <= 0;
                            state <= STATE_ATTENTION;
                        end else begin
                            seq_idx <= seq_idx + 1;
                        end
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                STATE_ATTENTION: begin
                    // Compute attention scores for current query position
                    // score[seq_idx][k_idx] = sum over head_dim (Q[seq_idx] * K[k_idx])
                    attn_score_acc = 0;
                    for (init_d = 0; init_d < HEAD_DIM; init_d = init_d + 1) begin
                        attn_score_acc = attn_score_acc + 
                            $signed(q_buffer[seq_idx][head_idx * HEAD_DIM + init_d]) *
                            $signed(k_buffer[k_idx][head_idx * HEAD_DIM + init_d]);
                    end
                    // Scale by 1/sqrt(d_k)
                    attn_scores[k_idx] <= saturate(attn_score_acc >>> SCALE_SHIFT);
                    
                    // Update indices
                    if (k_idx >= load_count - 1) begin
                        k_idx <= 0;
                        state <= STATE_SOFTMAX;
                    end else begin
                        k_idx <= k_idx + 1;
                    end
                end
                
                STATE_SOFTMAX: begin
                    // Simplified softmax: just normalize by sum
                    // In hardware, use softmax_approx module
                    // For now, pass through (implement proper softmax in integration)
                    state <= STATE_WEIGHTED;
                    dim_idx <= 0;
                end
                
                STATE_WEIGHTED: begin
                    // Compute weighted sum of V for this head dimension
                    weighted_sum = 0;
                    for (init_s = 0; init_s < SEQ_LEN; init_s = init_s + 1) begin
                        if (init_s < load_count) begin
                            weighted_sum = weighted_sum +
                                $signed(attn_scores[init_s]) *
                                $signed(v_buffer[init_s][head_idx * HEAD_DIM + dim_idx]);
                        end
                    end
                    attn_out[seq_idx][head_idx * HEAD_DIM + dim_idx] <= 
                        saturate(weighted_sum >>> FRAC_BITS);
                    
                    // Update indices
                    if (dim_idx >= HEAD_DIM - 1) begin
                        dim_idx <= 0;
                        if (head_idx >= NUM_HEADS - 1) begin
                            head_idx <= 0;
                            if (seq_idx >= load_count - 1) begin
                                state <= STATE_PROJ_OUT;
                                seq_idx <= 0;
                                dim_idx <= 0;
                            end else begin
                                seq_idx <= seq_idx + 1;
                                k_idx <= 0;
                                state <= STATE_ATTENTION;
                            end
                        end else begin
                            head_idx <= head_idx + 1;
                            k_idx <= 0;
                            state <= STATE_ATTENTION;
                        end
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                STATE_PROJ_OUT: begin
                    // Output projection: concat(heads) * W_o
                    // Compute for current token and dimension
                    begin : proj_out_block
                        reg signed [ACC_WIDTH-1:0] out_sum;
                        integer po_i;
                        out_sum = 0;
                        for (po_i = 0; po_i < DIM; po_i = po_i + 1) begin
                            out_sum = out_sum + ternary_mac_one(
                                attn_out[seq_idx][po_i],
                                w_o[(dim_idx * DIM + po_i) * 2 +: 2]
                            );
                        end
                        y_out[dim_idx*ACT_WIDTH +: ACT_WIDTH] <= saturate(out_sum);
                    end
                    
                    if (dim_idx >= DIM - 1) begin
                        dim_idx <= 0;
                        out_token_idx <= seq_idx;
                        state <= STATE_OUTPUT;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                STATE_OUTPUT: begin
                    out_valid <= 1'b1;
                    
                    if (out_ready) begin
                        out_valid <= 1'b0;
                        
                        if (seq_idx >= load_count - 1) begin
                            state <= STATE_IDLE;
                        end else begin
                            seq_idx <= seq_idx + 1;
                            dim_idx <= 0;
                            state <= STATE_PROJ_OUT;
                        end
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
                saturate = {1'b0, {(ACT_WIDTH-1){1'b1}}};  // Max positive
            else if (val < $signed({{(ACC_WIDTH-ACT_WIDTH+1){1'b1}}, {(ACT_WIDTH-1){1'b0}}}))
                saturate = {1'b1, {(ACT_WIDTH-1){1'b0}}};  // Min negative
            else
                saturate = val[ACT_WIDTH-1:0];
        end
    endfunction

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module vit_attention_tb;
    parameter DIM = 64;
    parameter NUM_HEADS = 4;
    parameter HEAD_DIM = 16;
    parameter SEQ_LEN = 8;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    
    reg clk, rst_n;
    reg [DIM*ACT_WIDTH-1:0] x_in;
    reg [$clog2(SEQ_LEN)-1:0] token_idx;
    reg token_valid;
    wire token_ready;
    reg seq_start, seq_done;
    reg [DIM*DIM*2-1:0] w_q, w_k, w_v, w_o;
    wire [DIM*ACT_WIDTH-1:0] y_out;
    wire [$clog2(SEQ_LEN)-1:0] out_token_idx;
    wire out_valid;
    reg out_ready;
    
    vit_attention #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .SEQ_LEN(SEQ_LEN),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i, j, outputs_received;
    
    initial begin
        $display("ViT Attention Testbench");
        $display("=======================");
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        token_idx = 0;
        token_valid = 0;
        seq_start = 0;
        seq_done = 0;
        out_ready = 1;
        
        // Initialize weights (identity-like for testing)
        w_q = {(DIM*DIM){2'b01}};  // All +1
        w_k = {(DIM*DIM){2'b01}};
        w_v = {(DIM*DIM){2'b01}};
        w_o = {(DIM*DIM){2'b01}};
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Start sequence
        seq_start = 1;
        @(posedge clk);
        seq_start = 0;
        
        // Send tokens
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            @(posedge clk);
            for (j = 0; j < DIM; j = j + 1) begin
                x_in[j*ACT_WIDTH +: ACT_WIDTH] = (i + j) % 16;
            end
            token_idx = i;
            token_valid = 1;
        end
        
        @(posedge clk);
        token_valid = 0;
        seq_done = 1;
        @(posedge clk);
        seq_done = 0;
        
        // Wait for outputs
        outputs_received = 0;
        repeat(10000) begin
            @(posedge clk);
            if (out_valid) begin
                outputs_received = outputs_received + 1;
                $display("Output token %0d received", out_token_idx);
            end
        end
        
        $display("Received %0d output tokens", outputs_received);
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #1000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
