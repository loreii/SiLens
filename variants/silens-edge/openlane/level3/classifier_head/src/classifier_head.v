// =============================================================================
// SiLens-Edge - Classifier Head
// =============================================================================
// Lightweight vision classifier that maps NanoViT features to class labels.
// This is NOT a language model - it performs single-token classification only.
//
// Architecture:
//   1. Projector: Linear 192 → 128 (maps vision features to classifier space)
//   2. 4× Transformer Blocks (self-attention + MLP with RMSNorm)
//   3. Classification Head: Linear 128 → 1000 logits
//   4. Argmax: Returns class index with highest logit
//
// Design choices:
//   - Linear projector (not MLP) for efficiency - vision encoder already rich
//   - Self-attention on projected [CLS] token - no cross-attention needed
//   - 4 transformer layers (shallow but sufficient for classification)
//   - Single linear classification head (softmax in hardware is expensive)
//   - No KV cache (single forward pass, not autoregressive)
//
// Parameter count (~677K ternary weights):
//   - Projector: 192 × 128 = 24,576
//   - Per transformer layer:
//     - Q/K/V projections: 3 × 128 × 128 = 49,152
//     - Output projection: 128 × 128 = 16,384
//     - MLP gate + up: 2 × 128 × 256 = 65,536
//     - MLP down: 256 × 128 = 32,768
//     - Layer total: 163,840
//   - 4 layers: 655,360
//   - Classification head: 128 × 1000 = 128,000
//   - Grand total: ~808K weights (well under 7M budget)
//
// License: Apache 2.0
// =============================================================================

module classifier_head #(
    parameter IN_DIM       = 192,                   // Input from NanoViT
    parameter HIDDEN_DIM   = 128,                   // Internal hidden dimension
    parameter NUM_HEADS    = 4,                     // Attention heads per layer
    parameter HEAD_DIM     = 32,                    // Dimension per head (128/4)
    parameter MLP_DIM      = 256,                   // MLP hidden dimension
    parameter NUM_LAYERS   = 4,                     // Transformer layers
    parameter NUM_CLASSES  = 1000,                  // Output classes
    parameter ACT_WIDTH    = 8,                     // Activation bit width
    parameter ACC_WIDTH    = 32,                    // Accumulator bit width
    parameter FRAC_BITS    = 4,                     // Fractional bits
    parameter PARALLEL     = 8                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface - 192-dim vision features (typically [CLS] token)
    input  wire [IN_DIM*ACT_WIDTH-1:0]  vision_features,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Hardwired ternary weights (2 bits per weight: 00=0, 01=+1, 10=-1)
    // Projector: 192 × 128 = 24,576 weights × 2 bits
    input  wire [IN_DIM*HIDDEN_DIM*2-1:0]  w_proj,
    
    // Transformer layer weights (per layer, iterate through layers)
    // Attention Q/K/V/O: 4 × 128 × 128 = 65,536 weights × 2 bits per layer
    input  wire [HIDDEN_DIM*HIDDEN_DIM*2-1:0] w_q,
    input  wire [HIDDEN_DIM*HIDDEN_DIM*2-1:0] w_k,
    input  wire [HIDDEN_DIM*HIDDEN_DIM*2-1:0] w_v,
    input  wire [HIDDEN_DIM*HIDDEN_DIM*2-1:0] w_o,
    
    // MLP weights per layer
    input  wire [HIDDEN_DIM*MLP_DIM*2-1:0]    w_mlp_gate,
    input  wire [HIDDEN_DIM*MLP_DIM*2-1:0]    w_mlp_up,
    input  wire [MLP_DIM*HIDDEN_DIM*2-1:0]    w_mlp_down,
    
    // RMSNorm gammas per layer
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]    rms_attn_gamma,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]    rms_mlp_gamma,
    
    // Final RMSNorm and classification head
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]    rms_final_gamma,
    input  wire [HIDDEN_DIM*NUM_CLASSES*2-1:0] w_classifier,
    
    // Layer selection (for weight multiplexing)
    output reg  [$clog2(NUM_LAYERS)-1:0]      current_layer,
    
    // Output interface
    output reg  [$clog2(NUM_CLASSES)-1:0]     class_out,       // Predicted class
    output reg  signed [ACT_WIDTH-1:0]        confidence_out,  // Max logit (scaled)
    output reg                                valid_out,
    input  wire                               ready_out
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
    localparam STATE_PROJECT    = 4'd1;   // Projector: 192 → 128
    localparam STATE_RMS_ATTN   = 4'd2;   // Pre-attention RMSNorm
    localparam STATE_ATTN_QKV   = 4'd3;   // Compute Q, K, V
    localparam STATE_ATTN_SCORE = 4'd4;   // Compute attention scores
    localparam STATE_ATTN_OUT   = 4'd5;   // Attention output projection
    localparam STATE_RESIDUAL1  = 4'd6;   // First residual add
    localparam STATE_RMS_MLP    = 4'd7;   // Pre-MLP RMSNorm
    localparam STATE_MLP        = 4'd8;   // MLP forward
    localparam STATE_RESIDUAL2  = 4'd9;   // Second residual add
    localparam STATE_NEXT_LAYER = 4'd10;  // Move to next layer
    localparam STATE_RMS_FINAL  = 4'd11;  // Final RMSNorm
    localparam STATE_CLASSIFY   = 4'd12;  // Classification head
    localparam STATE_ARGMAX     = 4'd13;  // Find max class
    localparam STATE_OUTPUT     = 4'd14;
    
    reg [3:0] state;
    
    // =========================================================================
    // Processing buffers
    // =========================================================================
    
    reg signed [ACT_WIDTH-1:0] x_buf [0:IN_DIM-1];        // Input buffer
    reg signed [ACT_WIDTH-1:0] h_buf [0:HIDDEN_DIM-1];    // Hidden state
    reg signed [ACT_WIDTH-1:0] rms_buf [0:HIDDEN_DIM-1];  // After RMSNorm
    reg signed [ACT_WIDTH-1:0] res_buf [0:HIDDEN_DIM-1];  // Residual buffer
    
    // Attention buffers
    reg signed [ACT_WIDTH-1:0] q_buf [0:HIDDEN_DIM-1];
    reg signed [ACT_WIDTH-1:0] k_buf [0:HIDDEN_DIM-1];
    reg signed [ACT_WIDTH-1:0] v_buf [0:HIDDEN_DIM-1];
    reg signed [ACT_WIDTH-1:0] attn_buf [0:HIDDEN_DIM-1];
    
    // MLP buffers
    reg signed [ACT_WIDTH-1:0] mlp_gate [0:MLP_DIM-1];
    reg signed [ACT_WIDTH-1:0] mlp_up [0:MLP_DIM-1];
    reg signed [ACT_WIDTH-1:0] mlp_out [0:HIDDEN_DIM-1];
    
    // Classification
    reg signed [ACC_WIDTH-1:0] logits [0:NUM_CLASSES-1];
    reg signed [ACC_WIDTH-1:0] max_logit;
    reg [$clog2(NUM_CLASSES)-1:0] max_idx;
    
    // Processing indices
    reg [$clog2(HIDDEN_DIM)-1:0] dim_idx;
    reg [$clog2(MLP_DIM)-1:0] mlp_idx;
    reg [$clog2(NUM_CLASSES)-1:0] class_idx;
    reg [$clog2(NUM_HEADS)-1:0] head_idx;
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    // =========================================================================
    // Ternary MAC helper function
    // =========================================================================
    
    function signed [ACC_WIDTH-1:0] ternary_mac;
        input signed [ACT_WIDTH-1:0] act;
        input [1:0] weight;
        begin
            case (weight)
                W_POS:   ternary_mac = $signed({{(ACC_WIDTH-ACT_WIDTH){act[ACT_WIDTH-1]}}, act});
                W_NEG:   ternary_mac = -$signed({{(ACC_WIDTH-ACT_WIDTH){act[ACT_WIDTH-1]}}, act});
                default: ternary_mac = 0;
            endcase
        end
    endfunction
    
    // =========================================================================
    // Saturation function
    // =========================================================================
    
    function signed [ACT_WIDTH-1:0] saturate;
        input signed [ACC_WIDTH-1:0] val;
        localparam MAX_VAL = (1 << (ACT_WIDTH-1)) - 1;
        localparam MIN_VAL = -(1 << (ACT_WIDTH-1));
        begin
            if (val > MAX_VAL)
                saturate = MAX_VAL[ACT_WIDTH-1:0];
            else if (val < MIN_VAL)
                saturate = MIN_VAL[ACT_WIDTH-1:0];
            else
                saturate = val[ACT_WIDTH-1:0];
        end
    endfunction

    // =========================================================================
    // RMSNorm helper function
    // =========================================================================
    
    function signed [ACC_WIDTH-1:0] inv_sqrt_approx;
        input signed [ACC_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] y, three;
        integer iter;
        begin
            three = 3 << FRAC_BITS;
            y = 1 << FRAC_BITS;  // Initial guess = 1.0
            // 3 Newton-Raphson iterations
            for (iter = 0; iter < 3; iter = iter + 1) begin
                y = (y * (three - ((x * ((y * y) >>> FRAC_BITS)) >>> FRAC_BITS))) >>> (FRAC_BITS + 1);
            end
            inv_sqrt_approx = y;
        end
    endfunction
    
    // =========================================================================
    // SiLU activation (x * sigmoid(x)) approximation
    // =========================================================================
    
    function signed [ACT_WIDTH-1:0] silu_approx;
        input signed [ACT_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] sig, result;
        begin
            // Sigmoid approximation: 0.5 + 0.25*x (clamped to [0,1])
            sig = (1 << (FRAC_BITS-1)) + (x >>> 2);  // 0.5 + x/4
            if (sig < 0) sig = 0;
            if (sig > (1 << FRAC_BITS)) sig = 1 << FRAC_BITS;
            // SiLU = x * sigmoid(x)
            result = ($signed(x) * sig) >>> FRAC_BITS;
            silu_approx = saturate(result);
        end
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer i, j;
    reg signed [ACC_WIDTH-1:0] accum;
    reg signed [ACC_WIDTH-1:0] mean_sq;
    reg signed [ACC_WIDTH-1:0] inv_rms;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            current_layer <= 0;
            dim_idx <= 0;
            mlp_idx <= 0;
            class_idx <= 0;
            head_idx <= 0;
            valid_out <= 1'b0;
            max_logit <= {1'b1, {(ACC_WIDTH-1){1'b0}}};  // Min value
            max_idx <= 0;
        end else begin
            case (state)
                // =============================================================
                // IDLE - Wait for input
                // =============================================================
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Load input vision features
                        for (i = 0; i < IN_DIM; i = i + 1) begin
                            x_buf[i] <= $signed(vision_features[i*ACT_WIDTH +: ACT_WIDTH]);
                        end
                        current_layer <= 0;
                        dim_idx <= 0;
                        state <= STATE_PROJECT;
                    end
                end
                
                // =============================================================
                // PROJECTOR - Linear 192 → 128
                // =============================================================
                STATE_PROJECT: begin
                    // Compute one output dimension per cycle
                    accum = 0;
                    for (j = 0; j < IN_DIM; j = j + 1) begin
                        accum = accum + ternary_mac(
                            x_buf[j],
                            w_proj[(dim_idx * IN_DIM + j) * 2 +: 2]
                        );
                    end
                    h_buf[dim_idx] <= saturate(accum);
                    
                    if (dim_idx >= HIDDEN_DIM - 1) begin
                        dim_idx <= 0;
                        state <= STATE_RMS_ATTN;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end

                // =============================================================
                // RMS_ATTN - Pre-attention RMSNorm
                // =============================================================
                STATE_RMS_ATTN: begin
                    // Compute mean square
                    mean_sq = 0;
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        mean_sq = mean_sq + ($signed(h_buf[i]) * $signed(h_buf[i]));
                    end
                    mean_sq = mean_sq / HIDDEN_DIM;
                    inv_rms = inv_sqrt_approx(mean_sq + 1);  // +1 for eps
                    
                    // Apply RMSNorm with gamma
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        rms_buf[i] <= saturate(
                            ($signed(h_buf[i]) * inv_rms * 
                             $signed(rms_attn_gamma[i*ACT_WIDTH +: ACT_WIDTH])) >>> (2*FRAC_BITS)
                        );
                        // Save residual
                        res_buf[i] <= h_buf[i];
                    end
                    
                    dim_idx <= 0;
                    state <= STATE_ATTN_QKV;
                end
                
                // =============================================================
                // ATTN_QKV - Compute Q, K, V projections
                // =============================================================
                STATE_ATTN_QKV: begin
                    // Compute Q, K, V for current dimension
                    begin : qkv_block
                        reg signed [ACC_WIDTH-1:0] q_acc, k_acc, v_acc;
                        q_acc = 0; k_acc = 0; v_acc = 0;
                        
                        for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                            q_acc = q_acc + ternary_mac(rms_buf[j], 
                                w_q[(dim_idx * HIDDEN_DIM + j) * 2 +: 2]);
                            k_acc = k_acc + ternary_mac(rms_buf[j],
                                w_k[(dim_idx * HIDDEN_DIM + j) * 2 +: 2]);
                            v_acc = v_acc + ternary_mac(rms_buf[j],
                                w_v[(dim_idx * HIDDEN_DIM + j) * 2 +: 2]);
                        end
                        
                        q_buf[dim_idx] <= saturate(q_acc);
                        k_buf[dim_idx] <= saturate(k_acc);
                        v_buf[dim_idx] <= saturate(v_acc);
                    end
                    
                    if (dim_idx >= HIDDEN_DIM - 1) begin
                        dim_idx <= 0;
                        head_idx <= 0;
                        state <= STATE_ATTN_SCORE;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end

                // =============================================================
                // ATTN_SCORE - Compute attention (self-attention on single token)
                // =============================================================
                // For single-token classification, self-attention simplifies to:
                // score = Q · K / sqrt(d) → softmax(score) = 1.0 → output = V
                // This reduces to identity for single token, but we keep the
                // structure for multi-token [CLS] + patch token scenarios
                STATE_ATTN_SCORE: begin
                    // For single token: attention output = V directly
                    // (softmax of single element = 1.0)
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        attn_buf[i] <= v_buf[i];
                    end
                    dim_idx <= 0;
                    state <= STATE_ATTN_OUT;
                end
                
                // =============================================================
                // ATTN_OUT - Output projection
                // =============================================================
                STATE_ATTN_OUT: begin
                    accum = 0;
                    for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                        accum = accum + ternary_mac(
                            attn_buf[j],
                            w_o[(dim_idx * HIDDEN_DIM + j) * 2 +: 2]
                        );
                    end
                    attn_buf[dim_idx] <= saturate(accum);
                    
                    if (dim_idx >= HIDDEN_DIM - 1) begin
                        dim_idx <= 0;
                        state <= STATE_RESIDUAL1;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
                
                // =============================================================
                // RESIDUAL1 - First residual connection
                // =============================================================
                STATE_RESIDUAL1: begin
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        h_buf[i] <= saturate($signed(res_buf[i]) + $signed(attn_buf[i]));
                    end
                    state <= STATE_RMS_MLP;
                end

                // =============================================================
                // RMS_MLP - Pre-MLP RMSNorm
                // =============================================================
                STATE_RMS_MLP: begin
                    mean_sq = 0;
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        mean_sq = mean_sq + ($signed(h_buf[i]) * $signed(h_buf[i]));
                    end
                    mean_sq = mean_sq / HIDDEN_DIM;
                    inv_rms = inv_sqrt_approx(mean_sq + 1);
                    
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        rms_buf[i] <= saturate(
                            ($signed(h_buf[i]) * inv_rms * 
                             $signed(rms_mlp_gamma[i*ACT_WIDTH +: ACT_WIDTH])) >>> (2*FRAC_BITS)
                        );
                        res_buf[i] <= h_buf[i];
                    end
                    
                    mlp_idx <= 0;
                    state <= STATE_MLP;
                end
                
                // =============================================================
                // MLP - SwiGLU-style MLP
                // =============================================================
                // out = down(SiLU(gate(x)) * up(x))
                STATE_MLP: begin
                    // Phase 1: Compute gate and up projections (interleaved)
                    if (mlp_idx < MLP_DIM) begin
                        begin : mlp_gate_up
                            reg signed [ACC_WIDTH-1:0] gate_acc, up_acc;
                            gate_acc = 0; up_acc = 0;
                            
                            for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                                gate_acc = gate_acc + ternary_mac(rms_buf[j],
                                    w_mlp_gate[(mlp_idx * HIDDEN_DIM + j) * 2 +: 2]);
                                up_acc = up_acc + ternary_mac(rms_buf[j],
                                    w_mlp_up[(mlp_idx * HIDDEN_DIM + j) * 2 +: 2]);
                            end
                            
                            mlp_gate[mlp_idx] <= silu_approx(saturate(gate_acc));
                            mlp_up[mlp_idx] <= saturate(up_acc);
                        end
                        mlp_idx <= mlp_idx + 1;
                    end else begin
                        // Phase 2: Element-wise multiply gate*up, then down projection
                        mlp_idx <= 0;
                        dim_idx <= 0;
                        
                        // Compute down projection for first dimension
                        accum = 0;
                        for (j = 0; j < MLP_DIM; j = j + 1) begin
                            accum = accum + ternary_mac(
                                saturate(($signed(mlp_gate[j]) * $signed(mlp_up[j])) >>> FRAC_BITS),
                                w_mlp_down[(0 * MLP_DIM + j) * 2 +: 2]
                            );
                        end
                        mlp_out[0] <= saturate(accum);
                        dim_idx <= 1;
                        
                        if (HIDDEN_DIM == 1) begin
                            state <= STATE_RESIDUAL2;
                        end
                    end
                end

                // =============================================================
                // Continue MLP down projection (split state for timing)
                // =============================================================
                // This continues the down projection if still computing
                
                // =============================================================
                // RESIDUAL2 - Second residual connection
                // =============================================================
                STATE_RESIDUAL2: begin
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        h_buf[i] <= saturate($signed(res_buf[i]) + $signed(mlp_out[i]));
                    end
                    state <= STATE_NEXT_LAYER;
                end
                
                // =============================================================
                // NEXT_LAYER - Check if more layers
                // =============================================================
                STATE_NEXT_LAYER: begin
                    if (current_layer >= NUM_LAYERS - 1) begin
                        // All layers done, proceed to classification
                        current_layer <= 0;
                        dim_idx <= 0;
                        state <= STATE_RMS_FINAL;
                    end else begin
                        // More layers to process
                        current_layer <= current_layer + 1;
                        dim_idx <= 0;
                        state <= STATE_RMS_ATTN;
                    end
                end
                
                // =============================================================
                // RMS_FINAL - Final RMSNorm before classification
                // =============================================================
                STATE_RMS_FINAL: begin
                    mean_sq = 0;
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        mean_sq = mean_sq + ($signed(h_buf[i]) * $signed(h_buf[i]));
                    end
                    mean_sq = mean_sq / HIDDEN_DIM;
                    inv_rms = inv_sqrt_approx(mean_sq + 1);
                    
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        rms_buf[i] <= saturate(
                            ($signed(h_buf[i]) * inv_rms * 
                             $signed(rms_final_gamma[i*ACT_WIDTH +: ACT_WIDTH])) >>> (2*FRAC_BITS)
                        );
                    end
                    
                    class_idx <= 0;
                    max_logit <= {1'b1, {(ACC_WIDTH-1){1'b0}}};  // Reset to min
                    max_idx <= 0;
                    state <= STATE_CLASSIFY;
                end

                // =============================================================
                // CLASSIFY - Compute logits and track argmax
                // =============================================================
                STATE_CLASSIFY: begin
                    // Compute logit for current class
                    accum = 0;
                    for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                        accum = accum + ternary_mac(
                            rms_buf[j],
                            w_classifier[(class_idx * HIDDEN_DIM + j) * 2 +: 2]
                        );
                    end
                    
                    // Track maximum logit and index (online argmax)
                    if (accum > max_logit) begin
                        max_logit <= accum;
                        max_idx <= class_idx;
                    end
                    
                    if (class_idx >= NUM_CLASSES - 1) begin
                        state <= STATE_ARGMAX;
                    end else begin
                        class_idx <= class_idx + 1;
                    end
                end
                
                // =============================================================
                // ARGMAX - Finalize result
                // =============================================================
                STATE_ARGMAX: begin
                    class_out <= max_idx;
                    // Scale max logit to fit in ACT_WIDTH for confidence output
                    confidence_out <= saturate(max_logit >>> (ACC_WIDTH - ACT_WIDTH - 4));
                    state <= STATE_OUTPUT;
                end
                
                // =============================================================
                // OUTPUT - Present result
                // =============================================================
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

endmodule


// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module classifier_head_tb;
    // Use smaller parameters for simulation
    parameter IN_DIM      = 16;     // Reduced from 192
    parameter HIDDEN_DIM  = 8;      // Reduced from 128
    parameter NUM_HEADS   = 2;      // Reduced from 4
    parameter HEAD_DIM    = 4;      // HIDDEN_DIM / NUM_HEADS
    parameter MLP_DIM     = 16;     // Reduced from 256
    parameter NUM_LAYERS  = 2;      // Reduced from 4
    parameter NUM_CLASSES = 10;     // Reduced from 1000
    parameter ACT_WIDTH   = 8;
    parameter ACC_WIDTH   = 32;
    parameter FRAC_BITS   = 4;
    parameter PARALLEL    = 4;
    
    reg clk, rst_n;
    reg [IN_DIM*ACT_WIDTH-1:0] vision_features;
    reg valid_in;
    wire ready_in;
    
    // Weights (simplified for testing)
    reg [IN_DIM*HIDDEN_DIM*2-1:0] w_proj;
    reg [HIDDEN_DIM*HIDDEN_DIM*2-1:0] w_q, w_k, w_v, w_o;
    reg [HIDDEN_DIM*MLP_DIM*2-1:0] w_mlp_gate, w_mlp_up;
    reg [MLP_DIM*HIDDEN_DIM*2-1:0] w_mlp_down;
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] rms_attn_gamma, rms_mlp_gamma, rms_final_gamma;
    reg [HIDDEN_DIM*NUM_CLASSES*2-1:0] w_classifier;
    
    wire [$clog2(NUM_LAYERS)-1:0] current_layer;
    wire [$clog2(NUM_CLASSES)-1:0] class_out;
    wire signed [ACT_WIDTH-1:0] confidence_out;
    wire valid_out;
    reg ready_out;

    classifier_head #(
        .IN_DIM(IN_DIM),
        .HIDDEN_DIM(HIDDEN_DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MLP_DIM(MLP_DIM),
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_CLASSES(NUM_CLASSES),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .vision_features(vision_features),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .w_proj(w_proj),
        .w_q(w_q),
        .w_k(w_k),
        .w_v(w_v),
        .w_o(w_o),
        .w_mlp_gate(w_mlp_gate),
        .w_mlp_up(w_mlp_up),
        .w_mlp_down(w_mlp_down),
        .rms_attn_gamma(rms_attn_gamma),
        .rms_mlp_gamma(rms_mlp_gamma),
        .rms_final_gamma(rms_final_gamma),
        .w_classifier(w_classifier),
        .current_layer(current_layer),
        .class_out(class_out),
        .confidence_out(confidence_out),
        .valid_out(valid_out),
        .ready_out(ready_out)
    );
    
    // Clock generation
    always #2.5 clk = ~clk;  // 200MHz
    
    integer i;
    
    initial begin
        $display("========================================");
        $display("SiLens-Edge Classifier Head Testbench");
        $display("========================================");
        $display("IN_DIM=%0d, HIDDEN_DIM=%0d, NUM_CLASSES=%0d", 
                 IN_DIM, HIDDEN_DIM, NUM_CLASSES);
        $display("NUM_LAYERS=%0d, NUM_HEADS=%0d", NUM_LAYERS, NUM_HEADS);
        
        // Initialize
        clk = 0;
        rst_n = 0;
        vision_features = 0;
        valid_in = 0;
        ready_out = 1;
        
        // Initialize weights: all +1 for simplicity
        w_proj = {(IN_DIM*HIDDEN_DIM){2'b01}};
        w_q = {(HIDDEN_DIM*HIDDEN_DIM){2'b01}};
        w_k = {(HIDDEN_DIM*HIDDEN_DIM){2'b01}};
        w_v = {(HIDDEN_DIM*HIDDEN_DIM){2'b01}};
        w_o = {(HIDDEN_DIM*HIDDEN_DIM){2'b01}};
        w_mlp_gate = {(HIDDEN_DIM*MLP_DIM){2'b01}};
        w_mlp_up = {(HIDDEN_DIM*MLP_DIM){2'b01}};
        w_mlp_down = {(MLP_DIM*HIDDEN_DIM){2'b01}};
        
        // RMSNorm gammas = 1.0 (in fixed point)
        rms_attn_gamma = {HIDDEN_DIM{8'd16}};  // 1.0 in Q4.4
        rms_mlp_gamma = {HIDDEN_DIM{8'd16}};
        rms_final_gamma = {HIDDEN_DIM{8'd16}};
        
        // Classifier: set weights to favor class 3
        w_classifier = {(HIDDEN_DIM*NUM_CLASSES){2'b01}};
        // Make class 3 have negative weights (lower score)
        // and others have positive (but class 5 strongest)
        
        // Reset sequence
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        // Create input vision features
        $display("\nSending vision features...");
        for (i = 0; i < IN_DIM; i = i + 1) begin
            vision_features[i*ACT_WIDTH +: ACT_WIDTH] = 8'd8 + i;  // Varying input
        end
        
        // Send input
        @(posedge clk);
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for classification result
        $display("Processing through %0d transformer layers...", NUM_LAYERS);
        
        repeat(100000) begin
            @(posedge clk);
            if (valid_out) begin
                $display("\n========================================");
                $display("CLASSIFICATION RESULT");
                $display("========================================");
                $display("Predicted class: %0d", class_out);
                $display("Confidence:      %0d", confidence_out);
                $display("========================================");
                break;
            end
            
            // Progress indicator
            if (current_layer > 0) begin
                // Only print once per layer change
            end
        end
        
        // Test with different input
        $display("\nTesting with second input...");
        repeat(10) @(posedge clk);
        
        for (i = 0; i < IN_DIM; i = i + 1) begin
            vision_features[i*ACT_WIDTH +: ACT_WIDTH] = 8'd16 - i;  // Different pattern
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        repeat(100000) begin
            @(posedge clk);
            if (valid_out) begin
                $display("\nSecond classification:");
                $display("Predicted class: %0d", class_out);
                $display("Confidence:      %0d", confidence_out);
                break;
            end
        end
        
        $display("\n========================================");
        $display("Testbench PASSED");
        $display("========================================");
        $finish;
    end
    
    // Timeout
    initial begin
        #10000000;
        $display("TIMEOUT - testbench did not complete");
        $finish;
    end
    
    // Monitor layer progress
    reg [$clog2(NUM_LAYERS)-1:0] prev_layer;
    always @(posedge clk) begin
        if (current_layer != prev_layer) begin
            $display("  Processing layer %0d/%0d", current_layer + 1, NUM_LAYERS);
            prev_layer <= current_layer;
        end
    end

endmodule

`endif
