// =============================================================================
// SiLens LLM Decoder Top (SmolLM2-135M Architecture)
// =============================================================================
// Complete language model decoder pipeline:
//   1. Input embedding (from projector or token lookup)
//   2. Transformer decoder blocks (30 layers)
//   3. RMS normalization
//   4. LM head (output projection to vocabulary)
//
// Architecture details:
//   - Hidden dimension: 576
//   - Layers: 30
//   - Attention heads: 9
//   - KV heads: 9 (MHA, not GQA)
//   - MLP dimension: 1536 (2.67x)
//   - Vocabulary: 49152
//   - Max sequence: 2048
//
// Total parameters: 135M (ternary hardwired)
// KV cache stored in external DDR3
//
// Target: SkyWater SKY130, ~400mm² area
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module llm_decoder_top #(
    parameter EMBED_DIM      = 576,
    parameter NUM_LAYERS     = 30,
    parameter NUM_HEADS      = 9,
    parameter NUM_KV_HEADS   = 9,
    parameter MLP_DIM        = 1536,
    parameter VOCAB_SIZE     = 49152,
    parameter MAX_SEQ_LEN    = 2048,
    parameter ACT_WIDTH      = 8,
    parameter ACC_WIDTH      = 32,
    parameter PARALLEL       = 64,
    parameter MEM_ADDR_WIDTH = 28,
    parameter MEM_DATA_WIDTH = 512
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Control
    input  wire                         prefill_mode,
    input  wire                         decode_mode,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0] seq_pos,
    
    // Input embedding
    input  wire [EMBED_DIM*ACT_WIDTH-1:0] embed_in,
    input  wire                         embed_valid,
    output wire                         embed_ready,
    
    // Output logits (argmax token ID)
    output wire [$clog2(VOCAB_SIZE)-1:0] logits_out,
    output wire                         logits_valid,
    input  wire                         logits_ready,
    
    // KV cache memory interface
    output reg  [MEM_ADDR_WIDTH-1:0]    mem_addr,
    output reg  [MEM_DATA_WIDTH-1:0]    mem_wdata,
    input  wire [MEM_DATA_WIDTH-1:0]    mem_rdata,
    output reg                          mem_rd,
    output reg                          mem_wr,
    input  wire                         mem_ready
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    
    localparam HEAD_DIM = EMBED_DIM / NUM_HEADS;  // 64
    localparam KV_SIZE_PER_LAYER = MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * 2;  // K and V
    // Per layer: 2048 × 9 × 64 × 2 = 2.25MB
    // Total: 30 × 2.25MB = 67.5MB for KV cache
    
    // =========================================================================
    // State Machine
    // =========================================================================
    
    localparam ST_IDLE       = 4'd0;
    localparam ST_RMS_PRE    = 4'd1;
    localparam ST_QKV_PROJ   = 4'd2;
    localparam ST_KV_CACHE   = 4'd3;
    localparam ST_ATTENTION  = 4'd4;
    localparam ST_ATTN_OUT   = 4'd5;
    localparam ST_RESIDUAL1  = 4'd6;
    localparam ST_RMS_POST   = 4'd7;
    localparam ST_MLP_UP     = 4'd8;
    localparam ST_MLP_GATE   = 4'd9;
    localparam ST_MLP_DOWN   = 4'd10;
    localparam ST_RESIDUAL2  = 4'd11;
    localparam ST_FINAL_NORM = 4'd12;
    localparam ST_LM_HEAD    = 4'd13;
    localparam ST_OUTPUT     = 4'd14;
    
    reg [3:0] state;
    reg [$clog2(NUM_LAYERS)-1:0] layer_cnt;
    reg [$clog2(MAX_SEQ_LEN)-1:0] attn_pos;  // Position in attention computation
    
    // =========================================================================
    // Working Registers
    // =========================================================================
    
    // Hidden state (576 × 8-bit = 576 bytes per token)
    reg [EMBED_DIM*ACT_WIDTH-1:0] hidden_state;
    reg [EMBED_DIM*ACT_WIDTH-1:0] residual;
    
    // QKV projections
    reg [EMBED_DIM*ACT_WIDTH-1:0] q_proj;
    reg [EMBED_DIM*ACT_WIDTH-1:0] k_proj;
    reg [EMBED_DIM*ACT_WIDTH-1:0] v_proj;
    
    // Attention output
    reg [EMBED_DIM*ACT_WIDTH-1:0] attn_out;
    
    // MLP intermediates
    reg [MLP_DIM*ACT_WIDTH-1:0] mlp_up;
    reg [MLP_DIM*ACT_WIDTH-1:0] mlp_gate;
    
    // =========================================================================
    // Weight ROM Interfaces (Hardwired)
    // =========================================================================
    
    // Each layer has:
    // - input_layernorm weights (576)
    // - q_proj weights (576 × 576 × 2-bit = 82KB)
    // - k_proj weights (576 × 576 × 2-bit = 82KB)
    // - v_proj weights (576 × 576 × 2-bit = 82KB)
    // - o_proj weights (576 × 576 × 2-bit = 82KB)
    // - post_attention_layernorm weights (576)
    // - gate_proj weights (576 × 1536 × 2-bit = 221KB)
    // - up_proj weights (576 × 1536 × 2-bit = 221KB)
    // - down_proj weights (1536 × 576 × 2-bit = 221KB)
    // Total per layer: ~1MB
    // 30 layers = ~30MB for LLM weights (ternary)
    
    // Plus embedding table: 49152 × 576 × 8-bit = 27MB
    // Plus LM head: 576 × 49152 × 2-bit = 7MB (tied with embedding)
    
    // Hardwired weight signals (actual impl would be ROM)
    wire [EMBED_DIM*EMBED_DIM*2-1:0] q_weights;
    wire [EMBED_DIM*EMBED_DIM*2-1:0] k_weights;
    wire [EMBED_DIM*EMBED_DIM*2-1:0] v_weights;
    wire [EMBED_DIM*EMBED_DIM*2-1:0] o_weights;
    wire [EMBED_DIM*MLP_DIM*2-1:0] gate_weights;
    wire [EMBED_DIM*MLP_DIM*2-1:0] up_weights;
    wire [MLP_DIM*EMBED_DIM*2-1:0] down_weights;
    wire [EMBED_DIM*VOCAB_SIZE*2-1:0] lm_head_weights;
    
    // Placeholder assignments
    assign q_weights = {(EMBED_DIM*EMBED_DIM){2'b01}};
    assign k_weights = {(EMBED_DIM*EMBED_DIM){2'b01}};
    assign v_weights = {(EMBED_DIM*EMBED_DIM){2'b01}};
    assign o_weights = {(EMBED_DIM*EMBED_DIM){2'b01}};
    assign gate_weights = {(EMBED_DIM*MLP_DIM){2'b01}};
    assign up_weights = {(EMBED_DIM*MLP_DIM){2'b01}};
    assign down_weights = {(MLP_DIM*EMBED_DIM){2'b01}};
    assign lm_head_weights = {(EMBED_DIM*VOCAB_SIZE){2'b01}};
    
    // =========================================================================
    // RMS Normalization
    // =========================================================================
    
    wire [EMBED_DIM*ACT_WIDTH-1:0] rms_in;
    wire [EMBED_DIM*ACT_WIDTH-1:0] rms_out;
    wire rms_valid_in;
    wire rms_valid_out;
    wire rms_ready_in;
    wire rms_ready_out;
    
    // Default gamma = 1.0 (scaled to fixed-point)
    wire [EMBED_DIM*ACT_WIDTH-1:0] rms_gamma;
    assign rms_gamma = {EMBED_DIM{8'd16}};  // 1.0 in Q4.4 format
    
    assign rms_in = (state == ST_RMS_PRE || state == ST_FINAL_NORM) ? hidden_state :
                    (state == ST_RMS_POST) ? residual : hidden_state;
    assign rms_valid_in = (state == ST_RMS_PRE || state == ST_RMS_POST || state == ST_FINAL_NORM);
    assign rms_ready_out = 1'b1;  // Always ready to receive
    
    rms_norm #(
        .DIM(EMBED_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_rms_norm (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(rms_in),
        .valid_in(rms_valid_in),
        .ready_in(rms_ready_in),
        .gamma(rms_gamma),
        .y_out(rms_out),
        .valid_out(rms_valid_out),
        .ready_out(rms_ready_out)
    );
    
    // =========================================================================
    // Ternary Matrix Multiply Units
    // =========================================================================
    
    // Shared compute unit for projections
    reg [EMBED_DIM*ACT_WIDTH-1:0] matmul_x;
    reg [EMBED_DIM*EMBED_DIM*2-1:0] matmul_w;
    reg matmul_start;
    wire [EMBED_DIM*ACT_WIDTH-1:0] matmul_y;
    wire matmul_done;
    
    ternary_matmul_engine #(
        .IN_DIM(EMBED_DIM),
        .OUT_DIM(EMBED_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) u_matmul (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(matmul_x),
        .weights(matmul_w),
        .start(matmul_start),
        .y_out(matmul_y),
        .done(matmul_done)
    );
    
    // =========================================================================
    // KV Cache Memory Access
    // =========================================================================
    
    // KV cache address calculation
    // Layout: [layer][kv_type][seq_pos][head][head_dim]
    // layer: 0-29
    // kv_type: 0=K, 1=V
    // seq_pos: 0-2047
    // head: 0-8
    // head_dim: 0-63
    
    wire [MEM_ADDR_WIDTH-1:0] kv_base_addr;
    assign kv_base_addr = (layer_cnt * KV_SIZE_PER_LAYER) >> 6;  // Word-aligned
    
    reg kv_cache_busy;
    reg kv_write_mode;
    reg [$clog2(NUM_KV_HEADS)-1:0] kv_head_cnt;
    
    // =========================================================================
    // Attention Computation
    // =========================================================================
    
    // Simplified attention (actual impl needs softmax, score accumulation)
    reg [NUM_HEADS*ACT_WIDTH-1:0] attn_scores;
    reg attn_busy;
    
    // =========================================================================
    // SiLU Activation (for gated MLP)
    // =========================================================================
    
    wire [MLP_DIM*ACT_WIDTH-1:0] silu_in;
    wire [MLP_DIM*ACT_WIDTH-1:0] silu_out;
    
    assign silu_in = mlp_gate;
    
    // SiLU(x) = x * sigmoid(x) ≈ x * (x > 0 ? 1 : exp(x)/(1+exp(x)))
    // Simplified piecewise linear approximation
    genvar i;
    generate
        for (i = 0; i < MLP_DIM; i = i + 1) begin : silu_approx
            wire signed [ACT_WIDTH-1:0] x_val = silu_in[i*ACT_WIDTH +: ACT_WIDTH];
            wire signed [2*ACT_WIDTH-1:0] silu_result;
            
            // Piecewise linear SiLU approximation
            assign silu_result = (x_val >= 0) ? x_val :  // x >= 0: ~x
                                 (x_val > -64) ? (x_val * x_val) >>> 7 :  // Small negative
                                 0;  // Large negative -> 0
            
            assign silu_out[i*ACT_WIDTH +: ACT_WIDTH] = silu_result[ACT_WIDTH-1:0];
        end
    endgenerate
    
    // =========================================================================
    // LM Head (Output Projection)
    // =========================================================================
    
    // Project hidden state to vocabulary size (576 -> 49152)
    // Then argmax to get token ID
    reg [$clog2(VOCAB_SIZE)-1:0] argmax_idx;
    reg signed [ACC_WIDTH-1:0] argmax_val;
    reg lm_head_busy;
    reg [$clog2(VOCAB_SIZE)-1:0] lm_head_pos;
    
    // =========================================================================
    // Main State Machine
    // =========================================================================
    
    assign embed_ready = (state == ST_IDLE);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            layer_cnt <= 0;
            attn_pos <= 0;
            hidden_state <= 0;
            residual <= 0;
            matmul_start <= 0;
            mem_rd <= 0;
            mem_wr <= 0;
            kv_cache_busy <= 0;
            attn_busy <= 0;
            lm_head_busy <= 0;
            argmax_idx <= 0;
            argmax_val <= {ACC_WIDTH{1'b1}};  // Minimum value
        end else begin
            matmul_start <= 0;
            mem_rd <= 0;
            mem_wr <= 0;
            
            case (state)
                ST_IDLE: begin
                    layer_cnt <= 0;
                    argmax_idx <= 0;
                    argmax_val <= {1'b1, {(ACC_WIDTH-1){1'b0}}};  // Min signed
                    
                    if (embed_valid && embed_ready) begin
                        hidden_state <= embed_in;
                        state <= ST_RMS_PRE;
                    end
                end
                
                ST_RMS_PRE: begin
                    // Pre-attention RMS normalization
                    if (rms_valid_out) begin
                        residual <= hidden_state;  // Save for residual
                        hidden_state <= rms_out;
                        state <= ST_QKV_PROJ;
                    end
                end
                
                ST_QKV_PROJ: begin
                    // Compute Q, K, V projections (sequential)
                    // In actual impl, could be parallel with 3 matmul units
                    if (!matmul_start && !matmul_done) begin
                        matmul_x <= hidden_state;
                        matmul_w <= q_weights;
                        matmul_start <= 1;
                    end else if (matmul_done) begin
                        q_proj <= matmul_y;
                        // Would continue with K, V - simplified here
                        k_proj <= matmul_y;  // Placeholder
                        v_proj <= matmul_y;  // Placeholder
                        state <= ST_KV_CACHE;
                    end
                end
                
                ST_KV_CACHE: begin
                    // Write K, V to cache, read past K, V
                    if (!kv_cache_busy) begin
                        // Write current K
                        mem_addr <= kv_base_addr + (seq_pos * NUM_KV_HEADS * HEAD_DIM / 64);
                        mem_wdata <= k_proj[MEM_DATA_WIDTH-1:0];
                        mem_wr <= 1;
                        kv_cache_busy <= 1;
                    end else if (mem_ready) begin
                        // Simplified - actual impl reads all past KV
                        kv_cache_busy <= 0;
                        state <= ST_ATTENTION;
                    end
                end
                
                ST_ATTENTION: begin
                    // Compute attention (Q @ K^T @ V)
                    // Simplified - actual impl needs full attention computation
                    if (!attn_busy) begin
                        attn_busy <= 1;
                        attn_pos <= 0;
                    end else begin
                        // Placeholder - copy Q to attention output
                        attn_out <= q_proj;
                        attn_busy <= 0;
                        state <= ST_ATTN_OUT;
                    end
                end
                
                ST_ATTN_OUT: begin
                    // Output projection
                    if (!matmul_start && !matmul_done) begin
                        matmul_x <= attn_out;
                        matmul_w <= o_weights;
                        matmul_start <= 1;
                    end else if (matmul_done) begin
                        attn_out <= matmul_y;
                        state <= ST_RESIDUAL1;
                    end
                end
                
                ST_RESIDUAL1: begin
                    // Add attention residual
                    // hidden_state = residual + attn_out
                    // Saturating addition
                    hidden_state <= residual;  // Simplified
                    state <= ST_RMS_POST;
                end
                
                ST_RMS_POST: begin
                    // Post-attention RMS norm
                    if (rms_valid_out) begin
                        residual <= hidden_state;
                        hidden_state <= rms_out;
                        state <= ST_MLP_UP;
                    end
                end
                
                ST_MLP_UP: begin
                    // MLP up projection (576 -> 1536)
                    // Simplified - would use larger matmul
                    mlp_up <= {MLP_DIM*ACT_WIDTH{1'b0}};  // Placeholder
                    state <= ST_MLP_GATE;
                end
                
                ST_MLP_GATE: begin
                    // MLP gate projection with SiLU
                    mlp_gate <= {MLP_DIM*ACT_WIDTH{1'b0}};  // Placeholder
                    // Element-wise: mlp_intermediate = silu(gate) * up
                    state <= ST_MLP_DOWN;
                end
                
                ST_MLP_DOWN: begin
                    // MLP down projection (1536 -> 576)
                    hidden_state <= residual;  // Placeholder
                    state <= ST_RESIDUAL2;
                end
                
                ST_RESIDUAL2: begin
                    // Add MLP residual
                    // Check if more layers
                    if (layer_cnt < NUM_LAYERS - 1) begin
                        layer_cnt <= layer_cnt + 1;
                        state <= ST_RMS_PRE;
                    end else begin
                        state <= ST_FINAL_NORM;
                    end
                end
                
                ST_FINAL_NORM: begin
                    // Final RMS normalization
                    if (rms_valid_out) begin
                        hidden_state <= rms_out;
                        state <= ST_LM_HEAD;
                        lm_head_pos <= 0;
                        lm_head_busy <= 1;
                    end
                end
                
                ST_LM_HEAD: begin
                    // LM head projection + argmax
                    // Compute dot product with each row of lm_head_weights
                    // Track maximum for argmax
                    if (lm_head_busy) begin
                        // Simplified argmax - actual impl computes all logits
                        lm_head_pos <= lm_head_pos + 1;
                        
                        // Placeholder score
                        if (lm_head_pos == 42) begin  // Some token
                            argmax_idx <= lm_head_pos;
                            argmax_val <= 100;
                        end
                        
                        if (lm_head_pos == VOCAB_SIZE - 1) begin
                            lm_head_busy <= 0;
                            state <= ST_OUTPUT;
                        end
                    end
                end
                
                ST_OUTPUT: begin
                    // Output the generated token
                    if (logits_ready) begin
                        state <= ST_IDLE;
                    end
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Output Assignments
    // =========================================================================
    
    assign logits_out = argmax_idx;
    assign logits_valid = (state == ST_OUTPUT);

endmodule

// =============================================================================
// Ternary Matrix Multiply Engine
// =============================================================================

module ternary_matmul_engine #(
    parameter IN_DIM = 576,
    parameter OUT_DIM = 576,
    parameter ACT_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter PARALLEL = 64
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire [IN_DIM*ACT_WIDTH-1:0]  x_in,
    input  wire [IN_DIM*OUT_DIM*2-1:0]  weights,
    input  wire                         start,
    output reg  [OUT_DIM*ACT_WIDTH-1:0] y_out,
    output reg                          done
);
    // Simplified placeholder
    // Actual implementation uses parallel ternary MAC units
    
    reg busy;
    reg [3:0] cycle_cnt;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out <= 0;
            done <= 0;
            busy <= 0;
            cycle_cnt <= 0;
        end else begin
            done <= 0;
            
            if (start && !busy) begin
                busy <= 1;
                cycle_cnt <= 0;
            end else if (busy) begin
                cycle_cnt <= cycle_cnt + 1;
                
                // Simulate computation time
                if (cycle_cnt == 15) begin
                    y_out <= x_in[OUT_DIM*ACT_WIDTH-1:0];  // Passthrough placeholder
                    done <= 1;
                    busy <= 0;
                end
            end
        end
    end
endmodule

`default_nettype wire
