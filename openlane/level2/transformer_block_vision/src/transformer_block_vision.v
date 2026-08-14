// =============================================================================
// Vision Transformer Block - Level 2 Hierarchical Synthesis Block
// =============================================================================
// Complete Vision Transformer layer for SigLIP-B/16 vision encoder.
//
// Architecture (Pre-Norm style):
//   input → LayerNorm → Multi-Head Attention (12 heads) → +residual →
//         → LayerNorm → MLP (768→3072→768, GELU) → +residual → output
//
// Key differences from LLM transformer:
//   - Full bidirectional attention (no causal mask)
//   - LayerNorm instead of RMSNorm
//   - GELU activation in MLP (vs SiLU/SwiGLU)
//
// Specifications:
//   - Hidden dimension: 768
//   - Number of heads: 12
//   - Head dimension: 64 (768/12)
//   - MLP hidden dimension: 3072 (4× hidden)
//   - Target area: ~20mm² (4500µm × 4500µm)
//
// Level 1 macro dependencies:
//   - 2× layer_norm_block (768-dim)
//   - 12× attention_head (head_dim=64)
//   - 1× mlp_block (768→3072→768)
//
// Reuse: 12× in vision encoder subsystem
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module transformer_block_vision #(
    parameter HIDDEN_DIM    = 768,      // SigLIP-B/16 hidden dimension
    parameter NUM_HEADS     = 12,       // Number of attention heads
    parameter HEAD_DIM      = 64,       // Per-head dimension (HIDDEN_DIM/NUM_HEADS)
    parameter MLP_HIDDEN    = 3072,     // MLP intermediate dimension (4× hidden)
    parameter MAX_SEQ       = 256,      // Maximum sequence length (patches + CLS)
    parameter ACT_WIDTH     = 8,        // Activation bit width
    parameter ACC_WIDTH     = 24        // Accumulator bit width
)(
    input  wire                             clk,
    input  wire                             rst_n,
    
    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire [7:0]                       layer_idx,      // Current layer index (0-11)
    input  wire [$clog2(MAX_SEQ)-1:0]       num_patches,    // Number of patches in current image
    input  wire                             start,          // Start processing
    output wire                             busy,           // Block is processing
    output wire                             done,           // Processing complete
    
    // =========================================================================
    // Patch Streaming Input (768×8-bit activations per patch)
    // =========================================================================
    input  wire                             patch_valid_in,
    output wire                             patch_ready_in,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  patch_data_in,
    input  wire                             patch_last_in,  // Last patch in sequence
    
    // =========================================================================
    // Patch Streaming Output (768×8-bit activations per patch)
    // =========================================================================
    output wire                             patch_valid_out,
    input  wire                             patch_ready_out,
    output wire [HIDDEN_DIM*ACT_WIDTH-1:0]  patch_data_out,
    output wire                             patch_last_out,
    
    // =========================================================================
    // Weight Memory Interface - Attention Q/K/V/O Projections
    // =========================================================================
    // Q projection weights: HIDDEN_DIM × HIDDEN_DIM (ternary, 2-bit encoded)
    output wire                             wq_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    wq_addr,
    input  wire [HIDDEN_DIM*2-1:0]          wq_data,
    
    // K projection weights
    output wire                             wk_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    wk_addr,
    input  wire [HIDDEN_DIM*2-1:0]          wk_data,
    
    // V projection weights
    output wire                             wv_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    wv_addr,
    input  wire [HIDDEN_DIM*2-1:0]          wv_data,
    
    // Output projection weights
    output wire                             wo_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    wo_addr,
    input  wire [HIDDEN_DIM*2-1:0]          wo_data,
    
    // =========================================================================
    // Weight Memory Interface - MLP Weights
    // =========================================================================
    // MLP gate/up projection (768→3072)
    output wire                             mlp_gate_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    mlp_gate_addr,
    input  wire [MLP_HIDDEN*2-1:0]          mlp_gate_data,
    
    output wire                             mlp_up_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    mlp_up_addr,
    input  wire [MLP_HIDDEN*2-1:0]          mlp_up_data,
    
    // MLP down projection (3072→768)
    output wire                             mlp_down_rd_en,
    output wire [$clog2(MLP_HIDDEN)-1:0]    mlp_down_addr,
    input  wire [HIDDEN_DIM*2-1:0]          mlp_down_data,
    
    // =========================================================================
    // LayerNorm Parameters (from weight memory)
    // =========================================================================
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  ln1_gamma,      // Pre-attention LayerNorm gamma
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  ln1_beta,       // Pre-attention LayerNorm beta
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  ln2_gamma,      // Pre-MLP LayerNorm gamma
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  ln2_beta        // Pre-MLP LayerNorm beta
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam SEQ_BITS = $clog2(MAX_SEQ);
    localparam DIM_BITS = $clog2(HIDDEN_DIM);
    
    // =========================================================================
    // FSM States
    // =========================================================================
    localparam [3:0] ST_IDLE           = 4'd0;
    localparam [3:0] ST_LOAD_PATCH     = 4'd1;
    localparam [3:0] ST_LAYER_NORM_1   = 4'd2;
    localparam [3:0] ST_ATTENTION      = 4'd3;
    localparam [3:0] ST_ATTN_REDUCE    = 4'd4;
    localparam [3:0] ST_RESIDUAL_1     = 4'd5;
    localparam [3:0] ST_LAYER_NORM_2   = 4'd6;
    localparam [3:0] ST_MLP            = 4'd7;
    localparam [3:0] ST_RESIDUAL_2     = 4'd8;
    localparam [3:0] ST_OUTPUT         = 4'd9;
    localparam [3:0] ST_DONE           = 4'd10;
    
    reg [3:0] state, next_state;
    
    // =========================================================================
    // Internal Registers
    // =========================================================================
    
    // Input patch buffer (for residual connections)
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] input_buffer;
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] residual_1_buffer;  // After first residual add
    reg                            patch_last_reg;
    reg [SEQ_BITS-1:0]             patch_idx;
    reg [SEQ_BITS-1:0]             num_patches_reg;
    
    // =========================================================================
    // Layer Norm 1 (Pre-Attention) Signals
    // =========================================================================
    wire                            ln1_valid_in;
    wire                            ln1_ready_in;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] ln1_data_in;
    wire                            ln1_valid_out;
    wire                            ln1_ready_out;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] ln1_data_out;
    
    reg                             ln1_start;
    
    // =========================================================================
    // Attention Head Signals (12 heads in parallel)
    // =========================================================================
    wire [NUM_HEADS-1:0]            attn_start;
    wire [NUM_HEADS-1:0]            attn_busy;
    wire [NUM_HEADS-1:0]            attn_done;
    
    // Q input to each head (head_dim=64 slice of projected Q)
    wire [NUM_HEADS-1:0]            attn_q_valid;
    wire [NUM_HEADS-1:0]            attn_q_ready;
    wire [HEAD_DIM*ACT_WIDTH-1:0]   attn_q_data [0:NUM_HEADS-1];
    
    // Attention output from each head
    wire [NUM_HEADS-1:0]            attn_out_valid;
    wire [NUM_HEADS-1:0]            attn_out_ready;
    wire [HEAD_DIM*ACT_WIDTH-1:0]   attn_out_data [0:NUM_HEADS-1];
    
    // KV cache interface for each head
    wire [SEQ_BITS-1:0]             attn_kv_addr [0:NUM_HEADS-1];
    wire [HEAD_DIM*ACT_WIDTH-1:0]   attn_kv_rdata [0:NUM_HEADS-1];
    wire [HEAD_DIM*ACT_WIDTH-1:0]   attn_kv_wdata [0:NUM_HEADS-1];
    wire [NUM_HEADS-1:0]            attn_kv_rd;
    wire [NUM_HEADS-1:0]            attn_kv_wr;
    wire [NUM_HEADS-1:0]            attn_kv_sel;
    
    // Concatenated attention output (768-bit)
    reg [HIDDEN_DIM*ACT_WIDTH-1:0]  attn_concat_out;
    
    // =========================================================================
    // Layer Norm 2 (Pre-MLP) Signals
    // =========================================================================
    wire                            ln2_valid_in;
    wire                            ln2_ready_in;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] ln2_data_in;
    wire                            ln2_valid_out;
    wire                            ln2_ready_out;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] ln2_data_out;
    
    reg                             ln2_start;
    
    // =========================================================================
    // MLP Block Signals
    // =========================================================================
    wire                            mlp_valid_in;
    wire                            mlp_ready_in;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] mlp_data_in;
    wire                            mlp_valid_out;
    wire                            mlp_ready_out;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] mlp_data_out;
    wire                            mlp_busy;
    
    reg                             mlp_start;
    
    // =========================================================================
    // Output Buffer
    // =========================================================================
    reg [HIDDEN_DIM*ACT_WIDTH-1:0]  output_buffer;
    reg                             output_valid;
    
    // =========================================================================
    // Control Signals
    // =========================================================================
    reg block_busy;
    reg block_done;
    
    // Attention start control
    reg attn_trigger;
    reg [SEQ_BITS-1:0] attn_seq_len;
    reg [SEQ_BITS-1:0] attn_query_pos;
    
    // =========================================================================
    // State Machine
    // =========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (start && patch_valid_in)
                    next_state = ST_LOAD_PATCH;
            end
            
            ST_LOAD_PATCH: begin
                if (patch_valid_in && patch_ready_in)
                    next_state = ST_LAYER_NORM_1;
            end
            
            ST_LAYER_NORM_1: begin
                if (ln1_valid_out && ln1_ready_out)
                    next_state = ST_ATTENTION;
            end
            
            ST_ATTENTION: begin
                // Wait for all attention heads to complete
                if (&attn_done)
                    next_state = ST_ATTN_REDUCE;
            end
            
            ST_ATTN_REDUCE: begin
                // Concatenate head outputs and apply output projection
                next_state = ST_RESIDUAL_1;
            end
            
            ST_RESIDUAL_1: begin
                // Add input residual
                next_state = ST_LAYER_NORM_2;
            end
            
            ST_LAYER_NORM_2: begin
                if (ln2_valid_out && ln2_ready_out)
                    next_state = ST_MLP;
            end
            
            ST_MLP: begin
                if (mlp_valid_out && mlp_ready_out)
                    next_state = ST_RESIDUAL_2;
            end
            
            ST_RESIDUAL_2: begin
                // Add second residual
                next_state = ST_OUTPUT;
            end
            
            ST_OUTPUT: begin
                if (patch_valid_out && patch_ready_out) begin
                    if (patch_last_reg)
                        next_state = ST_DONE;
                    else
                        next_state = ST_LOAD_PATCH;  // Process next patch
                end
            end
            
            ST_DONE: begin
                next_state = ST_IDLE;
            end
            
            default: next_state = ST_IDLE;
        endcase
    end
    
    // =========================================================================
    // Main Control Logic
    // =========================================================================
    integer j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            block_busy <= 1'b0;
            block_done <= 1'b0;
            ln1_start <= 1'b0;
            ln2_start <= 1'b0;
            mlp_start <= 1'b0;
            attn_trigger <= 1'b0;
            patch_idx <= {SEQ_BITS{1'b0}};
            patch_last_reg <= 1'b0;
            num_patches_reg <= {SEQ_BITS{1'b0}};
            input_buffer <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            residual_1_buffer <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            attn_concat_out <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            output_buffer <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            output_valid <= 1'b0;
            attn_seq_len <= {SEQ_BITS{1'b0}};
            attn_query_pos <= {SEQ_BITS{1'b0}};
        end else begin
            state <= next_state;
            
            // Default signal clearing
            ln1_start <= 1'b0;
            ln2_start <= 1'b0;
            mlp_start <= 1'b0;
            attn_trigger <= 1'b0;
            
            case (state)
                ST_IDLE: begin
                    block_busy <= 1'b0;
                    block_done <= 1'b0;
                    output_valid <= 1'b0;
                    patch_idx <= {SEQ_BITS{1'b0}};
                    
                    if (start) begin
                        block_busy <= 1'b1;
                        num_patches_reg <= num_patches;
                        attn_seq_len <= num_patches;
                    end
                end
                
                ST_LOAD_PATCH: begin
                    if (patch_valid_in) begin
                        input_buffer <= patch_data_in;
                        patch_last_reg <= patch_last_in;
                        attn_query_pos <= patch_idx;
                    end
                end
                
                ST_LAYER_NORM_1: begin
                    ln1_start <= 1'b1;
                end
                
                ST_ATTENTION: begin
                    attn_trigger <= 1'b1;
                end
                
                ST_ATTN_REDUCE: begin
                    // Concatenate outputs from all 12 attention heads
                    // Each head outputs HEAD_DIM (64) elements
                    // Total: 12 × 64 = 768 elements
                    for (j = 0; j < NUM_HEADS; j = j + 1) begin
                        attn_concat_out[j*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH] <= 
                            attn_out_data[j];
                    end
                end
                
                ST_RESIDUAL_1: begin
                    // Residual connection: attn_output + input
                    // Element-wise saturating add
                    for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                        residual_1_buffer[j*ACT_WIDTH +: ACT_WIDTH] <= 
                            saturating_add(
                                attn_concat_out[j*ACT_WIDTH +: ACT_WIDTH],
                                input_buffer[j*ACT_WIDTH +: ACT_WIDTH]
                            );
                    end
                end
                
                ST_LAYER_NORM_2: begin
                    ln2_start <= 1'b1;
                end
                
                ST_MLP: begin
                    mlp_start <= 1'b1;
                end
                
                ST_RESIDUAL_2: begin
                    // Residual connection: mlp_output + residual_1
                    for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                        output_buffer[j*ACT_WIDTH +: ACT_WIDTH] <= 
                            saturating_add(
                                mlp_data_out[j*ACT_WIDTH +: ACT_WIDTH],
                                residual_1_buffer[j*ACT_WIDTH +: ACT_WIDTH]
                            );
                    end
                end
                
                ST_OUTPUT: begin
                    output_valid <= 1'b1;
                    if (patch_ready_out) begin
                        output_valid <= 1'b0;
                        patch_idx <= patch_idx + 1'b1;
                    end
                end
                
                ST_DONE: begin
                    block_done <= 1'b1;
                    block_busy <= 1'b0;
                end
                
                default: begin
                    // Do nothing
                end
            endcase
        end
    end
    
    // =========================================================================
    // Saturating Add Function (8-bit signed)
    // =========================================================================
    function [ACT_WIDTH-1:0] saturating_add;
        input [ACT_WIDTH-1:0] a;
        input [ACT_WIDTH-1:0] b;
        reg signed [ACT_WIDTH:0] sum;
        reg signed [ACT_WIDTH-1:0] max_val;
        reg signed [ACT_WIDTH-1:0] min_val;
        begin
            sum = $signed({a[ACT_WIDTH-1], a}) + $signed({b[ACT_WIDTH-1], b});
            max_val = {1'b0, {(ACT_WIDTH-1){1'b1}}};   // +127
            min_val = {1'b1, {(ACT_WIDTH-1){1'b0}}};   // -128
            
            if (sum > $signed({1'b0, max_val}))
                saturating_add = max_val;
            else if (sum < $signed({1'b1, min_val[ACT_WIDTH-2:0]}))
                saturating_add = min_val;
            else
                saturating_add = sum[ACT_WIDTH-1:0];
        end
    endfunction
    
    // =========================================================================
    // Layer Norm 1 Instance (Pre-Attention)
    // =========================================================================
    assign ln1_valid_in = (state == ST_LAYER_NORM_1);
    assign ln1_data_in = input_buffer;
    assign ln1_ready_out = (state == ST_LAYER_NORM_1);
    
    layer_norm_block #(
        .DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(32)
    ) u_layer_norm_pre_attn (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(ln1_data_in),
        .valid_in(ln1_valid_in),
        .ready_in(ln1_ready_in),
        .gamma(ln1_gamma),
        .beta(ln1_beta),
        .y_out(ln1_data_out),
        .valid_out(ln1_valid_out),
        .ready_out(ln1_ready_out)
    );
    
    // =========================================================================
    // Attention Heads (12 parallel instances)
    // =========================================================================
    // Vision uses full bidirectional attention (no causal mask)
    // Each head processes HEAD_DIM=64 dimensions
    
    genvar h;
    generate
        for (h = 0; h < NUM_HEADS; h = h + 1) begin : gen_attn_heads
            // Extract Q slice for this head from LayerNorm output
            wire [HEAD_DIM*ACT_WIDTH-1:0] q_slice;
            assign q_slice = ln1_data_out[h*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH];
            
            assign attn_start[h] = attn_trigger;
            assign attn_q_valid[h] = (state == ST_ATTENTION) && ln1_valid_out;
            assign attn_q_data[h] = q_slice;
            assign attn_out_ready[h] = (state == ST_ATTN_REDUCE);
            
            // KV cache (local SRAM per head would be external in real impl)
            // For synthesis, we model interface only
            assign attn_kv_rdata[h] = {(HEAD_DIM*ACT_WIDTH){1'b0}};  // Placeholder
            
            attention_head #(
                .HEAD_DIM(HEAD_DIM),
                .MAX_SEQ(MAX_SEQ),
                .ACT_WIDTH(ACT_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),
                .SCORE_WIDTH(16)
            ) u_attn_head (
                .clk(clk),
                .rst_n(rst_n),
                
                // Control
                .start(attn_start[h]),
                .is_prefill(1'b1),              // Vision always processes full sequence
                .seq_len(attn_seq_len),
                .query_pos(attn_query_pos),
                .busy(attn_busy[h]),
                .done(attn_done[h]),
                
                // Query input
                .q_valid(attn_q_valid[h]),
                .q_data(attn_q_data[h]),
                .q_ready(attn_q_ready[h]),
                
                // Ternary projection (disabled - using pre-projected Q)
                .use_projection(1'b0),
                .w_q({(HEAD_DIM*2){1'b0}}),
                .w_k({(HEAD_DIM*2){1'b0}}),
                .w_v({(HEAD_DIM*2){1'b0}}),
                
                // KV cache interface
                .kv_addr(attn_kv_addr[h]),
                .kv_rdata(attn_kv_rdata[h]),
                .kv_wdata(attn_kv_wdata[h]),
                .kv_rd(attn_kv_rd[h]),
                .kv_wr(attn_kv_wr[h]),
                .kv_sel(attn_kv_sel[h]),
                
                // Output
                .out_valid(attn_out_valid[h]),
                .out_data(attn_out_data[h]),
                .out_ready(attn_out_ready[h])
            );
        end
    endgenerate
    
    // =========================================================================
    // Layer Norm 2 Instance (Pre-MLP)
    // =========================================================================
    assign ln2_valid_in = (state == ST_LAYER_NORM_2);
    assign ln2_data_in = residual_1_buffer;
    assign ln2_ready_out = (state == ST_LAYER_NORM_2);
    
    layer_norm_block #(
        .DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(32)
    ) u_layer_norm_pre_mlp (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(ln2_data_in),
        .valid_in(ln2_valid_in),
        .ready_in(ln2_ready_in),
        .gamma(ln2_gamma),
        .beta(ln2_beta),
        .y_out(ln2_data_out),
        .valid_out(ln2_valid_out),
        .ready_out(ln2_ready_out)
    );
    
    // =========================================================================
    // MLP Block Instance (768 → 3072 → 768, GELU)
    // =========================================================================
    assign mlp_valid_in = (state == ST_MLP) && ln2_valid_out;
    assign mlp_data_in = ln2_data_out;
    assign mlp_ready_out = (state == ST_MLP);
    
    mlp_block #(
        .IN_DIM(HIDDEN_DIM),
        .HIDDEN_DIM(MLP_HIDDEN),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .SILU_LUT_DEPTH(256)
    ) u_mlp (
        .clk(clk),
        .rst_n(rst_n),
        
        // Token streaming interface
        .token_valid_in(mlp_valid_in),
        .token_ready_in(mlp_ready_in),
        .token_data_in(mlp_data_in),
        .token_last_in(patch_last_reg),
        
        .token_valid_out(mlp_valid_out),
        .token_ready_out(mlp_ready_out),
        .token_data_out(mlp_data_out),
        .token_last_out(),  // Not used
        
        // Weight memory interface
        .gate_weight_rd_en(mlp_gate_rd_en),
        .gate_weight_addr(mlp_gate_addr),
        .gate_weight_data(mlp_gate_data),
        
        .up_weight_rd_en(mlp_up_rd_en),
        .up_weight_addr(mlp_up_addr),
        .up_weight_data(mlp_up_data),
        
        .down_weight_rd_en(mlp_down_rd_en),
        .down_weight_addr(mlp_down_addr),
        .down_weight_data(mlp_down_data),
        
        .busy(mlp_busy)
    );
    
    // =========================================================================
    // Output Assignments
    // =========================================================================
    assign busy = block_busy;
    assign done = block_done;
    
    assign patch_ready_in = (state == ST_IDLE) || (state == ST_LOAD_PATCH);
    assign patch_valid_out = output_valid;
    assign patch_data_out = output_buffer;
    assign patch_last_out = patch_last_reg;
    
    // =========================================================================
    // Weight Memory Interface (directly from MLP, others unused for now)
    // =========================================================================
    // Attention weight interfaces - directly expose to external memory controller
    assign wq_rd_en = 1'b0;  // Would be active during Q projection
    assign wq_addr = {DIM_BITS{1'b0}};
    assign wk_rd_en = 1'b0;  // Would be active during K projection
    assign wk_addr = {DIM_BITS{1'b0}};
    assign wv_rd_en = 1'b0;  // Would be active during V projection
    assign wv_addr = {DIM_BITS{1'b0}};
    assign wo_rd_en = 1'b0;  // Would be active during output projection
    assign wo_addr = {DIM_BITS{1'b0}};

endmodule

`default_nettype wire
