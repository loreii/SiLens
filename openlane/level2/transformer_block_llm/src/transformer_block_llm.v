// =============================================================================
// Transformer Block LLM - Level 2 Synthesis Block
// =============================================================================
// Complete LLM transformer layer for SmolLM2-135M architecture.
// Integrates Level 1 primitives: 2× RMS Norm, 9× Attention Heads, 1× MLP Block
//
// Architecture:
//   input → rms_norm_1 → multi_head_attention(9 heads) → +residual_1 →
//         → rms_norm_2 → mlp_block → +residual_2 → output
//
// Specifications:
//   - Hidden dimension: 576
//   - Number of heads: 9
//   - Head dimension: 64 (576 / 9)
//   - MLP hidden dimension: 1536
//
// Target: ~13mm² on SKY130 (3600µm × 3600µm)
// Reuse: 30× in LLM subsystem
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module transformer_block_llm #(
    parameter HIDDEN_DIM    = 576,      // Model hidden dimension
    parameter NUM_HEADS     = 9,        // Number of attention heads
    parameter HEAD_DIM      = 64,       // Per-head dimension (HIDDEN_DIM / NUM_HEADS)
    parameter MLP_HIDDEN    = 1536,     // MLP intermediate dimension
    parameter MAX_SEQ       = 256,      // Maximum sequence length
    parameter ACT_WIDTH     = 8,        // Activation bit width
    parameter ACC_WIDTH     = 24,       // Accumulator bit width
    parameter LAYER_BITS    = 5         // Bits for layer index (up to 32 layers)
)(
    input  wire                             clk,
    input  wire                             rst_n,
    
    // =========================================================================
    // Token streaming interface - Input
    // =========================================================================
    input  wire                             token_valid_in,
    output wire                             token_ready_in,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  token_data_in,  // 576 × 8-bit = 4608 bits
    input  wire                             token_last_in,   // Last token in sequence
    
    // =========================================================================
    // Token streaming interface - Output
    // =========================================================================
    output wire                             token_valid_out,
    input  wire                             token_ready_out,
    output wire [HIDDEN_DIM*ACT_WIDTH-1:0]  token_data_out, // 576 × 8-bit = 4608 bits
    output wire                             token_last_out,
    
    // =========================================================================
    // Control interface
    // =========================================================================
    input  wire [LAYER_BITS-1:0]            layer_idx,      // Current layer index (0-29)
    input  wire [$clog2(MAX_SEQ)-1:0]       seq_len,        // Sequence length
    input  wire [$clog2(MAX_SEQ)-1:0]       position,       // Current token position
    input  wire                             is_prefill,     // 1=prefill mode, 0=decode mode
    
    // =========================================================================
    // Weight memory interface - RMS Norm 1 (pre-attention)
    // =========================================================================
    output wire                             rms1_gamma_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    rms1_gamma_addr,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  rms1_gamma_data,
    
    // =========================================================================
    // Weight memory interface - RMS Norm 2 (pre-MLP)
    // =========================================================================
    output wire                             rms2_gamma_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    rms2_gamma_addr,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  rms2_gamma_data,
    
    // =========================================================================
    // Weight memory interface - Attention Q/K/V/O projections
    // =========================================================================
    // Q projection weights (HIDDEN_DIM × HIDDEN_DIM ternary = 576×576×2 bits)
    output wire                             attn_wq_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    attn_wq_addr,
    input  wire [HIDDEN_DIM*2-1:0]          attn_wq_data,
    
    // K projection weights
    output wire                             attn_wk_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    attn_wk_addr,
    input  wire [HIDDEN_DIM*2-1:0]          attn_wk_data,
    
    // V projection weights
    output wire                             attn_wv_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    attn_wv_addr,
    input  wire [HIDDEN_DIM*2-1:0]          attn_wv_data,
    
    // O projection weights (output projection)
    output wire                             attn_wo_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    attn_wo_addr,
    input  wire [HIDDEN_DIM*2-1:0]          attn_wo_data,
    
    // =========================================================================
    // Weight memory interface - MLP block
    // =========================================================================
    output wire                             mlp_gate_weight_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    mlp_gate_weight_addr,
    input  wire [MLP_HIDDEN*2-1:0]          mlp_gate_weight_data,
    
    output wire                             mlp_up_weight_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0]    mlp_up_weight_addr,
    input  wire [MLP_HIDDEN*2-1:0]          mlp_up_weight_data,
    
    output wire                             mlp_down_weight_rd_en,
    output wire [$clog2(MLP_HIDDEN)-1:0]    mlp_down_weight_addr,
    input  wire [HIDDEN_DIM*2-1:0]          mlp_down_weight_data,
    
    // =========================================================================
    // KV Cache memory interface (external SRAM, per head)
    // =========================================================================
    // Each head has its own K and V cache
    output wire [NUM_HEADS-1:0]                     kv_rd,      // Read enable per head
    output wire [NUM_HEADS-1:0]                     kv_wr,      // Write enable per head
    output wire [NUM_HEADS-1:0]                     kv_sel,     // 0=K, 1=V per head
    output wire [NUM_HEADS*$clog2(MAX_SEQ)-1:0]     kv_addr,    // Address per head
    output wire [NUM_HEADS*HEAD_DIM*ACT_WIDTH-1:0]  kv_wdata,   // Write data per head
    input  wire [NUM_HEADS*HEAD_DIM*ACT_WIDTH-1:0]  kv_rdata,   // Read data per head
    
    // =========================================================================
    // Status
    // =========================================================================
    output wire                             busy,
    output wire                             attn_busy,
    output wire                             mlp_busy
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam SEQ_BITS = $clog2(MAX_SEQ);
    localparam DIM_BITS = $clog2(HIDDEN_DIM);
    localparam HEAD_BITS = $clog2(NUM_HEADS);
    
    // =========================================================================
    // Pipeline state machine
    // =========================================================================
    localparam [3:0]
        ST_IDLE         = 4'd0,
        ST_RMS_NORM_1   = 4'd1,     // Pre-attention normalization
        ST_ATTN_START   = 4'd2,     // Start attention heads
        ST_ATTN_WAIT    = 4'd3,     // Wait for all heads to complete
        ST_ATTN_CONCAT  = 4'd4,     // Concatenate head outputs
        ST_ATTN_PROJ    = 4'd5,     // Output projection
        ST_RESIDUAL_1   = 4'd6,     // Add first residual
        ST_RMS_NORM_2   = 4'd7,     // Pre-MLP normalization
        ST_MLP          = 4'd8,     // MLP block
        ST_RESIDUAL_2   = 4'd9,     // Add second residual
        ST_OUTPUT       = 4'd10;    // Output token
    
    reg [3:0] state;
    
    // =========================================================================
    // Data buffers
    // =========================================================================
    
    // Input token buffer (residual path 1)
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] input_buffer;
    reg                            input_last_reg;
    
    // After RMS norm 1
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] rms1_out;
    wire                            rms1_valid;
    reg                             rms1_ready;
    
    // After attention + residual 1
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] post_attn_buffer;
    
    // After RMS norm 2
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] rms2_out;
    wire                            rms2_valid;
    reg                             rms2_ready;
    
    // MLP output
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] mlp_out;
    wire                            mlp_valid;
    wire                            mlp_ready_internal;
    
    // Output buffer
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] output_buffer;
    reg                            output_valid_reg;
    reg                            output_last_reg;
    
    // =========================================================================
    // Attention head control
    // =========================================================================
    reg                         attn_start;
    wire [NUM_HEADS-1:0]        head_busy;
    wire [NUM_HEADS-1:0]        head_done;
    reg  [NUM_HEADS-1:0]        head_q_valid;
    wire [NUM_HEADS-1:0]        head_q_ready;
    wire [NUM_HEADS-1:0]        head_out_valid;
    reg  [NUM_HEADS-1:0]        head_out_ready;
    
    // Per-head Q input (sliced from normalized hidden)
    wire [HEAD_DIM*ACT_WIDTH-1:0] head_q_data [0:NUM_HEADS-1];
    
    // Per-head output
    wire [HEAD_DIM*ACT_WIDTH-1:0] head_out_data [0:NUM_HEADS-1];
    
    // Concatenated attention output (all heads)
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] attn_concat;
    
    // =========================================================================
    // RMS Norm 1 Instance (Pre-Attention)
    // =========================================================================
    
    reg                             rms1_valid_in;
    wire                            rms1_ready_in;
    
    rms_norm_block #(
        .DIM        (HIDDEN_DIM),
        .ACT_WIDTH  (ACT_WIDTH),
        .ACC_WIDTH  (32),
        .FRAC_BITS  (8)
    ) u_rms_norm_1 (
        .clk        (clk),
        .rst_n      (rst_n),
        .x_in       (input_buffer),
        .valid_in   (rms1_valid_in),
        .ready_in   (rms1_ready_in),
        .gamma      (rms1_gamma_data),
        .y_out      (rms1_out),
        .valid_out  (rms1_valid),
        .ready_out  (rms1_ready)
    );
    
    // =========================================================================
    // Attention Heads (9 parallel instances)
    // =========================================================================
    
    genvar h;
    generate
        for (h = 0; h < NUM_HEADS; h = h + 1) begin : gen_attention_heads
            
            // Extract Q data for this head (64 elements from the 576-wide vector)
            assign head_q_data[h] = rms1_out[h*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH];
            
            // KV cache signals for this head
            wire                        head_kv_rd;
            wire                        head_kv_wr;
            wire                        head_kv_sel;
            wire [SEQ_BITS-1:0]         head_kv_addr;
            wire [HEAD_DIM*ACT_WIDTH-1:0] head_kv_wdata;
            wire [HEAD_DIM*ACT_WIDTH-1:0] head_kv_rdata;
            
            // Extract KV cache interface for this head
            assign kv_rd[h]     = head_kv_rd;
            assign kv_wr[h]     = head_kv_wr;
            assign kv_sel[h]    = head_kv_sel;
            assign kv_addr[h*SEQ_BITS +: SEQ_BITS] = head_kv_addr;
            assign kv_wdata[h*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH] = head_kv_wdata;
            assign head_kv_rdata = kv_rdata[h*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH];
            
            attention_head #(
                .HEAD_DIM   (HEAD_DIM),
                .MAX_SEQ    (MAX_SEQ),
                .ACT_WIDTH  (ACT_WIDTH),
                .ACC_WIDTH  (ACC_WIDTH),
                .SCORE_WIDTH(16)
            ) u_attention_head (
                .clk            (clk),
                .rst_n          (rst_n),
                
                // Control
                .start          (attn_start),
                .is_prefill     (is_prefill),
                .seq_len        (seq_len),
                .query_pos      (position),
                .busy           (head_busy[h]),
                .done           (head_done[h]),
                
                // Q input
                .q_valid        (head_q_valid[h]),
                .q_data         (head_q_data[h]),
                .q_ready        (head_q_ready[h]),
                
                // Projection (disabled, we do full projection externally)
                .use_projection (1'b0),
                .w_q            ({HEAD_DIM*2{1'b0}}),
                .w_k            ({HEAD_DIM*2{1'b0}}),
                .w_v            ({HEAD_DIM*2{1'b0}}),
                
                // KV Cache
                .kv_addr        (head_kv_addr),
                .kv_rdata       (head_kv_rdata),
                .kv_wdata       (head_kv_wdata),
                .kv_rd          (head_kv_rd),
                .kv_wr          (head_kv_wr),
                .kv_sel         (head_kv_sel),
                
                // Output
                .out_valid      (head_out_valid[h]),
                .out_data       (head_out_data[h]),
                .out_ready      (head_out_ready[h])
            );
            
            // Concatenate head output to full attention output
            assign attn_concat[h*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH] = head_out_data[h];
            
        end
    endgenerate
    
    // =========================================================================
    // RMS Norm 2 Instance (Pre-MLP)
    // =========================================================================
    
    reg                             rms2_valid_in;
    wire                            rms2_ready_in;
    
    rms_norm_block #(
        .DIM        (HIDDEN_DIM),
        .ACT_WIDTH  (ACT_WIDTH),
        .ACC_WIDTH  (32),
        .FRAC_BITS  (8)
    ) u_rms_norm_2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .x_in       (post_attn_buffer),
        .valid_in   (rms2_valid_in),
        .ready_in   (rms2_ready_in),
        .gamma      (rms2_gamma_data),
        .y_out      (rms2_out),
        .valid_out  (rms2_valid),
        .ready_out  (rms2_ready)
    );
    
    // =========================================================================
    // MLP Block Instance
    // =========================================================================
    
    reg                             mlp_token_valid_in;
    wire                            mlp_token_ready_in;
    wire                            mlp_token_last_out;
    
    mlp_block #(
        .IN_DIM         (HIDDEN_DIM),
        .HIDDEN_DIM     (MLP_HIDDEN),
        .ACT_WIDTH      (ACT_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .SILU_LUT_DEPTH (256)
    ) u_mlp_block (
        .clk                    (clk),
        .rst_n                  (rst_n),
        
        // Token input
        .token_valid_in         (mlp_token_valid_in),
        .token_ready_in         (mlp_token_ready_in),
        .token_data_in          (rms2_out),
        .token_last_in          (input_last_reg),
        
        // Token output
        .token_valid_out        (mlp_valid),
        .token_ready_out        (mlp_ready_internal),
        .token_data_out         (mlp_out),
        .token_last_out         (mlp_token_last_out),
        
        // Weight interfaces
        .gate_weight_rd_en      (mlp_gate_weight_rd_en),
        .gate_weight_addr       (mlp_gate_weight_addr),
        .gate_weight_data       (mlp_gate_weight_data),
        
        .up_weight_rd_en        (mlp_up_weight_rd_en),
        .up_weight_addr         (mlp_up_weight_addr),
        .up_weight_data         (mlp_up_weight_data),
        
        .down_weight_rd_en      (mlp_down_weight_rd_en),
        .down_weight_addr       (mlp_down_weight_addr),
        .down_weight_data       (mlp_down_weight_data),
        
        .busy                   (mlp_busy)
    );
    
    assign mlp_ready_internal = (state == ST_MLP);
    
    // =========================================================================
    // Residual Addition Functions
    // =========================================================================
    
    // Saturating signed addition for residual connections
    function automatic signed [ACT_WIDTH-1:0] saturate_add;
        input signed [ACT_WIDTH-1:0] a;
        input signed [ACT_WIDTH-1:0] b;
        reg signed [ACT_WIDTH:0] sum;
        begin
            sum = $signed({a[ACT_WIDTH-1], a}) + $signed({b[ACT_WIDTH-1], b});
            if (sum > $signed({{1{1'b0}}, {(ACT_WIDTH-1){1'b1}}}))
                saturate_add = {1'b0, {(ACT_WIDTH-1){1'b1}}};  // Max positive
            else if (sum < $signed({{2{1'b1}}, {(ACT_WIDTH-2){1'b0}}}))
                saturate_add = {1'b1, {(ACT_WIDTH-1){1'b0}}};  // Max negative
            else
                saturate_add = sum[ACT_WIDTH-1:0];
        end
    endfunction
    
    // Compute residual addition for full vector
    function automatic [HIDDEN_DIM*ACT_WIDTH-1:0] add_residual;
        input [HIDDEN_DIM*ACT_WIDTH-1:0] residual;
        input [HIDDEN_DIM*ACT_WIDTH-1:0] delta;
        integer idx;
        reg signed [ACT_WIDTH-1:0] r_elem, d_elem;
        begin
            for (idx = 0; idx < HIDDEN_DIM; idx = idx + 1) begin
                r_elem = $signed(residual[idx*ACT_WIDTH +: ACT_WIDTH]);
                d_elem = $signed(delta[idx*ACT_WIDTH +: ACT_WIDTH]);
                add_residual[idx*ACT_WIDTH +: ACT_WIDTH] = saturate_add(r_elem, d_elem);
            end
        end
    endfunction
    
    // =========================================================================
    // Main State Machine
    // =========================================================================
    
    // All heads done signal
    wire all_heads_done = &head_done;
    wire any_head_busy  = |head_busy;
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            input_buffer <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            input_last_reg <= 1'b0;
            post_attn_buffer <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            output_buffer <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            output_valid_reg <= 1'b0;
            output_last_reg <= 1'b0;
            
            rms1_valid_in <= 1'b0;
            rms1_ready <= 1'b0;
            rms2_valid_in <= 1'b0;
            rms2_ready <= 1'b0;
            mlp_token_valid_in <= 1'b0;
            
            attn_start <= 1'b0;
            head_q_valid <= {NUM_HEADS{1'b0}};
            head_out_ready <= {NUM_HEADS{1'b0}};
            
        end else begin
            // Default deassertions
            attn_start <= 1'b0;
            
            case (state)
                // ---------------------------------------------------------
                ST_IDLE: begin
                    output_valid_reg <= 1'b0;
                    rms1_valid_in <= 1'b0;
                    
                    if (token_valid_in && token_ready_in) begin
                        input_buffer <= token_data_in;
                        input_last_reg <= token_last_in;
                        state <= ST_RMS_NORM_1;
                        rms1_valid_in <= 1'b1;
                    end
                end
                
                // ---------------------------------------------------------
                ST_RMS_NORM_1: begin
                    // Wait for RMS norm 1 to complete
                    if (rms1_ready_in) begin
                        rms1_valid_in <= 1'b0;
                    end
                    
                    rms1_ready <= 1'b1;
                    
                    if (rms1_valid) begin
                        rms1_ready <= 1'b0;
                        state <= ST_ATTN_START;
                    end
                end
                
                // ---------------------------------------------------------
                ST_ATTN_START: begin
                    // Start all attention heads simultaneously
                    attn_start <= 1'b1;
                    head_q_valid <= {NUM_HEADS{1'b1}};
                    state <= ST_ATTN_WAIT;
                end
                
                // ---------------------------------------------------------
                ST_ATTN_WAIT: begin
                    // Deassert Q valid once accepted
                    for (i = 0; i < NUM_HEADS; i = i + 1) begin
                        if (head_q_ready[i]) begin
                            head_q_valid[i] <= 1'b0;
                        end
                    end
                    
                    // Wait for all heads to complete
                    if (all_heads_done && !any_head_busy) begin
                        state <= ST_ATTN_CONCAT;
                        head_out_ready <= {NUM_HEADS{1'b1}};
                    end
                end
                
                // ---------------------------------------------------------
                ST_ATTN_CONCAT: begin
                    // Collect outputs from all heads (they output in parallel)
                    if (&head_out_valid) begin
                        // All heads have valid output, concatenate them
                        // attn_concat is already assigned combinationally
                        head_out_ready <= {NUM_HEADS{1'b0}};
                        state <= ST_RESIDUAL_1;
                    end
                end
                
                // ---------------------------------------------------------
                ST_RESIDUAL_1: begin
                    // Add first residual: input + attention_output
                    post_attn_buffer <= add_residual(input_buffer, attn_concat);
                    state <= ST_RMS_NORM_2;
                    rms2_valid_in <= 1'b1;
                end
                
                // ---------------------------------------------------------
                ST_RMS_NORM_2: begin
                    // Wait for RMS norm 2 to complete
                    if (rms2_ready_in) begin
                        rms2_valid_in <= 1'b0;
                    end
                    
                    rms2_ready <= 1'b1;
                    
                    if (rms2_valid) begin
                        rms2_ready <= 1'b0;
                        state <= ST_MLP;
                        mlp_token_valid_in <= 1'b1;
                    end
                end
                
                // ---------------------------------------------------------
                ST_MLP: begin
                    // Wait for MLP to complete
                    if (mlp_token_ready_in) begin
                        mlp_token_valid_in <= 1'b0;
                    end
                    
                    if (mlp_valid) begin
                        state <= ST_RESIDUAL_2;
                    end
                end
                
                // ---------------------------------------------------------
                ST_RESIDUAL_2: begin
                    // Add second residual: post_attn + mlp_output
                    output_buffer <= add_residual(post_attn_buffer, mlp_out);
                    output_last_reg <= mlp_token_last_out;
                    state <= ST_OUTPUT;
                    output_valid_reg <= 1'b1;
                end
                
                // ---------------------------------------------------------
                ST_OUTPUT: begin
                    if (token_ready_out) begin
                        output_valid_reg <= 1'b0;
                        state <= ST_IDLE;
                    end
                end
                
                // ---------------------------------------------------------
                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end
    
    // =========================================================================
    // Output assignments
    // =========================================================================
    
    assign token_ready_in = (state == ST_IDLE);
    assign token_valid_out = output_valid_reg;
    assign token_data_out = output_buffer;
    assign token_last_out = output_last_reg;
    
    assign busy = (state != ST_IDLE);
    assign attn_busy = any_head_busy;
    
    // =========================================================================
    // Weight memory interface control
    // =========================================================================
    // RMS gamma weights are read continuously when active
    assign rms1_gamma_rd_en = (state == ST_RMS_NORM_1);
    assign rms1_gamma_addr = {DIM_BITS{1'b0}};  // Single read for full gamma
    
    assign rms2_gamma_rd_en = (state == ST_RMS_NORM_2);
    assign rms2_gamma_addr = {DIM_BITS{1'b0}};
    
    // Attention weight interfaces (directly from heads - not used in current config)
    // In full implementation, Q/K/V projections would use these
    assign attn_wq_rd_en = 1'b0;
    assign attn_wq_addr = {DIM_BITS{1'b0}};
    assign attn_wk_rd_en = 1'b0;
    assign attn_wk_addr = {DIM_BITS{1'b0}};
    assign attn_wv_rd_en = 1'b0;
    assign attn_wv_addr = {DIM_BITS{1'b0}};
    assign attn_wo_rd_en = 1'b0;
    assign attn_wo_addr = {DIM_BITS{1'b0}};

endmodule

`default_nettype wire
