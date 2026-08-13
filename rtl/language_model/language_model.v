// =============================================================================
// SiLens - Language Model (SmolLM2-135M)
// =============================================================================
// Complete language model decoder implementing SmolLM2-135M architecture.
//
// Architecture:
//   - Token embedding: 49152 vocab x 576 dim
//   - 30 decoder transformer blocks
//   - Final RMSNorm + vocabulary projection
//   - Autoregressive generation with KV cache
//
// Parameters (SmolLM2-135M):
//   - 135M parameters
//   - Hidden dim: 576
//   - Layers: 30
//   - Heads: 9
//   - MLP expansion: ~2.67x (576 -> 1536 -> 576)
//   - Vocabulary: 49,152
//   - Max context: 8,192 tokens
//
// License: Apache 2.0
// =============================================================================

module language_model #(
    parameter DIM         = 576,                    // Model dimension
    parameter NUM_LAYERS  = 30,                     // Number of decoder blocks
    parameter NUM_HEADS   = 9,                      // Attention heads
    parameter HEAD_DIM    = 64,                     // Dimension per head
    parameter MLP_DIM     = 1536,                   // MLP hidden dimension
    parameter VOCAB_SIZE  = 49152,                  // Vocabulary size
    parameter MAX_SEQ_LEN = 8192,                   // Maximum sequence length
    parameter KV_HEADS    = 9,                      // Number of KV heads (for GQA)
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [$clog2(VOCAB_SIZE)-1:0] token_in,          // Input token ID
    input  wire                         token_valid,
    output wire                         token_ready,
    
    // Vision embedding input (from projector)
    input  wire [DIM*ACT_WIDTH-1:0]     vision_embed,       // Vision token embedding
    input  wire                         vision_valid,
    output wire                         vision_ready,
    input  wire                         is_vision_token,    // Select vision vs text input
    
    // Control signals
    input  wire                         seq_start,          // Start new sequence
    input  wire                         generate,           // Start autoregressive generation
    
    // Token embedding weights (ROM)
    input  wire [VOCAB_SIZE*DIM*ACT_WIDTH-1:0] token_embed_weights,
    
    // Block weights (per layer - in practice from weight ROM)
    input  wire [DIM*ACT_WIDTH-1:0]     rms_attn_gamma [0:NUM_LAYERS-1],
    input  wire [DIM*ACT_WIDTH-1:0]     rms_mlp_gamma  [0:NUM_LAYERS-1],
    input  wire [DIM*DIM*2-1:0]         attn_w_q  [0:NUM_LAYERS-1],
    input  wire [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] attn_w_k [0:NUM_LAYERS-1],
    input  wire [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] attn_w_v [0:NUM_LAYERS-1],
    input  wire [DIM*DIM*2-1:0]         attn_w_o  [0:NUM_LAYERS-1],
    input  wire [DIM*MLP_DIM*2-1:0]     mlp_w_gate [0:NUM_LAYERS-1],
    input  wire [DIM*MLP_DIM*2-1:0]     mlp_w_up   [0:NUM_LAYERS-1],
    input  wire [MLP_DIM*DIM*2-1:0]     mlp_w_down [0:NUM_LAYERS-1],
    
    // RoPE frequencies (precomputed)
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] rope_cos [0:MAX_SEQ_LEN-1],
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] rope_sin [0:MAX_SEQ_LEN-1],
    
    // LM head weights
    input  wire [DIM*ACT_WIDTH-1:0]     final_rms_gamma,
    input  wire [VOCAB_SIZE*DIM*2-1:0]  lm_head_weights,
    
    // Output interface
    output wire [$clog2(VOCAB_SIZE)-1:0] token_out,         // Generated token
    output wire                         token_out_valid,
    input  wire                         token_out_ready,
    
    // Status
    output wire                         busy,
    output reg  [$clog2(MAX_SEQ_LEN)-1:0] current_position
);

    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE       = 4'd0;
    localparam STATE_EMBED      = 4'd1;   // Token embedding lookup
    localparam STATE_BLOCK      = 4'd2;   // Process through decoder block
    localparam STATE_NEXT_BLOCK = 4'd3;   // Move to next block
    localparam STATE_LM_HEAD    = 4'd4;   // Apply LM head
    localparam STATE_OUTPUT     = 4'd5;   // Output generated token
    localparam STATE_AUTOGEN    = 4'd6;   // Autoregressive loop
    
    reg [3:0] state;
    
    // =========================================================================
    // Layer counter
    // =========================================================================
    
    reg [$clog2(NUM_LAYERS)-1:0] layer_idx;
    
    // =========================================================================
    // Token embedding
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] current_embed;
    reg [$clog2(VOCAB_SIZE)-1:0] input_token_reg;
    
    // =========================================================================
    // Decoder block instance (shared across layers)
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] block_x_in;
    reg block_valid_in;
    wire block_ready_in;
    reg block_cache_clear;
    wire [DIM*ACT_WIDTH-1:0] block_y_out;
    wire block_valid_out;
    reg block_ready_out;
    
    // Weight selection based on current layer
    reg [DIM*ACT_WIDTH-1:0] cur_rms_attn_gamma, cur_rms_mlp_gamma;
    reg [DIM*DIM*2-1:0] cur_attn_w_q, cur_attn_w_o;
    reg [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] cur_attn_w_k, cur_attn_w_v;
    reg [DIM*MLP_DIM*2-1:0] cur_mlp_w_gate, cur_mlp_w_up;
    reg [MLP_DIM*DIM*2-1:0] cur_mlp_w_down;
    reg [HEAD_DIM*ACT_WIDTH-1:0] cur_rope_cos, cur_rope_sin;
    
    always @(*) begin
        cur_rms_attn_gamma = rms_attn_gamma[layer_idx];
        cur_rms_mlp_gamma = rms_mlp_gamma[layer_idx];
        cur_attn_w_q = attn_w_q[layer_idx];
        cur_attn_w_k = attn_w_k[layer_idx];
        cur_attn_w_v = attn_w_v[layer_idx];
        cur_attn_w_o = attn_w_o[layer_idx];
        cur_mlp_w_gate = mlp_w_gate[layer_idx];
        cur_mlp_w_up = mlp_w_up[layer_idx];
        cur_mlp_w_down = mlp_w_down[layer_idx];
        cur_rope_cos = rope_cos[current_position];
        cur_rope_sin = rope_sin[current_position];
    end
    
    llm_block #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MLP_DIM(MLP_DIM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .KV_HEADS(KV_HEADS),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) u_decoder_block (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(block_x_in),
        .position(current_position),
        .valid_in(block_valid_in),
        .ready_in(block_ready_in),
        .cache_clear(block_cache_clear),
        .rms_attn_gamma(cur_rms_attn_gamma),
        .rms_mlp_gamma(cur_rms_mlp_gamma),
        .attn_w_q(cur_attn_w_q),
        .attn_w_k(cur_attn_w_k),
        .attn_w_v(cur_attn_w_v),
        .attn_w_o(cur_attn_w_o),
        .rope_cos(cur_rope_cos),
        .rope_sin(cur_rope_sin),
        .mlp_w_gate(cur_mlp_w_gate),
        .mlp_w_up(cur_mlp_w_up),
        .mlp_w_down(cur_mlp_w_down),
        .y_out(block_y_out),
        .valid_out(block_valid_out),
        .ready_out(block_ready_out)
    );
    
    // =========================================================================
    // LM Head instance
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] head_x_in;
    reg head_valid_in;
    wire head_ready_in;
    wire [$clog2(VOCAB_SIZE)-1:0] head_token_out;
    wire signed [ACC_WIDTH-1:0] head_logit_out;
    wire head_valid_out;
    reg head_ready_out;
    
    llm_head #(
        .DIM(DIM),
        .VOCAB_SIZE(VOCAB_SIZE),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) u_lm_head (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(head_x_in),
        .valid_in(head_valid_in),
        .ready_in(head_ready_in),
        .rms_gamma(final_rms_gamma),
        .vocab_weights(lm_head_weights),
        .token_out(head_token_out),
        .logit_out(head_logit_out),
        .valid_out(head_valid_out),
        .ready_out(head_ready_out),
        .output_logits(1'b0)  // Argmax only
    );
    
    // =========================================================================
    // Ready signals
    // =========================================================================
    
    assign token_ready = (state == STATE_IDLE) && !is_vision_token;
    assign vision_ready = (state == STATE_IDLE) && is_vision_token;
    assign busy = (state != STATE_IDLE);
    
    // =========================================================================
    // Output
    // =========================================================================
    
    assign token_out = head_token_out;
    assign token_out_valid = (state == STATE_OUTPUT) && head_valid_out;
    
    // =========================================================================
    // Autoregressive control
    // =========================================================================
    
    reg generating;
    reg [$clog2(VOCAB_SIZE)-1:0] generated_token;
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer embed_i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            layer_idx <= 0;
            current_position <= 0;
            generating <= 1'b0;
            block_valid_in <= 1'b0;
            block_cache_clear <= 1'b0;
            block_ready_out <= 1'b0;
            head_valid_in <= 1'b0;
            head_ready_out <= 1'b0;
        end else begin
            // Default control signals
            block_valid_in <= 1'b0;
            block_cache_clear <= 1'b0;
            head_valid_in <= 1'b0;
            
            case (state)
                STATE_IDLE: begin
                    if (seq_start) begin
                        // Clear KV caches for new sequence
                        block_cache_clear <= 1'b1;
                        current_position <= 0;
                        generating <= 1'b0;
                    end else if (token_valid && !is_vision_token) begin
                        // Text token input
                        input_token_reg <= token_in;
                        state <= STATE_EMBED;
                    end else if (vision_valid && is_vision_token) begin
                        // Vision embedding input (already embedded)
                        current_embed <= vision_embed;
                        layer_idx <= 0;
                        state <= STATE_BLOCK;
                    end else if (generate) begin
                        generating <= 1'b1;
                    end
                end
                
                STATE_EMBED: begin
                    // Look up token embedding
                    for (embed_i = 0; embed_i < DIM; embed_i = embed_i + 1) begin
                        current_embed[embed_i*ACT_WIDTH +: ACT_WIDTH] <= 
                            token_embed_weights[(input_token_reg * DIM + embed_i) * ACT_WIDTH +: ACT_WIDTH];
                    end
                    
                    layer_idx <= 0;
                    state <= STATE_BLOCK;
                end
                
                STATE_BLOCK: begin
                    // Process through current decoder block
                    block_x_in <= current_embed;
                    block_valid_in <= 1'b1;
                    block_ready_out <= 1'b1;
                    
                    if (block_valid_out) begin
                        current_embed <= block_y_out;
                        state <= STATE_NEXT_BLOCK;
                    end
                end
                
                STATE_NEXT_BLOCK: begin
                    // Move to next layer or LM head
                    if (layer_idx >= NUM_LAYERS - 1) begin
                        state <= STATE_LM_HEAD;
                    end else begin
                        layer_idx <= layer_idx + 1;
                        state <= STATE_BLOCK;
                    end
                end
                
                STATE_LM_HEAD: begin
                    // Apply LM head for token prediction
                    head_x_in <= current_embed;
                    head_valid_in <= 1'b1;
                    head_ready_out <= 1'b1;
                    
                    if (head_valid_out) begin
                        generated_token <= head_token_out;
                        state <= STATE_OUTPUT;
                    end
                end
                
                STATE_OUTPUT: begin
                    // Output generated token
                    if (token_out_ready || !generating) begin
                        // Update position
                        current_position <= current_position + 1;
                        
                        if (generating) begin
                            // Continue autoregressive generation
                            // Feed generated token back as input
                            input_token_reg <= generated_token;
                            
                            // Check for EOS token (token ID 2 typically)
                            if (generated_token == 2) begin
                                generating <= 1'b0;
                                state <= STATE_IDLE;
                            end else begin
                                state <= STATE_EMBED;
                            end
                        end else begin
                            state <= STATE_IDLE;
                        end
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

module language_model_tb;
    // Small parameters for testing
    parameter DIM = 32;
    parameter NUM_LAYERS = 2;
    parameter NUM_HEADS = 2;
    parameter HEAD_DIM = 16;
    parameter MLP_DIM = 64;
    parameter VOCAB_SIZE = 64;
    parameter MAX_SEQ_LEN = 32;
    parameter KV_HEADS = 2;
    parameter ACT_WIDTH = 8;
    
    reg clk, rst_n;
    reg [$clog2(VOCAB_SIZE)-1:0] token_in;
    reg token_valid;
    wire token_ready;
    reg [DIM*ACT_WIDTH-1:0] vision_embed;
    reg vision_valid;
    wire vision_ready;
    reg is_vision_token;
    reg seq_start, generate;
    
    // Weight inputs
    reg [VOCAB_SIZE*DIM*ACT_WIDTH-1:0] token_embed_weights;
    reg [DIM*ACT_WIDTH-1:0] rms_attn_gamma [0:NUM_LAYERS-1];
    reg [DIM*ACT_WIDTH-1:0] rms_mlp_gamma [0:NUM_LAYERS-1];
    reg [DIM*DIM*2-1:0] attn_w_q [0:NUM_LAYERS-1];
    reg [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] attn_w_k [0:NUM_LAYERS-1];
    reg [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] attn_w_v [0:NUM_LAYERS-1];
    reg [DIM*DIM*2-1:0] attn_w_o [0:NUM_LAYERS-1];
    reg [DIM*MLP_DIM*2-1:0] mlp_w_gate [0:NUM_LAYERS-1];
    reg [DIM*MLP_DIM*2-1:0] mlp_w_up [0:NUM_LAYERS-1];
    reg [MLP_DIM*DIM*2-1:0] mlp_w_down [0:NUM_LAYERS-1];
    reg [HEAD_DIM*ACT_WIDTH-1:0] rope_cos [0:MAX_SEQ_LEN-1];
    reg [HEAD_DIM*ACT_WIDTH-1:0] rope_sin [0:MAX_SEQ_LEN-1];
    reg [DIM*ACT_WIDTH-1:0] final_rms_gamma;
    reg [VOCAB_SIZE*DIM*2-1:0] lm_head_weights;
    
    wire [$clog2(VOCAB_SIZE)-1:0] token_out;
    wire token_out_valid;
    reg token_out_ready;
    wire busy;
    wire [$clog2(MAX_SEQ_LEN)-1:0] current_position;
    
    language_model #(
        .DIM(DIM),
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MLP_DIM(MLP_DIM),
        .VOCAB_SIZE(VOCAB_SIZE),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .KV_HEADS(KV_HEADS),
        .ACT_WIDTH(ACT_WIDTH)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i, j;
    
    initial begin
        $display("Language Model Testbench");
        $display("========================");
        
        clk = 0;
        rst_n = 0;
        token_in = 0;
        token_valid = 0;
        vision_embed = 0;
        vision_valid = 0;
        is_vision_token = 0;
        seq_start = 0;
        generate = 0;
        token_out_ready = 1;
        
        // Initialize weights
        token_embed_weights = 0;
        for (i = 0; i < VOCAB_SIZE; i = i + 1) begin
            for (j = 0; j < DIM; j = j + 1) begin
                token_embed_weights[(i * DIM + j) * ACT_WIDTH +: ACT_WIDTH] = 8'd16;
            end
        end
        
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            rms_attn_gamma[i] = {DIM{8'd16}};
            rms_mlp_gamma[i] = {DIM{8'd16}};
            attn_w_q[i] = {(DIM*DIM){2'b01}};
            attn_w_k[i] = {(DIM*KV_HEADS*HEAD_DIM){2'b01}};
            attn_w_v[i] = {(DIM*KV_HEADS*HEAD_DIM){2'b01}};
            attn_w_o[i] = {(DIM*DIM){2'b01}};
            mlp_w_gate[i] = {(DIM*MLP_DIM){2'b01}};
            mlp_w_up[i] = {(DIM*MLP_DIM){2'b01}};
            mlp_w_down[i] = {(MLP_DIM*DIM){2'b01}};
        end
        
        for (i = 0; i < MAX_SEQ_LEN; i = i + 1) begin
            rope_cos[i] = {HEAD_DIM{8'd16}};
            rope_sin[i] = 0;
        end
        
        final_rms_gamma = {DIM{8'd16}};
        lm_head_weights = {(VOCAB_SIZE*DIM){2'b01}};
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Start new sequence
        seq_start = 1;
        @(posedge clk);
        seq_start = 0;
        repeat(2) @(posedge clk);
        
        // Send input token
        while (!token_ready) @(posedge clk);
        token_in = 5;
        token_valid = 1;
        @(posedge clk);
        token_valid = 0;
        
        // Wait for output
        repeat(10000000) begin
            @(posedge clk);
            if (token_out_valid) begin
                $display("Generated token: %0d at position %0d", token_out, current_position);
                break;
            end
        end
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #100000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
