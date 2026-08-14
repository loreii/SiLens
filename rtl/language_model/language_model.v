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
// Note: Weights are passed as flattened packed arrays for Icarus Verilog
//       compatibility. Use helper functions to index individual layers.
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
    input  wire                         gen_start,          // Start autoregressive generation
    
    // Token embedding weights (ROM) - simplified for compilation
    // Full implementation would use external memory
    input  wire [DIM*ACT_WIDTH-1:0]     token_embed_sample, // Sample embedding for one token
    
    // Block weights - simplified: single layer weights, instantiate with layer select
    input  wire [DIM*ACT_WIDTH-1:0]     rms_attn_gamma,
    input  wire [DIM*ACT_WIDTH-1:0]     rms_mlp_gamma,
    input  wire [DIM*DIM*2-1:0]         attn_w_q,
    input  wire [DIM*DIM*2-1:0]         attn_w_k,
    input  wire [DIM*DIM*2-1:0]         attn_w_v,
    input  wire [DIM*DIM*2-1:0]         attn_w_o,
    input  wire [DIM*MLP_DIM*2-1:0]     mlp_w_gate,
    input  wire [DIM*MLP_DIM*2-1:0]     mlp_w_up,
    input  wire [MLP_DIM*DIM*2-1:0]     mlp_w_down,
    
    // RoPE frequencies (for current position)
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] rope_cos,
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] rope_sin,
    
    // LM head weights
    input  wire [DIM*ACT_WIDTH-1:0]     final_rms_gamma,
    input  wire [DIM*2-1:0]             lm_head_sample,    // Sample LM head weights
    
    // Layer select (for external weight ROM indexing)
    output wire [$clog2(NUM_LAYERS)-1:0] layer_select,
    output wire [$clog2(MAX_SEQ_LEN)-1:0] position_select,
    
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
    
    assign layer_select = layer_idx;
    assign position_select = current_position;
    
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
        .rms_attn_gamma(rms_attn_gamma),
        .rms_mlp_gamma(rms_mlp_gamma),
        .attn_w_q(attn_w_q),
        .attn_w_k(attn_w_k),
        .attn_w_v(attn_w_v),
        .attn_w_o(attn_w_o),
        .rope_cos(rope_cos),
        .rope_sin(rope_sin),
        .mlp_w_gate(mlp_w_gate),
        .mlp_w_up(mlp_w_up),
        .mlp_w_down(mlp_w_down),
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
        .vocab_weight_sample(lm_head_sample),
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
            current_embed <= 0;
            input_token_reg <= 0;
            generated_token <= 0;
            block_x_in <= 0;
            head_x_in <= 0;
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
                    end else if (gen_start) begin
                        generating <= 1'b1;
                    end
                end
                
                STATE_EMBED: begin
                    // Use sample embedding (in full impl, would index into token_embed_weights)
                    current_embed <= token_embed_sample;
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
    reg seq_start, gen_start_sig;
    
    // Weight inputs (simplified)
    reg [DIM*ACT_WIDTH-1:0] token_embed_sample;
    reg [DIM*ACT_WIDTH-1:0] rms_attn_gamma;
    reg [DIM*ACT_WIDTH-1:0] rms_mlp_gamma;
    reg [DIM*DIM*2-1:0] attn_w_q;
    reg [DIM*DIM*2-1:0] attn_w_k;
    reg [DIM*DIM*2-1:0] attn_w_v;
    reg [DIM*DIM*2-1:0] attn_w_o;
    reg [DIM*MLP_DIM*2-1:0] mlp_w_gate;
    reg [DIM*MLP_DIM*2-1:0] mlp_w_up;
    reg [MLP_DIM*DIM*2-1:0] mlp_w_down;
    reg [HEAD_DIM*ACT_WIDTH-1:0] rope_cos;
    reg [HEAD_DIM*ACT_WIDTH-1:0] rope_sin;
    reg [DIM*ACT_WIDTH-1:0] final_rms_gamma;
    reg [DIM*2-1:0] lm_head_sample;
    
    wire [$clog2(NUM_LAYERS)-1:0] layer_select;
    wire [$clog2(MAX_SEQ_LEN)-1:0] position_select;
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
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .token_in(token_in),
        .token_valid(token_valid),
        .token_ready(token_ready),
        .vision_embed(vision_embed),
        .vision_valid(vision_valid),
        .vision_ready(vision_ready),
        .is_vision_token(is_vision_token),
        .seq_start(seq_start),
        .gen_start(gen_start_sig),
        .token_embed_sample(token_embed_sample),
        .rms_attn_gamma(rms_attn_gamma),
        .rms_mlp_gamma(rms_mlp_gamma),
        .attn_w_q(attn_w_q),
        .attn_w_k(attn_w_k),
        .attn_w_v(attn_w_v),
        .attn_w_o(attn_w_o),
        .mlp_w_gate(mlp_w_gate),
        .mlp_w_up(mlp_w_up),
        .mlp_w_down(mlp_w_down),
        .rope_cos(rope_cos),
        .rope_sin(rope_sin),
        .final_rms_gamma(final_rms_gamma),
        .lm_head_sample(lm_head_sample),
        .layer_select(layer_select),
        .position_select(position_select),
        .token_out(token_out),
        .token_out_valid(token_out_valid),
        .token_out_ready(token_out_ready),
        .busy(busy),
        .current_position(current_position)
    );
    
    always #5 clk = ~clk;
    
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
        gen_start_sig = 0;
        token_out_ready = 1;
        
        // Initialize weights with simple patterns
        token_embed_sample = {DIM{8'd16}};
        rms_attn_gamma = {DIM{8'd16}};
        rms_mlp_gamma = {DIM{8'd16}};
        attn_w_q = {(DIM*DIM){2'b01}};
        attn_w_k = {(DIM*DIM){2'b01}};
        attn_w_v = {(DIM*DIM){2'b01}};
        attn_w_o = {(DIM*DIM){2'b01}};
        mlp_w_gate = {(DIM*MLP_DIM){2'b01}};
        mlp_w_up = {(DIM*MLP_DIM){2'b01}};
        mlp_w_down = {(MLP_DIM*DIM){2'b01}};
        rope_cos = {HEAD_DIM{8'd16}};
        rope_sin = 0;
        final_rms_gamma = {DIM{8'd16}};
        lm_head_sample = {DIM{2'b01}};
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Start new sequence
        seq_start = 1;
        @(posedge clk);
        seq_start = 0;
        repeat(2) @(posedge clk);
        
        // Send input token
        @(posedge clk);
        while (!token_ready) @(posedge clk);
        token_in = 5;
        token_valid = 1;
        @(posedge clk);
        token_valid = 0;
        
        // Wait for output (with timeout)
        repeat(10000) begin
            @(posedge clk);
            if (token_out_valid) begin
                $display("Generated token: %0d at position %0d", token_out, current_position);
                break;
            end
        end
        
        $display("Testbench complete");
        $finish;
    end
    
    // Timeout
    initial begin
        #1000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
