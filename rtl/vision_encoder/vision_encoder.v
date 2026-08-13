// =============================================================================
// SiLens - Vision Encoder (SigLIP-B/16)
// =============================================================================
// Complete vision encoder implementing SigLIP-B/16 architecture.
//
// Architecture:
//   - Input: 384x384 RGB image
//   - Patch embedding: 24x24 grid of 16x16 patches -> 576 tokens x 768 dim
//   - 12 transformer blocks
//   - Final layer normalization
//   - Output: 576 tokens x 768 dimensions
//
// Parameters (SigLIP-B/16):
//   - 93M parameters
//   - Patch size: 16x16
//   - Hidden dim: 768
//   - Layers: 12
//   - Heads: 12
//   - MLP expansion: 4x (768 -> 3072 -> 768)
//
// License: Apache 2.0
// =============================================================================

module vision_encoder #(
    parameter IMG_SIZE    = 384,                    // Input image size
    parameter PATCH_SIZE  = 16,                     // Patch size
    parameter IN_CHANNELS = 3,                      // RGB channels
    parameter DIM         = 768,                    // Model dimension
    parameter NUM_LAYERS  = 12,                     // Number of transformer blocks
    parameter NUM_HEADS   = 12,                     // Attention heads per block
    parameter HEAD_DIM    = 64,                     // Dimension per head
    parameter MLP_DIM     = 3072,                   // MLP hidden dimension
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Image input interface (streaming pixels)
    input  wire [IN_CHANNELS*ACT_WIDTH-1:0] pixel_in,
    input  wire                         pixel_valid,
    output wire                         pixel_ready,
    
    // Frame control
    input  wire                         frame_start,
    
    // Weight interfaces (from weight ROM/hardwired)
    // Patch embedding weights
    input  wire [DIM*PATCH_PIXELS*2-1:0] patch_proj_weights,
    input  wire [NUM_PATCHES*DIM*ACT_WIDTH-1:0] pos_embed,
    
    // Transformer block weights (one set per layer - in practice, would be ROM)
    // For now, using single set and iterating
    input  wire [DIM*ACT_WIDTH-1:0]     ln1_gamma [0:NUM_LAYERS-1],
    input  wire [DIM*ACT_WIDTH-1:0]     ln1_beta  [0:NUM_LAYERS-1],
    input  wire [DIM*ACT_WIDTH-1:0]     ln2_gamma [0:NUM_LAYERS-1],
    input  wire [DIM*ACT_WIDTH-1:0]     ln2_beta  [0:NUM_LAYERS-1],
    input  wire [DIM*DIM*2-1:0]         attn_w_q  [0:NUM_LAYERS-1],
    input  wire [DIM*DIM*2-1:0]         attn_w_k  [0:NUM_LAYERS-1],
    input  wire [DIM*DIM*2-1:0]         attn_w_v  [0:NUM_LAYERS-1],
    input  wire [DIM*DIM*2-1:0]         attn_w_o  [0:NUM_LAYERS-1],
    input  wire [DIM*MLP_DIM*2-1:0]     mlp_w1    [0:NUM_LAYERS-1],
    input  wire [MLP_DIM*DIM*2-1:0]     mlp_w2    [0:NUM_LAYERS-1],
    input  wire [MLP_DIM*ACT_WIDTH-1:0] mlp_b1    [0:NUM_LAYERS-1],
    input  wire [DIM*ACT_WIDTH-1:0]     mlp_b2    [0:NUM_LAYERS-1],
    
    // Final layer norm weights
    input  wire [DIM*ACT_WIDTH-1:0]     final_ln_gamma,
    input  wire [DIM*ACT_WIDTH-1:0]     final_ln_beta,
    
    // Output interface (streaming tokens)
    output reg  [DIM*ACT_WIDTH-1:0]     token_out,
    output reg  [$clog2(NUM_PATCHES)-1:0] token_idx,
    output reg                          token_valid,
    input  wire                         token_ready,
    
    // Status
    output wire                         busy,
    output wire                         done
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    
    localparam GRID_SIZE = IMG_SIZE / PATCH_SIZE;   // 24
    localparam NUM_PATCHES = GRID_SIZE * GRID_SIZE; // 576
    localparam PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS; // 768
    
    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE       = 4'd0;
    localparam STATE_PATCH_EMB  = 4'd1;   // Patch embedding
    localparam STATE_BLOCK_LOAD = 4'd2;   // Load tokens to transformer block
    localparam STATE_BLOCK_PROC = 4'd3;   // Process transformer block
    localparam STATE_BLOCK_OUT  = 4'd4;   // Collect block output
    localparam STATE_FINAL_LN   = 4'd5;   // Final layer normalization
    localparam STATE_OUTPUT     = 4'd6;   // Output tokens
    localparam STATE_DONE       = 4'd7;
    
    reg [3:0] state;
    
    // =========================================================================
    // Layer counter
    // =========================================================================
    
    reg [$clog2(NUM_LAYERS)-1:0] layer_idx;
    
    // =========================================================================
    // Token buffer (shared between layers)
    // =========================================================================
    
    reg [ACT_WIDTH-1:0] token_buffer [0:NUM_PATCHES-1][0:DIM-1];
    reg [$clog2(NUM_PATCHES)-1:0] token_count;
    
    // =========================================================================
    // Patch Embedding Instance
    // =========================================================================
    
    wire [DIM*ACT_WIDTH-1:0] patch_token_out;
    wire [$clog2(NUM_PATCHES)-1:0] patch_token_idx;
    wire patch_token_valid;
    reg patch_token_ready;
    
    patch_embed #(
        .IMG_SIZE(IMG_SIZE),
        .PATCH_SIZE(PATCH_SIZE),
        .IN_CHANNELS(IN_CHANNELS),
        .EMBED_DIM(DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) u_patch_embed (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .pixel_ready(pixel_ready),
        .frame_start(frame_start && state == STATE_IDLE),
        .proj_weights(patch_proj_weights),
        .pos_embed(pos_embed),
        .token_out(patch_token_out),
        .token_idx(patch_token_idx),
        .token_valid(patch_token_valid),
        .token_ready(patch_token_ready)
    );
    
    // =========================================================================
    // Transformer Block Instance (shared, iterated NUM_LAYERS times)
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] block_x_in;
    reg [$clog2(NUM_PATCHES)-1:0] block_token_idx_in;
    reg block_token_valid_in;
    wire block_token_ready_in;
    reg block_seq_start;
    reg block_seq_done_in;
    
    wire [DIM*ACT_WIDTH-1:0] block_y_out;
    wire [$clog2(NUM_PATCHES)-1:0] block_token_idx_out;
    wire block_token_valid_out;
    reg block_token_ready_out;
    
    // Mux weights based on current layer
    reg [DIM*ACT_WIDTH-1:0] cur_ln1_gamma, cur_ln1_beta;
    reg [DIM*ACT_WIDTH-1:0] cur_ln2_gamma, cur_ln2_beta;
    reg [DIM*DIM*2-1:0] cur_attn_w_q, cur_attn_w_k, cur_attn_w_v, cur_attn_w_o;
    reg [DIM*MLP_DIM*2-1:0] cur_mlp_w1;
    reg [MLP_DIM*DIM*2-1:0] cur_mlp_w2;
    reg [MLP_DIM*ACT_WIDTH-1:0] cur_mlp_b1;
    reg [DIM*ACT_WIDTH-1:0] cur_mlp_b2;
    
    // Weight selection based on layer
    always @(*) begin
        cur_ln1_gamma = ln1_gamma[layer_idx];
        cur_ln1_beta  = ln1_beta[layer_idx];
        cur_ln2_gamma = ln2_gamma[layer_idx];
        cur_ln2_beta  = ln2_beta[layer_idx];
        cur_attn_w_q  = attn_w_q[layer_idx];
        cur_attn_w_k  = attn_w_k[layer_idx];
        cur_attn_w_v  = attn_w_v[layer_idx];
        cur_attn_w_o  = attn_w_o[layer_idx];
        cur_mlp_w1    = mlp_w1[layer_idx];
        cur_mlp_w2    = mlp_w2[layer_idx];
        cur_mlp_b1    = mlp_b1[layer_idx];
        cur_mlp_b2    = mlp_b2[layer_idx];
    end
    
    vit_block #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MLP_DIM(MLP_DIM),
        .SEQ_LEN(NUM_PATCHES),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) u_vit_block (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(block_x_in),
        .token_idx_in(block_token_idx_in),
        .token_valid_in(block_token_valid_in),
        .token_ready_in(block_token_ready_in),
        .seq_start(block_seq_start),
        .seq_done_in(block_seq_done_in),
        .ln1_gamma(cur_ln1_gamma),
        .ln1_beta(cur_ln1_beta),
        .ln2_gamma(cur_ln2_gamma),
        .ln2_beta(cur_ln2_beta),
        .attn_w_q(cur_attn_w_q),
        .attn_w_k(cur_attn_w_k),
        .attn_w_v(cur_attn_w_v),
        .attn_w_o(cur_attn_w_o),
        .mlp_w1(cur_mlp_w1),
        .mlp_w2(cur_mlp_w2),
        .mlp_b1(cur_mlp_b1),
        .mlp_b2(cur_mlp_b2),
        .y_out(block_y_out),
        .token_idx_out(block_token_idx_out),
        .token_valid_out(block_token_valid_out),
        .token_ready_out(block_token_ready_out)
    );
    
    // =========================================================================
    // Final Layer Norm Instance
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] final_ln_input;
    reg final_ln_valid_in;
    wire final_ln_ready_in;
    wire [DIM*ACT_WIDTH-1:0] final_ln_output;
    wire final_ln_valid_out;
    reg final_ln_ready_out;
    
    layer_norm #(
        .DIM(DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) u_final_ln (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(final_ln_input),
        .valid_in(final_ln_valid_in),
        .ready_in(final_ln_ready_in),
        .gamma(final_ln_gamma),
        .beta(final_ln_beta),
        .y_out(final_ln_output),
        .valid_out(final_ln_valid_out),
        .ready_out(final_ln_ready_out)
    );
    
    // =========================================================================
    // Processing indices
    // =========================================================================
    
    reg [$clog2(NUM_PATCHES)-1:0] load_idx;
    reg [$clog2(NUM_PATCHES)-1:0] out_idx;
    reg [$clog2(NUM_PATCHES)-1:0] recv_count;
    
    // =========================================================================
    // Status outputs
    // =========================================================================
    
    assign busy = (state != STATE_IDLE);
    assign done = (state == STATE_DONE);
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer buf_d, buf_t;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            layer_idx <= 0;
            token_count <= 0;
            load_idx <= 0;
            out_idx <= 0;
            recv_count <= 0;
            token_valid <= 1'b0;
            token_idx <= 0;
            
            patch_token_ready <= 1'b0;
            block_token_valid_in <= 1'b0;
            block_seq_start <= 1'b0;
            block_seq_done_in <= 1'b0;
            block_token_ready_out <= 1'b0;
            final_ln_valid_in <= 1'b0;
            final_ln_ready_out <= 1'b0;
        end else begin
            // Default control signals
            block_seq_start <= 1'b0;
            block_seq_done_in <= 1'b0;
            block_token_valid_in <= 1'b0;
            final_ln_valid_in <= 1'b0;
            
            case (state)
                STATE_IDLE: begin
                    token_valid <= 1'b0;
                    if (frame_start) begin
                        state <= STATE_PATCH_EMB;
                        token_count <= 0;
                        layer_idx <= 0;
                        patch_token_ready <= 1'b1;
                    end
                end
                
                STATE_PATCH_EMB: begin
                    // Collect embedded tokens from patch embedding
                    patch_token_ready <= 1'b1;
                    
                    if (patch_token_valid) begin
                        // Store token in buffer
                        for (buf_d = 0; buf_d < DIM; buf_d = buf_d + 1) begin
                            token_buffer[patch_token_idx][buf_d] <= 
                                patch_token_out[buf_d*ACT_WIDTH +: ACT_WIDTH];
                        end
                        token_count <= token_count + 1;
                    end
                    
                    // Check if all patches embedded
                    if (token_count >= NUM_PATCHES - 1 && patch_token_valid) begin
                        state <= STATE_BLOCK_LOAD;
                        load_idx <= 0;
                        layer_idx <= 0;
                        patch_token_ready <= 1'b0;
                    end
                end
                
                STATE_BLOCK_LOAD: begin
                    // Start feeding tokens to transformer block
                    block_seq_start <= (load_idx == 0);
                    block_token_valid_in <= 1'b1;
                    block_token_ready_out <= 1'b1;
                    recv_count <= 0;
                    
                    // Pack current token for block input
                    for (buf_d = 0; buf_d < DIM; buf_d = buf_d + 1) begin
                        block_x_in[buf_d*ACT_WIDTH +: ACT_WIDTH] <= 
                            token_buffer[load_idx][buf_d];
                    end
                    block_token_idx_in <= load_idx;
                    
                    if (block_token_ready_in) begin
                        if (load_idx >= token_count - 1) begin
                            block_seq_done_in <= 1'b1;
                            state <= STATE_BLOCK_PROC;
                        end else begin
                            load_idx <= load_idx + 1;
                        end
                    end
                end
                
                STATE_BLOCK_PROC: begin
                    // Wait for block processing and collect outputs
                    block_token_ready_out <= 1'b1;
                    
                    if (block_token_valid_out) begin
                        // Store output back to buffer
                        for (buf_d = 0; buf_d < DIM; buf_d = buf_d + 1) begin
                            token_buffer[block_token_idx_out][buf_d] <= 
                                block_y_out[buf_d*ACT_WIDTH +: ACT_WIDTH];
                        end
                        recv_count <= recv_count + 1;
                        
                        // Check if all tokens received
                        if (recv_count >= token_count - 1) begin
                            // Check if more layers to process
                            if (layer_idx >= NUM_LAYERS - 1) begin
                                state <= STATE_FINAL_LN;
                                out_idx <= 0;
                            end else begin
                                layer_idx <= layer_idx + 1;
                                load_idx <= 0;
                                state <= STATE_BLOCK_LOAD;
                            end
                        end
                    end
                end
                
                STATE_FINAL_LN: begin
                    // Apply final layer norm to each token
                    final_ln_ready_out <= 1'b1;
                    
                    // Pack current token
                    for (buf_d = 0; buf_d < DIM; buf_d = buf_d + 1) begin
                        final_ln_input[buf_d*ACT_WIDTH +: ACT_WIDTH] <= 
                            token_buffer[out_idx][buf_d];
                    end
                    final_ln_valid_in <= 1'b1;
                    
                    if (final_ln_valid_out) begin
                        // Store normalized token
                        for (buf_d = 0; buf_d < DIM; buf_d = buf_d + 1) begin
                            token_buffer[out_idx][buf_d] <= 
                                final_ln_output[buf_d*ACT_WIDTH +: ACT_WIDTH];
                        end
                        
                        if (out_idx >= token_count - 1) begin
                            state <= STATE_OUTPUT;
                            out_idx <= 0;
                        end else begin
                            out_idx <= out_idx + 1;
                        end
                    end
                end
                
                STATE_OUTPUT: begin
                    // Output tokens one at a time
                    token_valid <= 1'b1;
                    token_idx <= out_idx;
                    
                    // Pack output token
                    for (buf_d = 0; buf_d < DIM; buf_d = buf_d + 1) begin
                        token_out[buf_d*ACT_WIDTH +: ACT_WIDTH] <= 
                            token_buffer[out_idx][buf_d];
                    end
                    
                    if (token_ready) begin
                        if (out_idx >= token_count - 1) begin
                            token_valid <= 1'b0;
                            state <= STATE_DONE;
                        end else begin
                            out_idx <= out_idx + 1;
                        end
                    end
                end
                
                STATE_DONE: begin
                    token_valid <= 1'b0;
                    // Stay here until next frame_start (handled by STATE_IDLE check)
                    state <= STATE_IDLE;
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

module vision_encoder_tb;
    // Use smaller parameters for testing
    parameter IMG_SIZE = 32;
    parameter PATCH_SIZE = 8;
    parameter IN_CHANNELS = 3;
    parameter DIM = 32;
    parameter NUM_LAYERS = 2;
    parameter NUM_HEADS = 2;
    parameter HEAD_DIM = 16;
    parameter MLP_DIM = 64;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    
    localparam GRID_SIZE = IMG_SIZE / PATCH_SIZE;
    localparam NUM_PATCHES = GRID_SIZE * GRID_SIZE;
    localparam PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS;
    
    reg clk, rst_n;
    reg [IN_CHANNELS*ACT_WIDTH-1:0] pixel_in;
    reg pixel_valid;
    wire pixel_ready;
    reg frame_start;
    
    // Weight inputs (simplified for test)
    reg [DIM*PATCH_PIXELS*2-1:0] patch_proj_weights;
    reg [NUM_PATCHES*DIM*ACT_WIDTH-1:0] pos_embed;
    reg [DIM*ACT_WIDTH-1:0] ln1_gamma [0:NUM_LAYERS-1];
    reg [DIM*ACT_WIDTH-1:0] ln1_beta [0:NUM_LAYERS-1];
    reg [DIM*ACT_WIDTH-1:0] ln2_gamma [0:NUM_LAYERS-1];
    reg [DIM*ACT_WIDTH-1:0] ln2_beta [0:NUM_LAYERS-1];
    reg [DIM*DIM*2-1:0] attn_w_q [0:NUM_LAYERS-1];
    reg [DIM*DIM*2-1:0] attn_w_k [0:NUM_LAYERS-1];
    reg [DIM*DIM*2-1:0] attn_w_v [0:NUM_LAYERS-1];
    reg [DIM*DIM*2-1:0] attn_w_o [0:NUM_LAYERS-1];
    reg [DIM*MLP_DIM*2-1:0] mlp_w1 [0:NUM_LAYERS-1];
    reg [MLP_DIM*DIM*2-1:0] mlp_w2 [0:NUM_LAYERS-1];
    reg [MLP_DIM*ACT_WIDTH-1:0] mlp_b1 [0:NUM_LAYERS-1];
    reg [DIM*ACT_WIDTH-1:0] mlp_b2 [0:NUM_LAYERS-1];
    reg [DIM*ACT_WIDTH-1:0] final_ln_gamma;
    reg [DIM*ACT_WIDTH-1:0] final_ln_beta;
    
    wire [DIM*ACT_WIDTH-1:0] token_out;
    wire [$clog2(NUM_PATCHES)-1:0] token_idx;
    wire token_valid;
    reg token_ready;
    wire busy, done;
    
    vision_encoder #(
        .IMG_SIZE(IMG_SIZE),
        .PATCH_SIZE(PATCH_SIZE),
        .IN_CHANNELS(IN_CHANNELS),
        .DIM(DIM),
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MLP_DIM(MLP_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i, j, tokens_received;
    
    initial begin
        $display("Vision Encoder Testbench");
        $display("========================");
        $display("Image: %0dx%0d, Patches: %0d, DIM: %0d, Layers: %0d",
                 IMG_SIZE, IMG_SIZE, NUM_PATCHES, DIM, NUM_LAYERS);
        
        clk = 0;
        rst_n = 0;
        pixel_in = 0;
        pixel_valid = 0;
        frame_start = 0;
        token_ready = 1;
        
        // Initialize weights
        patch_proj_weights = {(DIM*PATCH_PIXELS){2'b01}};
        pos_embed = 0;
        final_ln_gamma = {DIM{8'd16}};
        final_ln_beta = 0;
        
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            ln1_gamma[i] = {DIM{8'd16}};
            ln1_beta[i] = 0;
            ln2_gamma[i] = {DIM{8'd16}};
            ln2_beta[i] = 0;
            attn_w_q[i] = {(DIM*DIM){2'b01}};
            attn_w_k[i] = {(DIM*DIM){2'b01}};
            attn_w_v[i] = {(DIM*DIM){2'b01}};
            attn_w_o[i] = {(DIM*DIM){2'b01}};
            mlp_w1[i] = {(DIM*MLP_DIM){2'b01}};
            mlp_w2[i] = {(MLP_DIM*DIM){2'b01}};
            mlp_b1[i] = 0;
            mlp_b2[i] = 0;
        end
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Start frame
        frame_start = 1;
        @(posedge clk);
        frame_start = 0;
        
        // Send image pixels
        tokens_received = 0;
        
        for (i = 0; i < IMG_SIZE * IMG_SIZE; i = i + 1) begin
            while (!pixel_ready) @(posedge clk);
            @(posedge clk);
            pixel_in = {8'd100, 8'd150, 8'd200};
            pixel_valid = 1;
            
            if (token_valid) begin
                tokens_received = tokens_received + 1;
                $display("Token %0d received during pixel load", token_idx);
            end
        end
        
        @(posedge clk);
        pixel_valid = 0;
        
        // Wait for all output tokens
        repeat(1000000) begin
            @(posedge clk);
            if (token_valid) begin
                tokens_received = tokens_received + 1;
                $display("Output token %0d", token_idx);
            end
            if (done) begin
                $display("Encoding complete!");
                break;
            end
        end
        
        $display("Received %0d tokens (expected %0d)", tokens_received, NUM_PATCHES);
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #50000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
