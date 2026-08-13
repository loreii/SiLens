// =============================================================================
// SiLens - Vision Transformer Block
// =============================================================================
// Implements a complete transformer block for Vision Transformer.
//
// Architecture:
//   1. Layer Norm -> Attention -> Residual Add
//   2. Layer Norm -> MLP -> Residual Add
//
// Block formula (pre-norm style like ViT):
//   x' = x + Attention(LayerNorm(x))
//   y  = x' + MLP(LayerNorm(x'))
//
// Parameters for SigLIP-B/16:
//   - Dimension: 768
//   - Heads: 12
//   - MLP expansion: 4x (768 -> 3072 -> 768)
//
// License: Apache 2.0
// =============================================================================

module vit_block #(
    parameter DIM         = 768,                    // Model dimension
    parameter NUM_HEADS   = 12,                     // Number of attention heads
    parameter HEAD_DIM    = 64,                     // Dimension per head
    parameter MLP_DIM     = 3072,                   // MLP hidden dimension
    parameter SEQ_LEN     = 576,                    // Sequence length
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface (streaming tokens)
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,               // Input token
    input  wire [$clog2(SEQ_LEN)-1:0]   token_idx_in,       // Token position
    input  wire                         token_valid_in,
    output wire                         token_ready_in,
    
    // Sequence control
    input  wire                         seq_start,          // Start of sequence
    input  wire                         seq_done_in,        // All input tokens received
    
    // Layer norm parameters
    input  wire [DIM*ACT_WIDTH-1:0]     ln1_gamma,          // Pre-attention layer norm
    input  wire [DIM*ACT_WIDTH-1:0]     ln1_beta,
    input  wire [DIM*ACT_WIDTH-1:0]     ln2_gamma,          // Pre-MLP layer norm
    input  wire [DIM*ACT_WIDTH-1:0]     ln2_beta,
    
    // Attention weights (Q, K, V, O projections)
    input  wire [DIM*DIM*2-1:0]         attn_w_q,
    input  wire [DIM*DIM*2-1:0]         attn_w_k,
    input  wire [DIM*DIM*2-1:0]         attn_w_v,
    input  wire [DIM*DIM*2-1:0]         attn_w_o,
    
    // MLP weights
    input  wire [DIM*MLP_DIM*2-1:0]     mlp_w1,
    input  wire [MLP_DIM*DIM*2-1:0]     mlp_w2,
    input  wire [MLP_DIM*ACT_WIDTH-1:0] mlp_b1,
    input  wire [DIM*ACT_WIDTH-1:0]     mlp_b2,
    
    // Output interface
    output wire [DIM*ACT_WIDTH-1:0]     y_out,              // Output token
    output wire [$clog2(SEQ_LEN)-1:0]   token_idx_out,
    output wire                         token_valid_out,
    input  wire                         token_ready_out
);

    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE       = 4'd0;
    localparam STATE_LN1_LOAD   = 4'd1;   // Load tokens for LN1
    localparam STATE_LN1        = 4'd2;   // Layer norm before attention
    localparam STATE_ATTN       = 4'd3;   // Self-attention
    localparam STATE_RESIDUAL1  = 4'd4;   // First residual add
    localparam STATE_LN2        = 4'd5;   // Layer norm before MLP
    localparam STATE_MLP        = 4'd6;   // MLP feedforward
    localparam STATE_RESIDUAL2  = 4'd7;   // Second residual add
    localparam STATE_OUTPUT     = 4'd8;   // Output tokens
    
    reg [3:0] state;
    
    // =========================================================================
    // Token buffers
    // =========================================================================
    
    // Input token storage (for residual connections)
    reg [ACT_WIDTH-1:0] input_buffer [0:SEQ_LEN-1][0:DIM-1];
    reg [$clog2(SEQ_LEN)-1:0] load_count;
    reg all_tokens_loaded;
    
    // Intermediate buffers
    reg [ACT_WIDTH-1:0] ln1_buffer [0:SEQ_LEN-1][0:DIM-1];    // After LN1
    reg [ACT_WIDTH-1:0] attn_buffer [0:SEQ_LEN-1][0:DIM-1];   // After attention
    reg [ACT_WIDTH-1:0] res1_buffer [0:SEQ_LEN-1][0:DIM-1];   // After residual 1
    reg [ACT_WIDTH-1:0] ln2_buffer [0:SEQ_LEN-1][0:DIM-1];    // After LN2
    reg [ACT_WIDTH-1:0] mlp_buffer [0:SEQ_LEN-1][0:DIM-1];    // After MLP
    reg [ACT_WIDTH-1:0] output_buffer [0:SEQ_LEN-1][0:DIM-1]; // Final output
    
    // Processing indices
    reg [$clog2(SEQ_LEN)-1:0] proc_idx;
    reg [$clog2(SEQ_LEN)-1:0] out_idx;
    
    // =========================================================================
    // Layer Normalization Instance
    // =========================================================================
    
    // LN input/output signals
    reg [DIM*ACT_WIDTH-1:0] ln_input;
    wire [DIM*ACT_WIDTH-1:0] ln_output;
    reg ln_valid_in;
    wire ln_ready_in;
    wire ln_valid_out;
    reg ln_ready_out;
    reg [DIM*ACT_WIDTH-1:0] ln_gamma_sel;
    reg [DIM*ACT_WIDTH-1:0] ln_beta_sel;
    
    layer_norm #(
        .DIM(DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) u_layer_norm (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(ln_input),
        .valid_in(ln_valid_in),
        .ready_in(ln_ready_in),
        .gamma(ln_gamma_sel),
        .beta(ln_beta_sel),
        .y_out(ln_output),
        .valid_out(ln_valid_out),
        .ready_out(ln_ready_out)
    );
    
    // =========================================================================
    // Attention Instance
    // =========================================================================
    
    wire [DIM*ACT_WIDTH-1:0] attn_x_in;
    wire [$clog2(SEQ_LEN)-1:0] attn_token_idx;
    reg attn_token_valid;
    wire attn_token_ready;
    reg attn_seq_start;
    reg attn_seq_done;
    wire [DIM*ACT_WIDTH-1:0] attn_y_out;
    wire [$clog2(SEQ_LEN)-1:0] attn_out_idx;
    wire attn_out_valid;
    reg attn_out_ready;
    
    vit_attention #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .SEQ_LEN(SEQ_LEN),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) u_attention (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(attn_x_in),
        .token_idx(attn_token_idx),
        .token_valid(attn_token_valid),
        .token_ready(attn_token_ready),
        .seq_start(attn_seq_start),
        .seq_done(attn_seq_done),
        .w_q(attn_w_q),
        .w_k(attn_w_k),
        .w_v(attn_w_v),
        .w_o(attn_w_o),
        .y_out(attn_y_out),
        .out_token_idx(attn_out_idx),
        .out_valid(attn_out_valid),
        .out_ready(attn_out_ready)
    );
    
    // =========================================================================
    // MLP Instance
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] mlp_x_in;
    reg mlp_valid_in;
    wire mlp_ready_in;
    wire [DIM*ACT_WIDTH-1:0] mlp_y_out;
    wire mlp_valid_out;
    reg mlp_ready_out;
    
    vit_mlp #(
        .DIM(DIM),
        .HIDDEN_DIM(MLP_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) u_mlp (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(mlp_x_in),
        .valid_in(mlp_valid_in),
        .ready_in(mlp_ready_in),
        .w1(mlp_w1),
        .w2(mlp_w2),
        .b1(mlp_b1),
        .b2(mlp_b2),
        .y_out(mlp_y_out),
        .valid_out(mlp_valid_out),
        .ready_out(mlp_ready_out)
    );
    
    // =========================================================================
    // Ready signal for input
    // =========================================================================
    
    assign token_ready_in = (state == STATE_LN1_LOAD);
    
    // =========================================================================
    // Input loading
    // =========================================================================
    
    integer load_d;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            load_count <= 0;
            all_tokens_loaded <= 1'b0;
        end else if (state == STATE_IDLE && seq_start) begin
            load_count <= 0;
            all_tokens_loaded <= 1'b0;
        end else if (state == STATE_LN1_LOAD && token_valid_in && token_ready_in) begin
            // Store input token
            for (load_d = 0; load_d < DIM; load_d = load_d + 1) begin
                input_buffer[token_idx_in][load_d] <= x_in[load_d*ACT_WIDTH +: ACT_WIDTH];
            end
            load_count <= load_count + 1;
        end else if (state == STATE_LN1_LOAD && seq_done_in) begin
            all_tokens_loaded <= 1'b1;
        end
    end
    
    // =========================================================================
    // Pack buffer to vector helpers
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] packed_input;
    reg [DIM*ACT_WIDTH-1:0] packed_ln1;
    reg [DIM*ACT_WIDTH-1:0] packed_res1;
    reg [DIM*ACT_WIDTH-1:0] packed_ln2;
    reg [DIM*ACT_WIDTH-1:0] packed_output;
    
    integer pack_i;
    
    // Pack input buffer for current token
    always @(*) begin
        for (pack_i = 0; pack_i < DIM; pack_i = pack_i + 1) begin
            packed_input[pack_i*ACT_WIDTH +: ACT_WIDTH] = input_buffer[proc_idx][pack_i];
            packed_ln1[pack_i*ACT_WIDTH +: ACT_WIDTH] = ln1_buffer[proc_idx][pack_i];
            packed_res1[pack_i*ACT_WIDTH +: ACT_WIDTH] = res1_buffer[proc_idx][pack_i];
            packed_ln2[pack_i*ACT_WIDTH +: ACT_WIDTH] = ln2_buffer[proc_idx][pack_i];
            packed_output[pack_i*ACT_WIDTH +: ACT_WIDTH] = output_buffer[out_idx][pack_i];
        end
    end
    
    // Assign to attention input
    assign attn_x_in = packed_ln1;
    assign attn_token_idx = proc_idx;
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer res_d;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            proc_idx <= 0;
            out_idx <= 0;
            ln_valid_in <= 1'b0;
            ln_ready_out <= 1'b0;
            attn_token_valid <= 1'b0;
            attn_seq_start <= 1'b0;
            attn_seq_done <= 1'b0;
            attn_out_ready <= 1'b0;
            mlp_valid_in <= 1'b0;
            mlp_ready_out <= 1'b0;
        end else begin
            // Default control signals
            ln_valid_in <= 1'b0;
            attn_seq_start <= 1'b0;
            attn_seq_done <= 1'b0;
            attn_token_valid <= 1'b0;
            mlp_valid_in <= 1'b0;
            
            case (state)
                STATE_IDLE: begin
                    if (seq_start) begin
                        state <= STATE_LN1_LOAD;
                        proc_idx <= 0;
                    end
                end
                
                STATE_LN1_LOAD: begin
                    // Wait for all tokens to be loaded
                    if (all_tokens_loaded) begin
                        state <= STATE_LN1;
                        proc_idx <= 0;
                        ln_gamma_sel <= ln1_gamma;
                        ln_beta_sel <= ln1_beta;
                    end
                end
                
                STATE_LN1: begin
                    // Apply layer norm to each token
                    ln_input <= packed_input;
                    ln_valid_in <= 1'b1;
                    ln_ready_out <= 1'b1;
                    
                    if (ln_valid_out) begin
                        // Store normalized output
                        for (res_d = 0; res_d < DIM; res_d = res_d + 1) begin
                            ln1_buffer[proc_idx][res_d] <= ln_output[res_d*ACT_WIDTH +: ACT_WIDTH];
                        end
                        
                        if (proc_idx >= load_count - 1) begin
                            state <= STATE_ATTN;
                            proc_idx <= 0;
                            attn_seq_start <= 1'b1;
                        end else begin
                            proc_idx <= proc_idx + 1;
                        end
                    end
                end
                
                STATE_ATTN: begin
                    // Feed tokens to attention
                    attn_token_valid <= 1'b1;
                    attn_out_ready <= 1'b1;
                    
                    if (attn_token_ready && proc_idx < load_count) begin
                        if (proc_idx >= load_count - 1) begin
                            attn_seq_done <= 1'b1;
                        end else begin
                            proc_idx <= proc_idx + 1;
                        end
                    end
                    
                    // Collect attention outputs
                    if (attn_out_valid) begin
                        for (res_d = 0; res_d < DIM; res_d = res_d + 1) begin
                            attn_buffer[attn_out_idx][res_d] <= attn_y_out[res_d*ACT_WIDTH +: ACT_WIDTH];
                        end
                    end
                    
                    // Check if all outputs received
                    if (attn_out_valid && attn_out_idx == load_count - 1) begin
                        state <= STATE_RESIDUAL1;
                        proc_idx <= 0;
                    end
                end
                
                STATE_RESIDUAL1: begin
                    // Add residual connection: x' = x + attn(ln(x))
                    for (res_d = 0; res_d < DIM; res_d = res_d + 1) begin
                        res1_buffer[proc_idx][res_d] <= saturate_add(
                            input_buffer[proc_idx][res_d],
                            attn_buffer[proc_idx][res_d]
                        );
                    end
                    
                    if (proc_idx >= load_count - 1) begin
                        state <= STATE_LN2;
                        proc_idx <= 0;
                        ln_gamma_sel <= ln2_gamma;
                        ln_beta_sel <= ln2_beta;
                    end else begin
                        proc_idx <= proc_idx + 1;
                    end
                end
                
                STATE_LN2: begin
                    // Apply layer norm before MLP
                    ln_input <= packed_res1;
                    ln_valid_in <= 1'b1;
                    ln_ready_out <= 1'b1;
                    
                    if (ln_valid_out) begin
                        for (res_d = 0; res_d < DIM; res_d = res_d + 1) begin
                            ln2_buffer[proc_idx][res_d] <= ln_output[res_d*ACT_WIDTH +: ACT_WIDTH];
                        end
                        
                        if (proc_idx >= load_count - 1) begin
                            state <= STATE_MLP;
                            proc_idx <= 0;
                        end else begin
                            proc_idx <= proc_idx + 1;
                        end
                    end
                end
                
                STATE_MLP: begin
                    // Apply MLP to each token
                    mlp_x_in <= packed_ln2;
                    mlp_valid_in <= 1'b1;
                    mlp_ready_out <= 1'b1;
                    
                    if (mlp_valid_out) begin
                        for (res_d = 0; res_d < DIM; res_d = res_d + 1) begin
                            mlp_buffer[proc_idx][res_d] <= mlp_y_out[res_d*ACT_WIDTH +: ACT_WIDTH];
                        end
                        
                        if (proc_idx >= load_count - 1) begin
                            state <= STATE_RESIDUAL2;
                            proc_idx <= 0;
                        end else begin
                            proc_idx <= proc_idx + 1;
                        end
                    end
                end
                
                STATE_RESIDUAL2: begin
                    // Add residual connection: y = x' + mlp(ln(x'))
                    for (res_d = 0; res_d < DIM; res_d = res_d + 1) begin
                        output_buffer[proc_idx][res_d] <= saturate_add(
                            res1_buffer[proc_idx][res_d],
                            mlp_buffer[proc_idx][res_d]
                        );
                    end
                    
                    if (proc_idx >= load_count - 1) begin
                        state <= STATE_OUTPUT;
                        out_idx <= 0;
                    end else begin
                        proc_idx <= proc_idx + 1;
                    end
                end
                
                STATE_OUTPUT: begin
                    // Output tokens one at a time
                    if (token_ready_out) begin
                        if (out_idx >= load_count - 1) begin
                            state <= STATE_IDLE;
                        end else begin
                            out_idx <= out_idx + 1;
                        end
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Output assignment
    // =========================================================================
    
    assign y_out = packed_output;
    assign token_idx_out = out_idx;
    assign token_valid_out = (state == STATE_OUTPUT);
    
    // =========================================================================
    // Saturating addition function
    // =========================================================================
    
    function [ACT_WIDTH-1:0] saturate_add;
        input [ACT_WIDTH-1:0] a;
        input [ACT_WIDTH-1:0] b;
        reg signed [ACT_WIDTH:0] sum;
        begin
            sum = $signed({a[ACT_WIDTH-1], a}) + $signed({b[ACT_WIDTH-1], b});
            
            if (sum > $signed({{2{1'b0}}, {(ACT_WIDTH-1){1'b1}}}))
                saturate_add = {1'b0, {(ACT_WIDTH-1){1'b1}}};
            else if (sum < $signed({{2{1'b1}}, {(ACT_WIDTH-1){1'b0}}}))
                saturate_add = {1'b1, {(ACT_WIDTH-1){1'b0}}};
            else
                saturate_add = sum[ACT_WIDTH-1:0];
        end
    endfunction

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module vit_block_tb;
    parameter DIM = 32;
    parameter NUM_HEADS = 2;
    parameter HEAD_DIM = 16;
    parameter MLP_DIM = 64;
    parameter SEQ_LEN = 4;
    parameter ACT_WIDTH = 8;
    
    reg clk, rst_n;
    reg [DIM*ACT_WIDTH-1:0] x_in;
    reg [$clog2(SEQ_LEN)-1:0] token_idx_in;
    reg token_valid_in;
    wire token_ready_in;
    reg seq_start, seq_done_in;
    
    reg [DIM*ACT_WIDTH-1:0] ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
    reg [DIM*DIM*2-1:0] attn_w_q, attn_w_k, attn_w_v, attn_w_o;
    reg [DIM*MLP_DIM*2-1:0] mlp_w1;
    reg [MLP_DIM*DIM*2-1:0] mlp_w2;
    reg [MLP_DIM*ACT_WIDTH-1:0] mlp_b1;
    reg [DIM*ACT_WIDTH-1:0] mlp_b2;
    
    wire [DIM*ACT_WIDTH-1:0] y_out;
    wire [$clog2(SEQ_LEN)-1:0] token_idx_out;
    wire token_valid_out;
    reg token_ready_out;
    
    vit_block #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MLP_DIM(MLP_DIM),
        .SEQ_LEN(SEQ_LEN),
        .ACT_WIDTH(ACT_WIDTH)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i, j;
    
    initial begin
        $display("ViT Block Testbench");
        $display("===================");
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        token_idx_in = 0;
        token_valid_in = 0;
        seq_start = 0;
        seq_done_in = 0;
        token_ready_out = 1;
        
        // Initialize all weights to +1 for testing
        ln1_gamma = {DIM{8'd16}};  // 1.0 in Q4.4
        ln1_beta = 0;
        ln2_gamma = {DIM{8'd16}};
        ln2_beta = 0;
        attn_w_q = {(DIM*DIM){2'b01}};
        attn_w_k = {(DIM*DIM){2'b01}};
        attn_w_v = {(DIM*DIM){2'b01}};
        attn_w_o = {(DIM*DIM){2'b01}};
        mlp_w1 = {(DIM*MLP_DIM){2'b01}};
        mlp_w2 = {(MLP_DIM*DIM){2'b01}};
        mlp_b1 = 0;
        mlp_b2 = 0;
        
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
            while (!token_ready_in) @(posedge clk);
            
            for (j = 0; j < DIM; j = j + 1) begin
                x_in[j*ACT_WIDTH +: ACT_WIDTH] = (i + j) % 64;
            end
            token_idx_in = i;
            token_valid_in = 1;
        end
        
        @(posedge clk);
        token_valid_in = 0;
        seq_done_in = 1;
        @(posedge clk);
        seq_done_in = 0;
        
        // Wait for outputs
        repeat(100000) begin
            @(posedge clk);
            if (token_valid_out) begin
                $display("Output token %0d received", token_idx_out);
            end
        end
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #10000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
