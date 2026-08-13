// =============================================================================
// SiLens - Language Model Decoder Block
// =============================================================================
// Implements a single decoder transformer block for the language model.
//
// Architecture (Pre-RMSNorm style):
//   1. RMSNorm -> Self-Attention -> Residual Add
//   2. RMSNorm -> MLP (SwiGLU) -> Residual Add
//
// Block formula:
//   x' = x + Attention(RMSNorm(x))
//   y  = x' + MLP(RMSNorm(x'))
//
// RMSNorm (Root Mean Square Normalization):
//   RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
//
// This is the standard LLaMA/SmolLM decoder block structure.
//
// License: Apache 2.0
// =============================================================================

module llm_block #(
    parameter DIM         = 576,                    // Model dimension
    parameter NUM_HEADS   = 9,                      // Number of attention heads
    parameter HEAD_DIM    = 64,                     // Dimension per head
    parameter MLP_DIM     = 1536,                   // MLP hidden dimension
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
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0] position,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Cache control
    input  wire                         cache_clear,
    
    // RMSNorm weights
    input  wire [DIM*ACT_WIDTH-1:0]     rms_attn_gamma,     // Pre-attention RMSNorm
    input  wire [DIM*ACT_WIDTH-1:0]     rms_mlp_gamma,      // Pre-MLP RMSNorm
    
    // Attention weights
    input  wire [DIM*DIM*2-1:0]         attn_w_q,
    input  wire [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] attn_w_k,
    input  wire [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] attn_w_v,
    input  wire [DIM*DIM*2-1:0]         attn_w_o,
    
    // RoPE frequencies
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] rope_cos,
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] rope_sin,
    
    // MLP weights
    input  wire [DIM*MLP_DIM*2-1:0]     mlp_w_gate,
    input  wire [DIM*MLP_DIM*2-1:0]     mlp_w_up,
    input  wire [MLP_DIM*DIM*2-1:0]     mlp_w_down,
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0]     y_out,
    output reg                          valid_out,
    input  wire                         ready_out
);

    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE       = 4'd0;
    localparam STATE_RMS1       = 4'd1;   // Pre-attention RMSNorm
    localparam STATE_ATTN       = 4'd2;   // Self-attention
    localparam STATE_RESIDUAL1  = 4'd3;   // First residual add
    localparam STATE_RMS2       = 4'd4;   // Pre-MLP RMSNorm
    localparam STATE_MLP        = 4'd5;   // MLP
    localparam STATE_RESIDUAL2  = 4'd6;   // Second residual add
    localparam STATE_OUTPUT     = 4'd7;
    
    reg [3:0] state;
    
    // =========================================================================
    // Buffers
    // =========================================================================
    
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];           // Input buffer
    reg signed [ACT_WIDTH-1:0] rms1_buf [0:DIM-1];        // After first RMSNorm
    reg signed [ACT_WIDTH-1:0] attn_buf [0:DIM-1];        // After attention
    reg signed [ACT_WIDTH-1:0] res1_buf [0:DIM-1];        // After first residual
    reg signed [ACT_WIDTH-1:0] rms2_buf [0:DIM-1];        // After second RMSNorm
    reg signed [ACT_WIDTH-1:0] mlp_buf [0:DIM-1];         // After MLP
    
    reg [$clog2(MAX_SEQ_LEN)-1:0] pos_reg;
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    // =========================================================================
    // RMSNorm computation
    // =========================================================================
    // RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
    
    function signed [ACT_WIDTH-1:0] rms_normalize;
        input signed [ACT_WIDTH-1:0] x_val;
        input signed [ACT_WIDTH-1:0] gamma;
        input signed [ACC_WIDTH-1:0] inv_rms;  // Precomputed 1/rms
        reg signed [ACC_WIDTH-1:0] result;
        begin
            result = ($signed(x_val) * $signed(inv_rms) * $signed(gamma)) >>> (2*FRAC_BITS);
            rms_normalize = saturate(result);
        end
    endfunction
    
    // Compute mean square
    function signed [ACC_WIDTH-1:0] compute_mean_sq;
        input integer dummy;  // Verilog function needs input
        integer cms_i;
        reg signed [ACC_WIDTH-1:0] sum;
        begin
            sum = 0;
            for (cms_i = 0; cms_i < DIM; cms_i = cms_i + 1) begin
                sum = sum + ($signed(x_buf[cms_i]) * $signed(x_buf[cms_i]));
            end
            compute_mean_sq = sum / DIM;
        end
    endfunction
    
    // Inverse square root approximation (Newton-Raphson)
    function signed [ACC_WIDTH-1:0] inv_sqrt_approx;
        input signed [ACC_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] y;
        reg signed [ACC_WIDTH-1:0] y_sq;
        reg signed [ACC_WIDTH-1:0] three;
        integer iter;
        begin
            three = 3 << FRAC_BITS;
            y = 1 << FRAC_BITS;  // Initial guess = 1.0
            
            // 3 Newton-Raphson iterations
            for (iter = 0; iter < 3; iter = iter + 1) begin
                y_sq = (y * y) >>> FRAC_BITS;
                y = (y * (three - ((x * y_sq) >>> FRAC_BITS))) >>> (FRAC_BITS + 1);
            end
            
            inv_sqrt_approx = y;
        end
    endfunction
    
    // =========================================================================
    // Attention instance
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] attn_x_in;
    reg attn_valid_in;
    wire attn_ready_in;
    reg attn_cache_clear;
    wire [DIM*ACT_WIDTH-1:0] attn_y_out;
    wire attn_valid_out;
    reg attn_ready_out;
    
    llm_attention #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .KV_HEADS(KV_HEADS),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) u_attention (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(attn_x_in),
        .position(pos_reg),
        .valid_in(attn_valid_in),
        .ready_in(attn_ready_in),
        .cache_clear(attn_cache_clear),
        .w_q(attn_w_q),
        .w_k(attn_w_k),
        .w_v(attn_w_v),
        .w_o(attn_w_o),
        .rope_cos(rope_cos),
        .rope_sin(rope_sin),
        .y_out(attn_y_out),
        .valid_out(attn_valid_out),
        .ready_out(attn_ready_out)
    );
    
    // =========================================================================
    // MLP instance
    // =========================================================================
    
    reg [DIM*ACT_WIDTH-1:0] mlp_x_in;
    reg mlp_valid_in;
    wire mlp_ready_in;
    wire [DIM*ACT_WIDTH-1:0] mlp_y_out;
    wire mlp_valid_out;
    reg mlp_ready_out;
    
    llm_mlp #(
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
        .w_gate(mlp_w_gate),
        .w_up(mlp_w_up),
        .w_down(mlp_w_down),
        .y_out(mlp_y_out),
        .valid_out(mlp_valid_out),
        .ready_out(mlp_ready_out)
    );
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer load_i, rms_i;
    reg signed [ACC_WIDTH-1:0] mean_sq_val;
    reg signed [ACC_WIDTH-1:0] inv_rms;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            valid_out <= 1'b0;
            attn_valid_in <= 1'b0;
            attn_cache_clear <= 1'b0;
            attn_ready_out <= 1'b0;
            mlp_valid_in <= 1'b0;
            mlp_ready_out <= 1'b0;
        end else begin
            // Default control signals
            attn_valid_in <= 1'b0;
            attn_cache_clear <= 1'b0;
            mlp_valid_in <= 1'b0;
            
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    
                    if (cache_clear) begin
                        attn_cache_clear <= 1'b1;
                    end
                    
                    if (valid_in) begin
                        // Load input
                        for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                            x_buf[load_i] <= $signed(x_in[load_i*ACT_WIDTH +: ACT_WIDTH]);
                        end
                        pos_reg <= position;
                        state <= STATE_RMS1;
                    end
                end
                
                STATE_RMS1: begin
                    // Compute RMSNorm for attention input
                    mean_sq_val = compute_mean_sq(0);
                    inv_rms = inv_sqrt_approx(mean_sq_val + 1);  // +1 for eps
                    
                    // Apply normalization
                    for (rms_i = 0; rms_i < DIM; rms_i = rms_i + 1) begin
                        rms1_buf[rms_i] <= rms_normalize(
                            x_buf[rms_i],
                            rms_attn_gamma[rms_i*ACT_WIDTH +: ACT_WIDTH],
                            inv_rms
                        );
                    end
                    
                    state <= STATE_ATTN;
                end
                
                STATE_ATTN: begin
                    // Pack input for attention
                    for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                        attn_x_in[load_i*ACT_WIDTH +: ACT_WIDTH] <= rms1_buf[load_i];
                    end
                    attn_valid_in <= 1'b1;
                    attn_ready_out <= 1'b1;
                    
                    if (attn_valid_out) begin
                        // Store attention output
                        for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                            attn_buf[load_i] <= attn_y_out[load_i*ACT_WIDTH +: ACT_WIDTH];
                        end
                        state <= STATE_RESIDUAL1;
                    end
                end
                
                STATE_RESIDUAL1: begin
                    // Add residual: res1 = x + attn(rms(x))
                    for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                        res1_buf[load_i] <= saturate_add(x_buf[load_i], attn_buf[load_i]);
                    end
                    state <= STATE_RMS2;
                end
                
                STATE_RMS2: begin
                    // Compute RMSNorm for MLP input
                    // Use res1_buf as input
                    begin : rms2_block
                        reg signed [ACC_WIDTH-1:0] ms2;
                        reg signed [ACC_WIDTH-1:0] inv2;
                        integer m2i;
                        
                        ms2 = 0;
                        for (m2i = 0; m2i < DIM; m2i = m2i + 1) begin
                            ms2 = ms2 + ($signed(res1_buf[m2i]) * $signed(res1_buf[m2i]));
                        end
                        ms2 = ms2 / DIM;
                        inv2 = inv_sqrt_approx(ms2 + 1);
                        
                        for (m2i = 0; m2i < DIM; m2i = m2i + 1) begin
                            rms2_buf[m2i] <= rms_normalize(
                                res1_buf[m2i],
                                rms_mlp_gamma[m2i*ACT_WIDTH +: ACT_WIDTH],
                                inv2
                            );
                        end
                    end
                    
                    state <= STATE_MLP;
                end
                
                STATE_MLP: begin
                    // Pack input for MLP
                    for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                        mlp_x_in[load_i*ACT_WIDTH +: ACT_WIDTH] <= rms2_buf[load_i];
                    end
                    mlp_valid_in <= 1'b1;
                    mlp_ready_out <= 1'b1;
                    
                    if (mlp_valid_out) begin
                        // Store MLP output
                        for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                            mlp_buf[load_i] <= mlp_y_out[load_i*ACT_WIDTH +: ACT_WIDTH];
                        end
                        state <= STATE_RESIDUAL2;
                    end
                end
                
                STATE_RESIDUAL2: begin
                    // Add residual: y = res1 + mlp(rms(res1))
                    for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                        y_out[load_i*ACT_WIDTH +: ACT_WIDTH] <= 
                            saturate_add(res1_buf[load_i], mlp_buf[load_i]);
                    end
                    state <= STATE_OUTPUT;
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
    // Helper functions
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
    
    function signed [ACT_WIDTH-1:0] saturate_add;
        input signed [ACT_WIDTH-1:0] a;
        input signed [ACT_WIDTH-1:0] b;
        reg signed [ACT_WIDTH:0] sum;
        begin
            sum = $signed({a[ACT_WIDTH-1], a}) + $signed({b[ACT_WIDTH-1], b});
            saturate_add = saturate(sum);
        end
    endfunction

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module llm_block_tb;
    parameter DIM = 32;
    parameter NUM_HEADS = 2;
    parameter HEAD_DIM = 16;
    parameter MLP_DIM = 64;
    parameter MAX_SEQ_LEN = 32;
    parameter KV_HEADS = 2;
    parameter ACT_WIDTH = 8;
    
    reg clk, rst_n;
    reg [DIM*ACT_WIDTH-1:0] x_in;
    reg [$clog2(MAX_SEQ_LEN)-1:0] position;
    reg valid_in;
    wire ready_in;
    reg cache_clear;
    
    reg [DIM*ACT_WIDTH-1:0] rms_attn_gamma, rms_mlp_gamma;
    reg [DIM*DIM*2-1:0] attn_w_q, attn_w_o;
    reg [DIM*(KV_HEADS*HEAD_DIM)*2-1:0] attn_w_k, attn_w_v;
    reg [HEAD_DIM*ACT_WIDTH-1:0] rope_cos, rope_sin;
    reg [DIM*MLP_DIM*2-1:0] mlp_w_gate, mlp_w_up;
    reg [MLP_DIM*DIM*2-1:0] mlp_w_down;
    
    wire [DIM*ACT_WIDTH-1:0] y_out;
    wire valid_out;
    reg ready_out;
    
    llm_block #(
        .DIM(DIM),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MLP_DIM(MLP_DIM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .KV_HEADS(KV_HEADS),
        .ACT_WIDTH(ACT_WIDTH)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i;
    
    initial begin
        $display("LLM Block Testbench");
        $display("===================");
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        position = 0;
        valid_in = 0;
        cache_clear = 0;
        ready_out = 1;
        
        // Initialize
        rms_attn_gamma = {DIM{8'd16}};
        rms_mlp_gamma = {DIM{8'd16}};
        attn_w_q = {(DIM*DIM){2'b01}};
        attn_w_k = {(DIM*KV_HEADS*HEAD_DIM){2'b01}};
        attn_w_v = {(DIM*KV_HEADS*HEAD_DIM){2'b01}};
        attn_w_o = {(DIM*DIM){2'b01}};
        rope_cos = {HEAD_DIM{8'd16}};
        rope_sin = 0;
        mlp_w_gate = {(DIM*MLP_DIM){2'b01}};
        mlp_w_up = {(DIM*MLP_DIM){2'b01}};
        mlp_w_down = {(MLP_DIM*DIM){2'b01}};
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Clear cache
        cache_clear = 1;
        @(posedge clk);
        cache_clear = 0;
        
        // Process tokens
        for (i = 0; i < 3; i = i + 1) begin
            while (!ready_in) @(posedge clk);
            
            x_in = {DIM{8'd10}};
            position = i;
            valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            
            while (!valid_out) @(posedge clk);
            $display("Token %0d processed", i);
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
