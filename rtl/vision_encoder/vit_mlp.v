// =============================================================================
// SiLens - Vision Transformer MLP Block
// =============================================================================
// Implements the MLP (feedforward) block for Vision Transformer.
//
// Architecture:
//   - Two-layer feedforward network: 768 -> 3072 -> 768
//   - GELU activation between layers
//   - Hardwired ternary weights
//
// MLP formula:
//   hidden = GELU(x * W1 + b1)
//   output = hidden * W2 + b2
//
// Where:
//   W1: 768 x 3072 (expansion)
//   W2: 3072 x 768 (projection)
//   Expansion ratio: 4x
//
// License: Apache 2.0
// =============================================================================

module vit_mlp #(
    parameter DIM         = 768,                    // Input/output dimension
    parameter HIDDEN_DIM  = 3072,                   // Hidden dimension (4x expansion)
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits for fixed-point
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,               // Input vector
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Hardwired ternary weights
    // W1: DIM x HIDDEN_DIM = 768 x 3072 = 2,359,296 weights x 2 bits = 4.5 Mbit
    // W2: HIDDEN_DIM x DIM = 3072 x 768 = 2,359,296 weights x 2 bits = 4.5 Mbit
    input  wire [DIM*HIDDEN_DIM*2-1:0]  w1,                 // First layer weights
    input  wire [HIDDEN_DIM*DIM*2-1:0]  w2,                 // Second layer weights
    
    // Bias terms (optional, can be folded into layer norm)
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0] b1,              // First layer bias
    input  wire [DIM*ACT_WIDTH-1:0]     b2,                 // Second layer bias
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0]     y_out,              // Output vector
    output reg                          valid_out,
    input  wire                         ready_out
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
    
    localparam STATE_IDLE     = 3'd0;
    localparam STATE_FC1      = 3'd1;   // First linear layer
    localparam STATE_GELU     = 3'd2;   // GELU activation
    localparam STATE_FC2      = 3'd3;   // Second linear layer
    localparam STATE_OUTPUT   = 3'd4;
    
    reg [2:0] state;
    
    // =========================================================================
    // Processing indices
    // =========================================================================
    
    reg [$clog2(HIDDEN_DIM)-1:0] hidden_idx;    // Hidden dimension index
    reg [$clog2(DIM)-1:0] out_idx;              // Output dimension index
    reg [$clog2(DIM/PARALLEL+1)-1:0] fc1_iter;  // FC1 MAC iteration
    reg [$clog2(HIDDEN_DIM/PARALLEL+1)-1:0] fc2_iter;  // FC2 MAC iteration
    
    localparam FC1_ITERS = (DIM + PARALLEL - 1) / PARALLEL;
    localparam FC2_ITERS = (HIDDEN_DIM + PARALLEL - 1) / PARALLEL;
    
    // =========================================================================
    // Buffers
    // =========================================================================
    
    // Input buffer
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];
    
    // Hidden layer buffer (after FC1 + GELU)
    reg signed [ACT_WIDTH-1:0] hidden_buf [0:HIDDEN_DIM-1];
    
    // Accumulator for MAC operations
    reg signed [ACC_WIDTH-1:0] mac_accum;
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    // =========================================================================
    // Input loading
    // =========================================================================
    
    integer load_i;
    
    always @(posedge clk) begin
        if (state == STATE_IDLE && valid_in) begin
            for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                x_buf[load_i] <= $signed(x_in[load_i*ACT_WIDTH +: ACT_WIDTH]);
            end
        end
    end
    
    // =========================================================================
    // Ternary MAC computation
    // =========================================================================
    
    // FC1: Compute one hidden dimension at a time
    // hidden[hidden_idx] = sum over input_dim (x[i] * W1[hidden_idx][i])
    
    wire signed [ACC_WIDTH-1:0] fc1_partial;
    reg signed [ACC_WIDTH-1:0] fc1_partial_sum [0:PARALLEL-1];
    
    integer fc1_i;
    always @(*) begin
        for (fc1_i = 0; fc1_i < PARALLEL; fc1_i = fc1_i + 1) begin
            fc1_partial_sum[fc1_i] = 0;
            if (fc1_iter * PARALLEL + fc1_i < DIM) begin
                case (w1[(hidden_idx * DIM + fc1_iter * PARALLEL + fc1_i) * 2 +: 2])
                    W_POS:   fc1_partial_sum[fc1_i] = $signed({1'b0, x_buf[fc1_iter * PARALLEL + fc1_i]});
                    W_NEG:   fc1_partial_sum[fc1_i] = -$signed({1'b0, x_buf[fc1_iter * PARALLEL + fc1_i]});
                    default: fc1_partial_sum[fc1_i] = 0;
                endcase
            end
        end
    end
    
    // Sum partial results
    reg signed [ACC_WIDTH-1:0] fc1_tree;
    integer fc1_t;
    always @(*) begin
        fc1_tree = 0;
        for (fc1_t = 0; fc1_t < PARALLEL; fc1_t = fc1_t + 1) begin
            fc1_tree = fc1_tree + fc1_partial_sum[fc1_t];
        end
    end
    assign fc1_partial = fc1_tree;
    
    // FC2: Compute one output dimension at a time
    // out[out_idx] = sum over hidden_dim (hidden[i] * W2[out_idx][i])
    
    wire signed [ACC_WIDTH-1:0] fc2_partial;
    reg signed [ACC_WIDTH-1:0] fc2_partial_sum [0:PARALLEL-1];
    
    integer fc2_i;
    always @(*) begin
        for (fc2_i = 0; fc2_i < PARALLEL; fc2_i = fc2_i + 1) begin
            fc2_partial_sum[fc2_i] = 0;
            if (fc2_iter * PARALLEL + fc2_i < HIDDEN_DIM) begin
                case (w2[(out_idx * HIDDEN_DIM + fc2_iter * PARALLEL + fc2_i) * 2 +: 2])
                    W_POS:   fc2_partial_sum[fc2_i] = $signed({1'b0, hidden_buf[fc2_iter * PARALLEL + fc2_i]});
                    W_NEG:   fc2_partial_sum[fc2_i] = -$signed({1'b0, hidden_buf[fc2_iter * PARALLEL + fc2_i]});
                    default: fc2_partial_sum[fc2_i] = 0;
                endcase
            end
        end
    end
    
    // Sum partial results for FC2
    reg signed [ACC_WIDTH-1:0] fc2_tree;
    integer fc2_t;
    always @(*) begin
        fc2_tree = 0;
        for (fc2_t = 0; fc2_t < PARALLEL; fc2_t = fc2_t + 1) begin
            fc2_tree = fc2_tree + fc2_partial_sum[fc2_t];
        end
    end
    assign fc2_partial = fc2_tree;
    
    // =========================================================================
    // GELU activation (piece-wise linear approximation)
    // =========================================================================
    
    function signed [ACT_WIDTH-1:0] gelu_approx;
        input signed [ACC_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] x_scaled;
        reg signed [ACC_WIDTH-1:0] result;
        begin
            x_scaled = x >>> FRAC_BITS;  // Scale to integer range
            
            // GELU approximation using piece-wise linear
            // GELU(x) ≈ x * sigmoid(1.702 * x)
            // Simplified: 
            //   x < -3: 0
            //   -3 <= x < 0: ~0.5 * x * (1 + x/3)
            //   x >= 0: x * (1 - e^(-1.702*x)) ≈ x for large x
            
            if (x_scaled < -48) begin
                // x < -3 (in fixed point with FRAC_BITS=4)
                result = 0;
            end else if (x_scaled < 0) begin
                // Linear ramp from 0 to 0.5*x
                result = (x_scaled * (16 + x_scaled)) >>> 5;  // Approximate
            end else begin
                // x >= 0: approximately identity with slight reduction
                result = (x_scaled * 14) >>> 4;  // ~0.875 * x
            end
            
            gelu_approx = saturate(result);
        end
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer gelu_i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            hidden_idx <= 0;
            out_idx <= 0;
            fc1_iter <= 0;
            fc2_iter <= 0;
            mac_accum <= 0;
            valid_out <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        state <= STATE_FC1;
                        hidden_idx <= 0;
                        fc1_iter <= 0;
                        mac_accum <= 0;
                    end
                end
                
                STATE_FC1: begin
                    // Accumulate partial sum for current hidden dimension
                    mac_accum <= mac_accum + fc1_partial;
                    
                    if (fc1_iter >= FC1_ITERS - 1) begin
                        // Finished computing one hidden dimension
                        // Add bias and apply GELU
                        hidden_buf[hidden_idx] <= gelu_approx(
                            mac_accum + fc1_partial + 
                            $signed(b1[hidden_idx*ACT_WIDTH +: ACT_WIDTH])
                        );
                        
                        fc1_iter <= 0;
                        mac_accum <= 0;
                        
                        if (hidden_idx >= HIDDEN_DIM - 1) begin
                            hidden_idx <= 0;
                            state <= STATE_FC2;
                            out_idx <= 0;
                            fc2_iter <= 0;
                        end else begin
                            hidden_idx <= hidden_idx + 1;
                        end
                    end else begin
                        fc1_iter <= fc1_iter + 1;
                    end
                end
                
                STATE_FC2: begin
                    // Accumulate partial sum for current output dimension
                    mac_accum <= mac_accum + fc2_partial;
                    
                    if (fc2_iter >= FC2_ITERS - 1) begin
                        // Finished computing one output dimension
                        // Add bias and store result
                        y_out[out_idx*ACT_WIDTH +: ACT_WIDTH] <= saturate(
                            mac_accum + fc2_partial +
                            $signed(b2[out_idx*ACT_WIDTH +: ACT_WIDTH])
                        );
                        
                        fc2_iter <= 0;
                        mac_accum <= 0;
                        
                        if (out_idx >= DIM - 1) begin
                            out_idx <= 0;
                            state <= STATE_OUTPUT;
                        end else begin
                            out_idx <= out_idx + 1;
                        end
                    end else begin
                        fc2_iter <= fc2_iter + 1;
                    end
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
    // Saturation function
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

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module vit_mlp_tb;
    parameter DIM = 32;
    parameter HIDDEN_DIM = 128;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 4;
    parameter PARALLEL = 8;
    
    reg clk, rst_n;
    reg [DIM*ACT_WIDTH-1:0] x_in;
    reg valid_in;
    wire ready_in;
    reg [DIM*HIDDEN_DIM*2-1:0] w1;
    reg [HIDDEN_DIM*DIM*2-1:0] w2;
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] b1;
    reg [DIM*ACT_WIDTH-1:0] b2;
    wire [DIM*ACT_WIDTH-1:0] y_out;
    wire valid_out;
    reg ready_out;
    
    vit_mlp #(
        .DIM(DIM),
        .HIDDEN_DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i;
    
    initial begin
        $display("ViT MLP Testbench");
        $display("=================");
        $display("DIM=%0d, HIDDEN_DIM=%0d", DIM, HIDDEN_DIM);
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        valid_in = 0;
        ready_out = 1;
        
        // Initialize weights to +1
        w1 = {(DIM*HIDDEN_DIM){2'b01}};
        w2 = {(HIDDEN_DIM*DIM){2'b01}};
        b1 = 0;
        b2 = 0;
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Test with simple input
        for (i = 0; i < DIM; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = (i % 16);
        end
        
        $display("Sending input vector...");
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for output
        repeat(100000) begin
            @(posedge clk);
            if (valid_out) begin
                $display("Output received!");
                for (i = 0; i < DIM; i = i + 1) begin
                    $write("%3d ", $signed(y_out[i*ACT_WIDTH +: ACT_WIDTH]));
                    if ((i+1) % 8 == 0) $display("");
                end
                break;
            end
        end
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
