// =============================================================================
// SiLens - Approximate Softmax Module
// =============================================================================
// Implements softmax using piece-wise linear approximation of exp().
//
// softmax(x_i) = exp(x_i) / sum(exp(x_j))
//
// Approximation approach:
//   1. Find max value for numerical stability: x_i' = x_i - max(x)
//   2. Approximate exp(x) using piece-wise linear segments
//   3. Compute sum of all exp values
//   4. Divide each exp(x_i) by sum (using reciprocal LUT)
//
// Piece-wise linear exp approximation for x in [-8, 0]:
//   Segment [-8, -4]: exp(x) ≈ 0 (very small)
//   Segment [-4, -2]: exp(x) ≈ 0.0183 + 0.0366*(x+4)
//   Segment [-2, -1]: exp(x) ≈ 0.1353 + 0.2325*(x+2)
//   Segment [-1, 0]:  exp(x) ≈ 0.3679 + 0.6321*(x+1)
//
// License: Apache 2.0
// =============================================================================

module softmax_approx #(
    parameter SEQ_LEN     = 256,                    // Sequence length
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 6,                      // Fractional bits
    parameter PARALLEL    = 16                      // Elements processed per cycle
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [SEQ_LEN*ACT_WIDTH-1:0] x_in,               // Input logits
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Output interface
    output reg  [SEQ_LEN*ACT_WIDTH-1:0] y_out,              // Softmax probabilities
    output reg                          valid_out,
    input  wire                         ready_out
);

    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE       = 3'd0;
    localparam STATE_FIND_MAX   = 3'd1;
    localparam STATE_COMPUTE_EXP = 3'd2;
    localparam STATE_SUM_EXP    = 3'd3;
    localparam STATE_NORMALIZE  = 3'd4;
    localparam STATE_DONE       = 3'd5;
    
    reg [2:0] state;
    reg [$clog2(SEQ_LEN/PARALLEL+1)-1:0] elem_idx;
    localparam NUM_ITERS = (SEQ_LEN + PARALLEL - 1) / PARALLEL;
    
    // =========================================================================
    // Registers
    // =========================================================================
    
    // Input buffer
    reg signed [ACT_WIDTH-1:0] x_buf [0:SEQ_LEN-1];
    
    // Exponential buffer
    reg [ACT_WIDTH+FRAC_BITS-1:0] exp_buf [0:SEQ_LEN-1];
    
    // Max value and sum
    reg signed [ACT_WIDTH-1:0] max_val;
    reg [ACC_WIDTH-1:0] exp_sum;
    reg [ACC_WIDTH-1:0] inv_sum;  // Reciprocal of sum
    
    // =========================================================================
    // Piece-wise linear exponential approximation
    // =========================================================================
    // Input: signed fixed-point, output: unsigned fixed-point
    // Approximates exp(x) for x <= 0
    
    function [ACT_WIDTH+FRAC_BITS-1:0] exp_pwl;
        input signed [ACT_WIDTH-1:0] x;
        reg signed [ACT_WIDTH+FRAC_BITS-1:0] x_fp;
        reg [ACT_WIDTH+FRAC_BITS-1:0] result;
        begin
            x_fp = x <<< FRAC_BITS;  // Convert to higher precision
            
            if (x >= 0) begin
                // For x >= 0, clamp to 1.0 (this is after subtracting max)
                result = (1 << FRAC_BITS);
            end else if (x < -((1 << (ACT_WIDTH-2)))) begin
                // Very negative: exp ≈ 0
                result = 0;
            end else if (x < -4) begin
                // Segment [-8, -4]: very small values
                // exp(x) ≈ 0.018 * exp(0.693*(x+8))
                // Simplified: linear approx near 0
                result = 1;  // Minimal value
            end else if (x < -2) begin
                // Segment [-4, -2]: exp(-4)=0.018, exp(-2)=0.135
                // Slope ≈ 0.058 per unit
                // exp(x) ≈ 0.018 + 0.058*(x+4)
                result = (1 << (FRAC_BITS-6)) + 
                         ((x + 4) * (4 << (FRAC_BITS-6)));
            end else if (x < -1) begin
                // Segment [-2, -1]: exp(-2)=0.135, exp(-1)=0.368
                // Slope ≈ 0.233 per unit
                // exp(x) ≈ 0.135 + 0.233*(x+2)
                result = (9 << (FRAC_BITS-6)) + 
                         ((x + 2) * (15 << (FRAC_BITS-6)));
            end else begin
                // Segment [-1, 0]: exp(-1)=0.368, exp(0)=1.0
                // Slope ≈ 0.632 per unit
                // exp(x) ≈ 0.368 + 0.632*(x+1)
                result = (24 << (FRAC_BITS-6)) + 
                         ((x + 1) * (40 << (FRAC_BITS-6)));
            end
            
            exp_pwl = result;
        end
    endfunction
    
    // =========================================================================
    // Input buffering
    // =========================================================================
    
    integer load_idx;
    always @(posedge clk) begin
        if (state == STATE_IDLE && valid_in) begin
            for (load_idx = 0; load_idx < SEQ_LEN; load_idx = load_idx + 1) begin
                x_buf[load_idx] <= $signed(x_in[load_idx*ACT_WIDTH +: ACT_WIDTH]);
            end
        end
    end
    
    // =========================================================================
    // Find maximum (parallel reduction)
    // =========================================================================
    
    reg signed [ACT_WIDTH-1:0] partial_max [0:PARALLEL-1];
    reg signed [ACT_WIDTH-1:0] max_tree;
    
    integer max_i;
    always @(*) begin
        for (max_i = 0; max_i < PARALLEL; max_i = max_i + 1) begin
            if (elem_idx * PARALLEL + max_i < SEQ_LEN) begin
                partial_max[max_i] = x_buf[elem_idx * PARALLEL + max_i];
            end else begin
                partial_max[max_i] = {1'b1, {(ACT_WIDTH-1){1'b0}}};  // Min value
            end
        end
        
        // Tree reduction for max
        max_tree = partial_max[0];
        for (max_i = 1; max_i < PARALLEL; max_i = max_i + 1) begin
            if (partial_max[max_i] > max_tree)
                max_tree = partial_max[max_i];
        end
    end
    
    // =========================================================================
    // Compute exponentials (parallel)
    // =========================================================================
    
    reg [ACC_WIDTH-1:0] partial_exp_sum;
    integer exp_i;
    
    always @(*) begin
        partial_exp_sum = 0;
        for (exp_i = 0; exp_i < PARALLEL; exp_i = exp_i + 1) begin
            if (elem_idx * PARALLEL + exp_i < SEQ_LEN) begin
                partial_exp_sum = partial_exp_sum + 
                    exp_pwl(x_buf[elem_idx * PARALLEL + exp_i] - max_val);
            end
        end
    end
    
    // =========================================================================
    // Reciprocal approximation for division
    // =========================================================================
    // Compute 1/x using Newton-Raphson or LUT
    // For simplicity, use shift-based approximation
    
    function [ACC_WIDTH-1:0] reciprocal_approx;
        input [ACC_WIDTH-1:0] x;
        reg [5:0] leading_zeros;
        reg [ACC_WIDTH-1:0] result;
        integer ri;
        begin
            // Find leading one position
            leading_zeros = 0;
            for (ri = ACC_WIDTH-1; ri >= 0; ri = ri - 1) begin
                if (x[ri] && leading_zeros == 0)
                    leading_zeros = ACC_WIDTH - 1 - ri;
            end
            
            // Approximate reciprocal: 1/x ≈ 2^(2*FRAC_BITS) / x
            // Using shift: result = (1 << (2*FRAC_BITS + leading_zeros)) >> log2(x)
            if (x == 0)
                result = {ACC_WIDTH{1'b1}};  // Max value for div by 0
            else
                result = ((1 << (2*FRAC_BITS)) / x);  // Simplified division
            
            reciprocal_approx = result;
        end
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    integer norm_idx;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= STATE_IDLE;
            elem_idx  <= 0;
            max_val   <= {1'b1, {(ACT_WIDTH-1){1'b0}}};  // Min value
            exp_sum   <= 0;
            inv_sum   <= 0;
            valid_out <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        state    <= STATE_FIND_MAX;
                        elem_idx <= 0;
                        max_val  <= {1'b1, {(ACT_WIDTH-1){1'b0}}};
                    end
                end
                
                STATE_FIND_MAX: begin
                    // Update max
                    if (max_tree > max_val)
                        max_val <= max_tree;
                    
                    if (elem_idx >= NUM_ITERS - 1) begin
                        state    <= STATE_COMPUTE_EXP;
                        elem_idx <= 0;
                        exp_sum  <= 0;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end
                
                STATE_COMPUTE_EXP: begin
                    // Compute exp(x - max) for each element and store
                    for (norm_idx = 0; norm_idx < PARALLEL; norm_idx = norm_idx + 1) begin
                        if (elem_idx * PARALLEL + norm_idx < SEQ_LEN) begin
                            exp_buf[elem_idx * PARALLEL + norm_idx] <= 
                                exp_pwl(x_buf[elem_idx * PARALLEL + norm_idx] - max_val);
                        end
                    end
                    
                    // Accumulate sum
                    exp_sum <= exp_sum + partial_exp_sum;
                    
                    if (elem_idx >= NUM_ITERS - 1) begin
                        state    <= STATE_SUM_EXP;
                        elem_idx <= 0;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end
                
                STATE_SUM_EXP: begin
                    // Compute reciprocal of sum
                    inv_sum <= reciprocal_approx(exp_sum);
                    state   <= STATE_NORMALIZE;
                    elem_idx <= 0;
                end
                
                STATE_NORMALIZE: begin
                    // Normalize: y = exp(x) * inv_sum
                    for (norm_idx = 0; norm_idx < SEQ_LEN; norm_idx = norm_idx + 1) begin
                        y_out[norm_idx*ACT_WIDTH +: ACT_WIDTH] <= 
                            (exp_buf[norm_idx] * inv_sum) >> FRAC_BITS;
                    end
                    state <= STATE_DONE;
                end
                
                STATE_DONE: begin
                    valid_out <= 1'b1;
                    if (ready_out) begin
                        state <= STATE_IDLE;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule

// =============================================================================
// Testbench (for simulation)
// =============================================================================
`ifdef SIMULATION

module softmax_approx_tb;
    parameter SEQ_LEN = 8;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 6;
    parameter PARALLEL = 4;
    
    reg                          clk;
    reg                          rst_n;
    reg  [SEQ_LEN*ACT_WIDTH-1:0] x_in;
    reg                          valid_in;
    wire                         ready_in;
    wire [SEQ_LEN*ACT_WIDTH-1:0] y_out;
    wire                         valid_out;
    reg                          ready_out;
    
    softmax_approx #(
        .SEQ_LEN(SEQ_LEN),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .PARALLEL(PARALLEL)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(x_in),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .y_out(y_out),
        .valid_out(valid_out),
        .ready_out(ready_out)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    integer i;
    reg signed [ACT_WIDTH-1:0] val;
    reg [ACT_WIDTH-1:0] out_val;
    integer sum;
    
    initial begin
        $display("Softmax Approximation Testbench");
        $display("===============================");
        
        // Initialize
        clk = 0;
        rst_n = 0;
        x_in = 0;
        valid_in = 0;
        ready_out = 1;
        
        // Reset
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Test 1: Equal inputs (should give equal outputs)
        $display("\nTest 1: Equal inputs");
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = 0;  // All zeros
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        
        $display("Input: all zeros");
        $display("Output (should be equal):");
        sum = 0;
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            out_val = y_out[i*ACT_WIDTH +: ACT_WIDTH];
            $write("%3d ", out_val);
            sum = sum + out_val;
        end
        $display(" (sum=%d)", sum);
        
        @(posedge clk);
        
        // Test 2: One hot (one large, rest small)
        $display("\nTest 2: One dominant value");
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            if (i == 0)
                x_in[i*ACT_WIDTH +: ACT_WIDTH] = 10;  // Large positive
            else
                x_in[i*ACT_WIDTH +: ACT_WIDTH] = -10; // Negative
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        
        $display("Input: [10, -10, -10, ...]");
        $display("Output (first should dominate):");
        sum = 0;
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            out_val = y_out[i*ACT_WIDTH +: ACT_WIDTH];
            $write("%3d ", out_val);
            sum = sum + out_val;
        end
        $display(" (sum=%d)", sum);
        
        @(posedge clk);
        
        // Test 3: Increasing values
        $display("\nTest 3: Increasing values");
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = i - (SEQ_LEN/2);
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        
        $display("Input: [-4, -3, -2, -1, 0, 1, 2, 3]");
        $display("Output (should increase):");
        sum = 0;
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            out_val = y_out[i*ACT_WIDTH +: ACT_WIDTH];
            $write("%3d ", out_val);
            sum = sum + out_val;
        end
        $display(" (sum=%d)", sum);
        
        $display("\nTestbench complete");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
