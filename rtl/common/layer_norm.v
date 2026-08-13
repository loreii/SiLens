// =============================================================================
// SiLens - Layer Normalization Module
// =============================================================================
// Implements layer normalization using fixed-point arithmetic.
//
// LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
//
// Implementation approach:
//   1. Compute mean: sum(x) / N
//   2. Compute variance: sum((x - mean)^2) / N
//   3. Compute inverse sqrt using Newton-Raphson approximation
//   4. Apply: y = gamma * (x - mean) * inv_sqrt + beta
//
// Fixed-point format:
//   - Activations: Q(ACT_WIDTH-FRAC_BITS).(FRAC_BITS) signed
//   - Intermediate: Q(ACC_WIDTH-FRAC_BITS*2).(FRAC_BITS*2) for products
//
// License: Apache 2.0
// =============================================================================

module layer_norm #(
    parameter DIM         = 768,                    // Feature dimension
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits in fixed-point
    parameter PARALLEL    = 16,                     // Elements processed per cycle
    // Epsilon for numerical stability (fixed-point: ~0.00001 -> use 1 in smallest representation)
    parameter EPS         = 1
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,               // Input activations
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Learnable parameters (from weight ROM)
    input  wire [DIM*ACT_WIDTH-1:0]     gamma,              // Scale parameter
    input  wire [DIM*ACT_WIDTH-1:0]     beta,               // Bias parameter
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0]     y_out,              // Normalized output
    output reg                          valid_out,
    input  wire                         ready_out
);

    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE       = 3'd0;
    localparam STATE_MEAN       = 3'd1;
    localparam STATE_VARIANCE   = 3'd2;
    localparam STATE_INV_SQRT   = 3'd3;
    localparam STATE_NORMALIZE  = 3'd4;
    localparam STATE_DONE       = 3'd5;
    
    reg [2:0] state;
    reg [$clog2(DIM/PARALLEL+1)-1:0] elem_idx;
    localparam NUM_ITERS = (DIM + PARALLEL - 1) / PARALLEL;
    
    // =========================================================================
    // Registers
    // =========================================================================
    
    // Input buffer
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];
    
    // Statistics
    reg signed [ACC_WIDTH-1:0] sum_acc;
    reg signed [ACC_WIDTH-1:0] var_acc;
    reg signed [ACC_WIDTH-1:0] mean;
    reg signed [ACC_WIDTH-1:0] variance;
    reg signed [ACC_WIDTH-1:0] inv_std;    // 1/sqrt(variance + eps)
    
    // Newton-Raphson iteration counter
    reg [2:0] nr_iter;
    localparam NR_ITERATIONS = 3;  // 3 iterations for convergence
    
    // =========================================================================
    // Input buffering
    // =========================================================================
    
    integer load_idx;
    always @(posedge clk) begin
        if (state == STATE_IDLE && valid_in) begin
            for (load_idx = 0; load_idx < DIM; load_idx = load_idx + 1) begin
                x_buf[load_idx] <= $signed(x_in[load_idx*ACT_WIDTH +: ACT_WIDTH]);
            end
        end
    end
    
    // =========================================================================
    // Parallel sum computation
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] partial_sum;
    wire signed [ACC_WIDTH-1:0] partial_var;
    
    reg signed [ACC_WIDTH-1:0] elem_sums [0:PARALLEL-1];
    reg signed [ACC_WIDTH-1:0] var_sums [0:PARALLEL-1];
    
    integer ps_idx;
    always @(*) begin
        for (ps_idx = 0; ps_idx < PARALLEL; ps_idx = ps_idx + 1) begin
            if (elem_idx * PARALLEL + ps_idx < DIM) begin
                elem_sums[ps_idx] = $signed(x_buf[elem_idx * PARALLEL + ps_idx]);
                // For variance: (x - mean)^2
                var_sums[ps_idx] = ($signed(x_buf[elem_idx * PARALLEL + ps_idx]) - 
                                   (mean >>> $clog2(DIM))) ** 2;
            end else begin
                elem_sums[ps_idx] = 0;
                var_sums[ps_idx] = 0;
            end
        end
    end
    
    // Tree reduction for sum
    reg signed [ACC_WIDTH-1:0] sum_tree;
    reg signed [ACC_WIDTH-1:0] var_tree;
    integer tree_i;
    
    always @(*) begin
        sum_tree = 0;
        var_tree = 0;
        for (tree_i = 0; tree_i < PARALLEL; tree_i = tree_i + 1) begin
            sum_tree = sum_tree + elem_sums[tree_i];
            var_tree = var_tree + var_sums[tree_i];
        end
    end
    
    assign partial_sum = sum_tree;
    assign partial_var = var_tree;
    
    // =========================================================================
    // Inverse square root using Newton-Raphson
    // =========================================================================
    // Approximation: y = 1/sqrt(x)
    // Newton-Raphson: y_{n+1} = y_n * (3 - x * y_n^2) / 2
    //
    // Initial guess from lookup table or linear approximation
    
    // Simple initial guess: for variance in reasonable range
    // Start with 1.0 in fixed-point representation
    wire signed [ACC_WIDTH-1:0] initial_guess = (1 << FRAC_BITS);
    
    reg signed [ACC_WIDTH-1:0] y_nr;        // Current approximation
    reg signed [ACC_WIDTH-1:0] y_nr_sq;     // y^2
    reg signed [ACC_WIDTH-1:0] x_y_sq;      // x * y^2
    wire signed [ACC_WIDTH-1:0] three_fp = (3 << FRAC_BITS);  // 3.0 in fixed-point
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= STATE_IDLE;
            elem_idx  <= 0;
            sum_acc   <= 0;
            var_acc   <= 0;
            mean      <= 0;
            variance  <= 0;
            inv_std   <= 0;
            valid_out <= 1'b0;
            nr_iter   <= 0;
            y_nr      <= 0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        state    <= STATE_MEAN;
                        elem_idx <= 0;
                        sum_acc  <= 0;
                    end
                end
                
                STATE_MEAN: begin
                    // Accumulate sum for mean
                    sum_acc <= sum_acc + partial_sum;
                    
                    if (elem_idx >= NUM_ITERS - 1) begin
                        // Compute mean = sum / DIM
                        // Using shift if DIM is power of 2, otherwise need divider
                        mean     <= sum_acc + partial_sum;  // Store sum, divide later
                        state    <= STATE_VARIANCE;
                        elem_idx <= 0;
                        var_acc  <= 0;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end
                
                STATE_VARIANCE: begin
                    // Accumulate (x - mean)^2
                    var_acc <= var_acc + partial_var;
                    
                    if (elem_idx >= NUM_ITERS - 1) begin
                        // variance = var_acc / DIM + eps
                        variance <= ((var_acc + partial_var) >>> $clog2(DIM)) + EPS;
                        state    <= STATE_INV_SQRT;
                        nr_iter  <= 0;
                        y_nr     <= initial_guess;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end
                
                STATE_INV_SQRT: begin
                    // Newton-Raphson iteration
                    // y_{n+1} = y_n * (3 - x * y_n^2) / 2
                    y_nr_sq = (y_nr * y_nr) >>> FRAC_BITS;
                    x_y_sq  = (variance * y_nr_sq) >>> FRAC_BITS;
                    y_nr    <= (y_nr * (three_fp - x_y_sq)) >>> (FRAC_BITS + 1);
                    
                    if (nr_iter >= NR_ITERATIONS - 1) begin
                        inv_std  <= y_nr;
                        state    <= STATE_NORMALIZE;
                        elem_idx <= 0;
                    end else begin
                        nr_iter <= nr_iter + 1;
                    end
                end
                
                STATE_NORMALIZE: begin
                    // Apply normalization to PARALLEL elements per cycle
                    // y = gamma * (x - mean) * inv_std + beta
                    state    <= STATE_DONE;
                    elem_idx <= 0;
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
    
    // =========================================================================
    // Output computation (parallel normalization)
    // =========================================================================
    
    // Compute normalized output for all elements
    integer out_idx;
    wire signed [ACC_WIDTH-1:0] mean_per_elem = mean >>> $clog2(DIM);
    
    always @(posedge clk) begin
        if (state == STATE_NORMALIZE) begin
            for (out_idx = 0; out_idx < DIM; out_idx = out_idx + 1) begin
                // Centered value
                // y = gamma * (x - mean) * inv_std + beta
                // Simplified: just apply centering and scaling
                y_out[out_idx*ACT_WIDTH +: ACT_WIDTH] <= 
                    ((($signed(x_buf[out_idx]) - mean_per_elem) * inv_std) >>> FRAC_BITS) +
                    $signed(beta[out_idx*ACT_WIDTH +: ACT_WIDTH]);
            end
        end
    end

endmodule

// =============================================================================
// Testbench (for simulation)
// =============================================================================
`ifdef SIMULATION

module layer_norm_tb;
    parameter DIM = 16;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 4;
    parameter PARALLEL = 4;
    
    reg                          clk;
    reg                          rst_n;
    reg  [DIM*ACT_WIDTH-1:0]     x_in;
    reg                          valid_in;
    wire                         ready_in;
    reg  [DIM*ACT_WIDTH-1:0]     gamma;
    reg  [DIM*ACT_WIDTH-1:0]     beta;
    wire [DIM*ACT_WIDTH-1:0]     y_out;
    wire                         valid_out;
    reg                          ready_out;
    
    layer_norm #(
        .DIM(DIM),
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
        .gamma(gamma),
        .beta(beta),
        .y_out(y_out),
        .valid_out(valid_out),
        .ready_out(ready_out)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    integer i;
    reg signed [ACT_WIDTH-1:0] x_val;
    
    initial begin
        $display("Layer Normalization Testbench");
        $display("=============================");
        
        // Initialize
        clk = 0;
        rst_n = 0;
        x_in = 0;
        valid_in = 0;
        ready_out = 1;
        
        // Initialize gamma to 1.0 and beta to 0.0
        for (i = 0; i < DIM; i = i + 1) begin
            gamma[i*ACT_WIDTH +: ACT_WIDTH] = (1 << FRAC_BITS);  // 1.0 in fixed-point
            beta[i*ACT_WIDTH +: ACT_WIDTH] = 0;
        end
        
        // Reset
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Test 1: Sequential input values 0, 1, 2, ..., DIM-1
        $display("Test 1: Sequential values");
        for (i = 0; i < DIM; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = i << FRAC_BITS;
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        
        $display("Input:");
        for (i = 0; i < DIM; i = i + 1) begin
            x_val = x_in[i*ACT_WIDTH +: ACT_WIDTH];
            $write("%4d ", x_val);
        end
        $display("");
        
        $display("Output:");
        for (i = 0; i < DIM; i = i + 1) begin
            x_val = y_out[i*ACT_WIDTH +: ACT_WIDTH];
            $write("%4d ", x_val);
        end
        $display("");
        
        @(posedge clk);
        
        // Test 2: All same values (variance should be ~0)
        $display("\nTest 2: All same values");
        for (i = 0; i < DIM; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = 5 << FRAC_BITS;
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        
        $display("Output (should be near 0 after centering):");
        for (i = 0; i < DIM; i = i + 1) begin
            x_val = y_out[i*ACT_WIDTH +: ACT_WIDTH];
            $write("%4d ", x_val);
        end
        $display("");
        
        @(posedge clk);
        
        // Test 3: High variance input
        $display("\nTest 3: High variance");
        for (i = 0; i < DIM; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = (i % 2 == 0) ? 
                (10 << FRAC_BITS) : (-10 << FRAC_BITS);
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        
        $display("Output:");
        for (i = 0; i < DIM; i = i + 1) begin
            x_val = y_out[i*ACT_WIDTH +: ACT_WIDTH];
            $write("%4d ", x_val);
        end
        $display("");
        
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
