// =============================================================================
// SiLens - Approximate GELU Activation Module
// =============================================================================
// Implements GELU (Gaussian Error Linear Unit) using piece-wise linear
// approximation.
//
// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))
//
// Approximation approaches:
//   1. Tanh approximation: GELU(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
//   2. Sigmoid approximation: GELU(x) ≈ x * σ(1.702*x)
//   3. Piece-wise linear: Most hardware-efficient
//
// This implementation uses piece-wise linear approximation:
//   x < -3:     GELU(x) ≈ 0
//   -3 ≤ x < -1: GELU(x) ≈ linear segment 1
//   -1 ≤ x < 0:  GELU(x) ≈ linear segment 2
//   0 ≤ x < 1:   GELU(x) ≈ linear segment 3
//   1 ≤ x < 3:   GELU(x) ≈ linear segment 4
//   x ≥ 3:       GELU(x) ≈ x (identity)
//
// Key GELU values:
//   GELU(-3) ≈ -0.004,  GELU(-2) ≈ -0.045
//   GELU(-1) ≈ -0.159,  GELU(0)  = 0
//   GELU(1)  ≈ 0.841,   GELU(2)  ≈ 1.955
//   GELU(3)  ≈ 2.996
//
// License: Apache 2.0
// =============================================================================

module gelu_approx #(
    parameter WIDTH       = 256,                    // Number of elements
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits in fixed-point
    parameter PARALLEL    = 16                      // Elements processed per cycle
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [WIDTH*ACT_WIDTH-1:0]   x_in,               // Input activations
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Output interface
    output reg  [WIDTH*ACT_WIDTH-1:0]   y_out,              // GELU output
    output reg                          valid_out,
    input  wire                         ready_out
);

    // =========================================================================
    // Fixed-point constants for piece-wise linear approximation
    // =========================================================================
    // Format: Q(ACT_WIDTH-FRAC_BITS).(FRAC_BITS)
    
    // Threshold values in fixed-point (assuming FRAC_BITS=4, so 1.0 = 16)
    localparam signed [ACT_WIDTH-1:0] THRESH_N3 = -3 * (1 << FRAC_BITS);  // -3.0
    localparam signed [ACT_WIDTH-1:0] THRESH_N2 = -2 * (1 << FRAC_BITS);  // -2.0
    localparam signed [ACT_WIDTH-1:0] THRESH_N1 = -1 * (1 << FRAC_BITS);  // -1.0
    localparam signed [ACT_WIDTH-1:0] THRESH_0  = 0;                       // 0.0
    localparam signed [ACT_WIDTH-1:0] THRESH_P1 = 1 * (1 << FRAC_BITS);   // 1.0
    localparam signed [ACT_WIDTH-1:0] THRESH_P2 = 2 * (1 << FRAC_BITS);   // 2.0
    localparam signed [ACT_WIDTH-1:0] THRESH_P3 = 3 * (1 << FRAC_BITS);   // 3.0
    
    // Slope and intercept for each segment (scaled by 2^FRAC_BITS)
    // Segment 1: x in [-3, -2], GELU goes from ~0 to -0.045
    // Slope ≈ -0.045, intercept ≈ -0.135
    localparam signed [ACT_WIDTH:0] SLOPE_1 = -1;   // ≈ -0.045 * 16 ≈ -1
    localparam signed [ACT_WIDTH:0] INTER_1 = -2;   // ≈ -0.135 * 16 ≈ -2
    
    // Segment 2: x in [-2, -1], GELU goes from -0.045 to -0.159
    // Slope ≈ -0.114, intercept ≈ -0.273
    localparam signed [ACT_WIDTH:0] SLOPE_2 = -2;   // ≈ -0.114 * 16 ≈ -2
    localparam signed [ACT_WIDTH:0] INTER_2 = -4;   // ≈ -0.273 * 16 ≈ -4
    
    // Segment 3: x in [-1, 0], GELU goes from -0.159 to 0
    // Slope ≈ 0.159, intercept ≈ 0
    localparam signed [ACT_WIDTH:0] SLOPE_3 = 3;    // ≈ 0.159 * 16 ≈ 3
    localparam signed [ACT_WIDTH:0] INTER_3 = 0;
    
    // Segment 4: x in [0, 1], GELU goes from 0 to 0.841
    // Slope ≈ 0.841, intercept ≈ 0
    localparam signed [ACT_WIDTH:0] SLOPE_4 = 13;   // ≈ 0.841 * 16 ≈ 13
    localparam signed [ACT_WIDTH:0] INTER_4 = 0;
    
    // Segment 5: x in [1, 2], GELU goes from 0.841 to 1.955
    // Slope ≈ 1.114, intercept ≈ -0.273
    localparam signed [ACT_WIDTH:0] SLOPE_5 = 18;   // ≈ 1.114 * 16 ≈ 18
    localparam signed [ACT_WIDTH:0] INTER_5 = -4;   // ≈ -0.273 * 16 ≈ -4
    
    // Segment 6: x in [2, 3], GELU goes from 1.955 to 2.996
    // Slope ≈ 1.041 ≈ 1.0 (almost identity)
    localparam signed [ACT_WIDTH:0] SLOPE_6 = 17;   // ≈ 1.041 * 16 ≈ 17
    localparam signed [ACT_WIDTH:0] INTER_6 = -2;
    
    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE    = 2'd0;
    localparam STATE_COMPUTE = 2'd1;
    localparam STATE_DONE    = 2'd2;
    
    reg [1:0] state;
    reg [$clog2(WIDTH/PARALLEL+1)-1:0] elem_idx;
    localparam NUM_ITERS = (WIDTH + PARALLEL - 1) / PARALLEL;
    
    // =========================================================================
    // GELU piece-wise linear function
    // =========================================================================
    
    function signed [ACT_WIDTH-1:0] gelu_pwl;
        input signed [ACT_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] result;
        reg signed [ACC_WIDTH-1:0] x_ext;
        begin
            x_ext = x;  // Sign extend
            
            if (x < THRESH_N3) begin
                // x < -3: GELU ≈ 0
                result = 0;
            end else if (x < THRESH_N2) begin
                // -3 ≤ x < -2: linear segment 1
                result = (SLOPE_1 * x_ext + (INTER_1 << FRAC_BITS)) >>> FRAC_BITS;
            end else if (x < THRESH_N1) begin
                // -2 ≤ x < -1: linear segment 2
                result = (SLOPE_2 * x_ext + (INTER_2 << FRAC_BITS)) >>> FRAC_BITS;
            end else if (x < THRESH_0) begin
                // -1 ≤ x < 0: linear segment 3
                result = (SLOPE_3 * x_ext + (INTER_3 << FRAC_BITS)) >>> FRAC_BITS;
            end else if (x < THRESH_P1) begin
                // 0 ≤ x < 1: linear segment 4
                result = (SLOPE_4 * x_ext + (INTER_4 << FRAC_BITS)) >>> FRAC_BITS;
            end else if (x < THRESH_P2) begin
                // 1 ≤ x < 2: linear segment 5
                result = (SLOPE_5 * x_ext + (INTER_5 << FRAC_BITS)) >>> FRAC_BITS;
            end else if (x < THRESH_P3) begin
                // 2 ≤ x < 3: linear segment 6
                result = (SLOPE_6 * x_ext + (INTER_6 << FRAC_BITS)) >>> FRAC_BITS;
            end else begin
                // x ≥ 3: GELU ≈ x (identity)
                result = x;
            end
            
            // Saturate to output range
            if (result > ((1 << (ACT_WIDTH-1)) - 1))
                gelu_pwl = (1 << (ACT_WIDTH-1)) - 1;
            else if (result < -(1 << (ACT_WIDTH-1)))
                gelu_pwl = -(1 << (ACT_WIDTH-1));
            else
                gelu_pwl = result[ACT_WIDTH-1:0];
        end
    endfunction
    
    // =========================================================================
    // Main FSM and computation
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    // Input buffer for pipelining
    reg [WIDTH*ACT_WIDTH-1:0] x_buf;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= STATE_IDLE;
            elem_idx  <= 0;
            valid_out <= 1'b0;
            x_buf     <= 0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        x_buf    <= x_in;
                        state    <= STATE_COMPUTE;
                        elem_idx <= 0;
                    end
                end
                
                STATE_COMPUTE: begin
                    // Process PARALLEL elements per cycle
                    if (elem_idx >= NUM_ITERS - 1) begin
                        state <= STATE_DONE;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
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
    // Parallel GELU computation
    // =========================================================================
    
    genvar g;
    generate
        for (g = 0; g < WIDTH; g = g + 1) begin : gen_gelu
            wire signed [ACT_WIDTH-1:0] x_val = $signed(x_buf[g*ACT_WIDTH +: ACT_WIDTH]);
            wire signed [ACT_WIDTH-1:0] y_val = gelu_pwl(x_val);
            
            always @(posedge clk) begin
                if (state == STATE_COMPUTE) begin
                    y_out[g*ACT_WIDTH +: ACT_WIDTH] <= y_val;
                end
            end
        end
    endgenerate

endmodule

// =============================================================================
// Testbench (for simulation)
// =============================================================================
`ifdef SIMULATION

module gelu_approx_tb;
    parameter WIDTH = 16;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 4;
    parameter PARALLEL = 4;
    
    reg                          clk;
    reg                          rst_n;
    reg  [WIDTH*ACT_WIDTH-1:0]   x_in;
    reg                          valid_in;
    wire                         ready_in;
    wire [WIDTH*ACT_WIDTH-1:0]   y_out;
    wire                         valid_out;
    reg                          ready_out;
    
    gelu_approx #(
        .WIDTH(WIDTH),
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
    reg signed [ACT_WIDTH-1:0] in_val, out_val;
    real x_real, y_real, gelu_expected;
    
    // Reference GELU calculation (approximate)
    function real gelu_ref;
        input real x;
        real phi;
        begin
            // Simplified sigmoid approximation: GELU(x) ≈ x * sigmoid(1.702*x)
            phi = 1.0 / (1.0 + $exp(-1.702 * x));
            gelu_ref = x * phi;
        end
    endfunction
    
    initial begin
        $display("GELU Approximation Testbench");
        $display("============================");
        $display("Fixed-point format: Q%0d.%0d", ACT_WIDTH-FRAC_BITS, FRAC_BITS);
        $display("1.0 in fixed-point = %0d", 1 << FRAC_BITS);
        
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
        
        // Test: Range of values from -4 to +4 in fixed-point
        $display("\nTest: GELU across input range");
        $display("%-10s %-10s %-10s %-10s", "x_fp", "x_real", "gelu_out", "gelu_ref");
        $display("%-10s %-10s %-10s %-10s", "------", "------", "--------", "--------");
        
        // Create input vector: -4, -3.5, -3, ..., 3, 3.5, 4
        for (i = 0; i < WIDTH; i = i + 1) begin
            // Range from -4 to +4, 16 values
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = (i - WIDTH/2) * (1 << FRAC_BITS) / 2;
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        
        for (i = 0; i < WIDTH; i = i + 1) begin
            in_val = $signed(x_in[i*ACT_WIDTH +: ACT_WIDTH]);
            out_val = $signed(y_out[i*ACT_WIDTH +: ACT_WIDTH]);
            x_real = $itor(in_val) / $itor(1 << FRAC_BITS);
            y_real = $itor(out_val) / $itor(1 << FRAC_BITS);
            gelu_expected = gelu_ref(x_real);
            $display("%-10d %-10.3f %-10.3f %-10.3f", in_val, x_real, y_real, gelu_expected);
        end
        
        @(posedge clk);
        
        // Test key values
        $display("\nTest: Key GELU values");
        
        // Input: -1, 0, 1 in fixed-point
        for (i = 0; i < WIDTH; i = i + 1) begin
            case (i)
                0: x_in[i*ACT_WIDTH +: ACT_WIDTH] = -1 * (1 << FRAC_BITS);  // -1.0
                1: x_in[i*ACT_WIDTH +: ACT_WIDTH] = 0;                       // 0.0
                2: x_in[i*ACT_WIDTH +: ACT_WIDTH] = 1 * (1 << FRAC_BITS);   // 1.0
                3: x_in[i*ACT_WIDTH +: ACT_WIDTH] = 2 * (1 << FRAC_BITS);   // 2.0
                default: x_in[i*ACT_WIDTH +: ACT_WIDTH] = 0;
            endcase
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        
        $display("x=-1: expected GELU=-0.159, got %0.3f", 
                 $itor($signed(y_out[0*ACT_WIDTH +: ACT_WIDTH])) / $itor(1 << FRAC_BITS));
        $display("x=0:  expected GELU=0.000,  got %0.3f",
                 $itor($signed(y_out[1*ACT_WIDTH +: ACT_WIDTH])) / $itor(1 << FRAC_BITS));
        $display("x=1:  expected GELU=0.841,  got %0.3f",
                 $itor($signed(y_out[2*ACT_WIDTH +: ACT_WIDTH])) / $itor(1 << FRAC_BITS));
        $display("x=2:  expected GELU=1.955,  got %0.3f",
                 $itor($signed(y_out[3*ACT_WIDTH +: ACT_WIDTH])) / $itor(1 << FRAC_BITS));
        
        $display("\nTestbench complete");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #50000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
