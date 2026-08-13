// =============================================================================
// SiLens - Binary Dot Product Module
// =============================================================================
// Computes dot product between binary activations and binary weights using
// XNOR + popcount. For binary neural networks where weights are {-1, +1}
// encoded as {0, 1}.
//
// Mathematical basis:
//   For binary values a, w ∈ {-1, +1} encoded as {0, 1}:
//   a * w = 2 * XNOR(a, w) - 1
//   
//   For vector dot product:
//   sum(a_i * w_i) = 2 * popcount(XNOR(a, w)) - N
//
// License: Apache 2.0
// =============================================================================

module binary_dot_product #(
    parameter WIDTH       = 512,                    // Number of binary elements
    parameter ACT_WIDTH   = 8,                      // Output activation width
    parameter ACC_WIDTH   = 32,                     // Accumulator width
    parameter OUT_WIDTH   = $clog2(WIDTH) + 2       // Output width for signed result
)(
    input  wire                 clk,
    input  wire                 rst_n,
    
    // Input interface
    input  wire [WIDTH-1:0]     act_in,             // Binary activations (packed)
    input  wire [WIDTH-1:0]     weight_in,          // Binary weights (packed)
    input  wire                 valid_in,
    output wire                 ready_in,
    
    // Output interface
    output reg  [ACC_WIDTH-1:0] result,             // Signed dot product result
    output reg                  valid_out,
    input  wire                 ready_out
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    
    localparam POPCNT_WIDTH = $clog2(WIDTH) + 1;
    
    // XNOR result
    wire [WIDTH-1:0] xnor_result;
    
    // Popcount output
    wire [POPCNT_WIDTH-1:0] ones_count;
    
    // Pipeline registers
    reg [POPCNT_WIDTH-1:0] ones_count_r;
    reg                    valid_pipe;
    
    // =========================================================================
    // XNOR operation - combinational
    // =========================================================================
    
    assign xnor_result = ~(act_in ^ weight_in);
    
    // =========================================================================
    // Population count
    // =========================================================================
    
    popcount #(
        .WIDTH(WIDTH)
    ) u_popcount (
        .in(xnor_result),
        .count(ones_count)
    );
    
    // =========================================================================
    // Pipeline stage and output computation
    // =========================================================================
    
    // Ready signal - simple backpressure
    assign ready_in = ready_out || !valid_out;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipe   <= 1'b0;
            ones_count_r <= {POPCNT_WIDTH{1'b0}};
            valid_out    <= 1'b0;
            result       <= {ACC_WIDTH{1'b0}};
        end else begin
            // Pipeline stage 1: Register popcount result
            if (ready_in) begin
                valid_pipe   <= valid_in;
                ones_count_r <= ones_count;
            end
            
            // Pipeline stage 2: Compute final result
            // result = 2 * popcount - WIDTH (signed arithmetic)
            if (ready_out || !valid_out) begin
                valid_out <= valid_pipe;
                if (valid_pipe) begin
                    // Signed extension: result = 2*ones_count - WIDTH
                    // This gives range [-(WIDTH), +(WIDTH)]
                    result <= $signed({1'b0, ones_count_r, 1'b0}) - $signed(WIDTH);
                end
            end
        end
    end

endmodule

// =============================================================================
// Testbench (for simulation)
// =============================================================================
`ifdef SIMULATION

module binary_dot_product_tb;
    parameter WIDTH = 16;
    parameter ACC_WIDTH = 32;
    
    reg                  clk;
    reg                  rst_n;
    reg  [WIDTH-1:0]     act_in;
    reg  [WIDTH-1:0]     weight_in;
    reg                  valid_in;
    wire                 ready_in;
    wire [ACC_WIDTH-1:0] result;
    wire                 valid_out;
    reg                  ready_out;
    
    binary_dot_product #(
        .WIDTH(WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .act_in(act_in),
        .weight_in(weight_in),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .result(result),
        .valid_out(valid_out),
        .ready_out(ready_out)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Expected result calculation
    function signed [31:0] expected_dot;
        input [WIDTH-1:0] a;
        input [WIDTH-1:0] w;
        integer i;
        reg signed [31:0] sum;
        begin
            sum = 0;
            for (i = 0; i < WIDTH; i = i + 1) begin
                // Convert {0,1} to {-1,+1} and multiply
                sum = sum + (a[i] ? 1 : -1) * (w[i] ? 1 : -1);
            end
            expected_dot = sum;
        end
    endfunction
    
    integer i;
    reg signed [31:0] expected;
    
    initial begin
        $display("Binary Dot Product Testbench");
        $display("============================");
        
        // Initialize
        clk = 0;
        rst_n = 0;
        act_in = 0;
        weight_in = 0;
        valid_in = 0;
        ready_out = 1;
        
        // Reset
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Test 1: All same (all +1 * +1 = +WIDTH)
        act_in = {WIDTH{1'b1}};
        weight_in = {WIDTH{1'b1}};
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for result
        while (!valid_out) @(posedge clk);
        expected = expected_dot(act_in, weight_in);
        if ($signed(result) !== expected)
            $display("FAIL Test1: got %d, expected %d", $signed(result), expected);
        else
            $display("PASS Test1: all ones = %d", $signed(result));
        
        @(posedge clk);
        
        // Test 2: All opposite (all +1 * -1 = -WIDTH)
        act_in = {WIDTH{1'b1}};
        weight_in = {WIDTH{1'b0}};
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        expected = expected_dot(act_in, weight_in);
        if ($signed(result) !== expected)
            $display("FAIL Test2: got %d, expected %d", $signed(result), expected);
        else
            $display("PASS Test2: opposite = %d", $signed(result));
        
        @(posedge clk);
        
        // Test 3: Half match (should be 0)
        act_in = 16'hFF00;
        weight_in = 16'hFF00;
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        while (!valid_out) @(posedge clk);
        expected = expected_dot(act_in, weight_in);
        if ($signed(result) !== expected)
            $display("FAIL Test3: got %d, expected %d", $signed(result), expected);
        else
            $display("PASS Test3: half match = %d", $signed(result));
        
        @(posedge clk);
        
        // Test 4: Random patterns
        for (i = 0; i < 20; i = i + 1) begin
            act_in = $random;
            weight_in = $random;
            valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            
            while (!valid_out) @(posedge clk);
            expected = expected_dot(act_in, weight_in);
            if ($signed(result) !== expected)
                $display("FAIL Random%0d: a=%h, w=%h, got %d, expected %d", 
                         i, act_in, weight_in, $signed(result), expected);
            @(posedge clk);
        end
        
        $display("Testbench complete");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #10000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
