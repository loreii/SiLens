// =============================================================================
// SiLens - Ternary Multiply-Accumulate Module
// =============================================================================
// Computes multiply-accumulate for ternary weights (-1, 0, +1).
// Encoding: 2 bits per weight
//   00 = 0  (zero)
//   01 = +1 (add)
//   10 = -1 (subtract)
//   11 = reserved
//
// Operation: For each element i:
//   if weight[i] == +1: acc += activation[i]
//   if weight[i] == -1: acc -= activation[i]
//   if weight[i] == 0:  acc unchanged
//
// License: Apache 2.0
// =============================================================================

module ternary_mac #(
    parameter NUM_ELEMENTS = 256,                   // Number of elements in vector
    parameter ACT_WIDTH    = 8,                     // Activation bit width
    parameter ACC_WIDTH    = 32,                    // Accumulator bit width
    parameter PARALLEL     = 16                     // Elements processed per cycle
)(
    input  wire                                 clk,
    input  wire                                 rst_n,
    
    // Input interface
    input  wire [NUM_ELEMENTS*ACT_WIDTH-1:0]    act_in,         // Packed activations
    input  wire [NUM_ELEMENTS*2-1:0]            weight_in,      // Packed ternary weights
    input  wire                                 valid_in,
    output wire                                 ready_in,
    
    // Accumulator control
    input  wire                                 acc_clear,      // Clear accumulator
    
    // Output interface
    output reg  signed [ACC_WIDTH-1:0]          result,         // Accumulated result
    output reg                                  valid_out,
    input  wire                                 ready_out
);

    // =========================================================================
    // Weight encoding constants
    // =========================================================================
    
    localparam W_ZERO  = 2'b00;
    localparam W_POS   = 2'b01;
    localparam W_NEG   = 2'b10;
    
    // =========================================================================
    // Control FSM
    // =========================================================================
    
    localparam STATE_IDLE    = 2'b00;
    localparam STATE_COMPUTE = 2'b01;
    localparam STATE_DONE    = 2'b10;
    
    reg [1:0] state;
    reg [$clog2(NUM_ELEMENTS/PARALLEL+1)-1:0] elem_idx;
    
    localparam NUM_ITERS = (NUM_ELEMENTS + PARALLEL - 1) / PARALLEL;
    
    // =========================================================================
    // Accumulator
    // =========================================================================
    
    reg signed [ACC_WIDTH-1:0] accumulator;
    
    // =========================================================================
    // Parallel MAC computation
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] partial_sum;
    
    // Compute partial sum for PARALLEL elements
    reg signed [ACC_WIDTH-1:0] mac_results [0:PARALLEL-1];
    integer k;
    
    always @(*) begin
        for (k = 0; k < PARALLEL; k = k + 1) begin
            // Calculate actual element index
            if (elem_idx * PARALLEL + k < NUM_ELEMENTS) begin
                // Extract activation and weight for this element
                // Note: These are just wires indexed by elem_idx in the always block
                mac_results[k] = 0;  // Default
            end else begin
                mac_results[k] = 0;
            end
        end
    end
    
    // Compute partial sum combinationally
    wire signed [ACT_WIDTH:0] signed_acts [0:PARALLEL-1];
    wire [1:0] weights [0:PARALLEL-1];
    wire signed [ACC_WIDTH-1:0] products [0:PARALLEL-1];
    
    genvar g;
    generate
        for (g = 0; g < PARALLEL; g = g + 1) begin : gen_mac
            // Calculate which element we're processing
            wire [$clog2(NUM_ELEMENTS)-1:0] cur_elem = elem_idx * PARALLEL + g;
            
            // Bounds check
            wire in_bounds = (cur_elem < NUM_ELEMENTS);
            
            // Extract activation (unsigned to signed extension)
            wire [ACT_WIDTH-1:0] act_unsigned = act_in[cur_elem*ACT_WIDTH +: ACT_WIDTH];
            assign signed_acts[g] = $signed({1'b0, act_unsigned});
            
            // Extract weight
            assign weights[g] = weight_in[cur_elem*2 +: 2];
            
            // Compute product based on weight encoding
            assign products[g] = (state != STATE_COMPUTE || !in_bounds) ? 0 :
                                 (weights[g] == W_POS) ?  signed_acts[g] :
                                 (weights[g] == W_NEG) ? -signed_acts[g] :
                                 0;  // W_ZERO or reserved
        end
    endgenerate
    
    // Sum all products (tree reduction for better timing)
    wire signed [ACC_WIDTH-1:0] sum_level0 [0:(PARALLEL+1)/2-1];
    wire signed [ACC_WIDTH-1:0] sum_level1 [0:(PARALLEL+3)/4-1];
    wire signed [ACC_WIDTH-1:0] sum_level2 [0:(PARALLEL+7)/8-1];
    wire signed [ACC_WIDTH-1:0] sum_level3;
    
    generate
        // Level 0: pairs
        for (g = 0; g < PARALLEL/2; g = g + 1) begin : sum_l0
            assign sum_level0[g] = products[2*g] + products[2*g+1];
        end
        if (PARALLEL % 2 == 1) begin : sum_l0_odd
            assign sum_level0[PARALLEL/2] = products[PARALLEL-1];
        end
        
        // Level 1: pairs of pairs
        localparam L0_SIZE = (PARALLEL+1)/2;
        for (g = 0; g < L0_SIZE/2; g = g + 1) begin : sum_l1
            assign sum_level1[g] = sum_level0[2*g] + sum_level0[2*g+1];
        end
        if (L0_SIZE % 2 == 1) begin : sum_l1_odd
            assign sum_level1[L0_SIZE/2] = sum_level0[L0_SIZE-1];
        end
        
        // Level 2: groups of 4
        localparam L1_SIZE = (L0_SIZE+1)/2;
        for (g = 0; g < L1_SIZE/2; g = g + 1) begin : sum_l2
            assign sum_level2[g] = sum_level1[2*g] + sum_level1[2*g+1];
        end
        if (L1_SIZE % 2 == 1) begin : sum_l2_odd
            assign sum_level2[L1_SIZE/2] = sum_level1[L1_SIZE-1];
        end
        
        // Level 3: final sum
        localparam L2_SIZE = (L1_SIZE+1)/2;
        if (L2_SIZE == 1) begin : sum_final_1
            assign sum_level3 = sum_level2[0];
        end else if (L2_SIZE == 2) begin : sum_final_2
            assign sum_level3 = sum_level2[0] + sum_level2[1];
        end else begin : sum_final_n
            // For very large PARALLEL values
            reg signed [ACC_WIDTH-1:0] final_sum;
            integer fs;
            always @(*) begin
                final_sum = 0;
                for (fs = 0; fs < L2_SIZE; fs = fs + 1)
                    final_sum = final_sum + sum_level2[fs];
            end
            assign sum_level3 = final_sum;
        end
    endgenerate
    
    assign partial_sum = sum_level3;
    
    // =========================================================================
    // Control FSM
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= STATE_IDLE;
            elem_idx    <= 0;
            accumulator <= 0;
            valid_out   <= 1'b0;
            result      <= 0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    if (acc_clear) begin
                        accumulator <= 0;
                    end
                    if (valid_in) begin
                        state    <= STATE_COMPUTE;
                        elem_idx <= 0;
                        if (acc_clear) begin
                            accumulator <= 0;
                        end
                    end
                end
                
                STATE_COMPUTE: begin
                    // Accumulate partial sum
                    accumulator <= accumulator + partial_sum;
                    
                    if (elem_idx >= NUM_ITERS - 1) begin
                        state <= STATE_DONE;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end
                
                STATE_DONE: begin
                    result    <= accumulator + partial_sum;  // Include last iteration
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

module ternary_mac_tb;
    parameter NUM_ELEMENTS = 16;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter PARALLEL = 4;
    
    reg                                  clk;
    reg                                  rst_n;
    reg  [NUM_ELEMENTS*ACT_WIDTH-1:0]    act_in;
    reg  [NUM_ELEMENTS*2-1:0]            weight_in;
    reg                                  valid_in;
    wire                                 ready_in;
    reg                                  acc_clear;
    wire signed [ACC_WIDTH-1:0]          result;
    wire                                 valid_out;
    reg                                  ready_out;
    
    ternary_mac #(
        .NUM_ELEMENTS(NUM_ELEMENTS),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .act_in(act_in),
        .weight_in(weight_in),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .acc_clear(acc_clear),
        .result(result),
        .valid_out(valid_out),
        .ready_out(ready_out)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Weight encoding
    localparam W_ZERO = 2'b00;
    localparam W_POS  = 2'b01;
    localparam W_NEG  = 2'b10;
    
    // Expected result calculation
    function signed [31:0] expected_mac;
        input [NUM_ELEMENTS*ACT_WIDTH-1:0] a;
        input [NUM_ELEMENTS*2-1:0] w;
        integer i;
        reg signed [31:0] sum;
        reg [ACT_WIDTH-1:0] act_val;
        reg [1:0] w_val;
        begin
            sum = 0;
            for (i = 0; i < NUM_ELEMENTS; i = i + 1) begin
                act_val = a[i*ACT_WIDTH +: ACT_WIDTH];
                w_val = w[i*2 +: 2];
                case (w_val)
                    W_POS: sum = sum + act_val;
                    W_NEG: sum = sum - act_val;
                    default: ; // W_ZERO: no change
                endcase
            end
            expected_mac = sum;
        end
    endfunction
    
    // Pack weight array helper
    task set_weights;
        input [1:0] w0, w1, w2, w3, w4, w5, w6, w7;
        input [1:0] w8, w9, w10, w11, w12, w13, w14, w15;
        begin
            weight_in = {w15, w14, w13, w12, w11, w10, w9, w8,
                         w7, w6, w5, w4, w3, w2, w1, w0};
        end
    endtask
    
    integer i;
    reg signed [31:0] expected;
    
    initial begin
        $display("Ternary MAC Testbench");
        $display("=====================");
        
        // Initialize
        clk = 0;
        rst_n = 0;
        act_in = 0;
        weight_in = 0;
        valid_in = 0;
        acc_clear = 1;
        ready_out = 1;
        
        // Reset
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Test 1: All +1 weights
        for (i = 0; i < NUM_ELEMENTS; i = i + 1)
            act_in[i*ACT_WIDTH +: ACT_WIDTH] = i + 1;  // 1,2,3,...,16
        weight_in = {NUM_ELEMENTS{W_POS}};
        acc_clear = 1;
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        acc_clear = 0;
        
        while (!valid_out) @(posedge clk);
        expected = expected_mac(act_in, weight_in);
        if (result !== expected)
            $display("FAIL Test1: got %d, expected %d", result, expected);
        else
            $display("PASS Test1: all +1 weights = %d", result);
        
        @(posedge clk);
        
        // Test 2: All -1 weights
        weight_in = {NUM_ELEMENTS{W_NEG}};
        acc_clear = 1;
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        acc_clear = 0;
        
        while (!valid_out) @(posedge clk);
        expected = expected_mac(act_in, weight_in);
        if (result !== expected)
            $display("FAIL Test2: got %d, expected %d", result, expected);
        else
            $display("PASS Test2: all -1 weights = %d", result);
        
        @(posedge clk);
        
        // Test 3: All zero weights
        weight_in = {NUM_ELEMENTS{W_ZERO}};
        acc_clear = 1;
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        acc_clear = 0;
        
        while (!valid_out) @(posedge clk);
        expected = 0;
        if (result !== expected)
            $display("FAIL Test3: got %d, expected %d", result, expected);
        else
            $display("PASS Test3: all zero weights = %d", result);
        
        @(posedge clk);
        
        // Test 4: Mixed weights
        set_weights(W_POS, W_NEG, W_ZERO, W_POS, W_NEG, W_ZERO, W_POS, W_NEG,
                    W_ZERO, W_POS, W_NEG, W_ZERO, W_POS, W_NEG, W_ZERO, W_POS);
        acc_clear = 1;
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        acc_clear = 0;
        
        while (!valid_out) @(posedge clk);
        expected = expected_mac(act_in, weight_in);
        if (result !== expected)
            $display("FAIL Test4: got %d, expected %d", result, expected);
        else
            $display("PASS Test4: mixed weights = %d", result);
        
        @(posedge clk);
        
        // Test 5: Random patterns
        for (i = 0; i < 10; i = i + 1) begin
            act_in = {$random, $random, $random, $random};
            weight_in = $random;
            acc_clear = 1;
            valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            acc_clear = 0;
            
            while (!valid_out) @(posedge clk);
            expected = expected_mac(act_in, weight_in);
            if (result !== expected)
                $display("FAIL Random%0d: got %d, expected %d", i, result, expected);
            @(posedge clk);
        end
        
        $display("Testbench complete");
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
