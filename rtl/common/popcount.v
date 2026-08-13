// =============================================================================
// SiLens - Population Count Module
// =============================================================================
// Counts the number of 1 bits in the input vector.
// Used for binary/ternary dot product computation.
//
// License: Apache 2.0
// =============================================================================

module popcount #(
    parameter WIDTH = 512,
    parameter OUT_WIDTH = $clog2(WIDTH) + 1
)(
    input  wire [WIDTH-1:0]     in,
    output wire [OUT_WIDTH-1:0] count
);

    // Implementation options:
    // 1. Tree adder (balanced, good for synthesis)
    // 2. LUT-based (fast for small widths)
    // 3. Sequential (area-efficient, slower)

    // For now: hierarchical tree adder approach
    
    generate
        if (WIDTH == 1) begin : base_case
            assign count = in;
        end
        else if (WIDTH == 2) begin : two_bits
            assign count = in[0] + in[1];
        end
        else begin : recursive
            localparam HALF = WIDTH / 2;
            localparam HALF_OUT = $clog2(HALF) + 1;
            
            wire [HALF_OUT-1:0] count_lo, count_hi;
            
            popcount #(.WIDTH(HALF)) pc_lo (
                .in(in[HALF-1:0]),
                .count(count_lo)
            );
            
            popcount #(.WIDTH(WIDTH-HALF)) pc_hi (
                .in(in[WIDTH-1:HALF]),
                .count(count_hi)
            );
            
            assign count = count_lo + count_hi;
        end
    endgenerate

endmodule

// =============================================================================
// Testbench (for simulation)
// =============================================================================
`ifdef SIMULATION

module popcount_tb;
    parameter WIDTH = 16;
    parameter OUT_WIDTH = $clog2(WIDTH) + 1;
    
    reg  [WIDTH-1:0]     in;
    wire [OUT_WIDTH-1:0] count;
    
    popcount #(.WIDTH(WIDTH)) dut (
        .in(in),
        .count(count)
    );
    
    integer i, expected;
    
    initial begin
        $display("Popcount Testbench");
        $display("==================");
        
        // Test all zeros
        in = 0;
        #10;
        if (count !== 0) $display("FAIL: all zeros");
        else $display("PASS: all zeros");
        
        // Test all ones
        in = {WIDTH{1'b1}};
        #10;
        if (count !== WIDTH) $display("FAIL: all ones");
        else $display("PASS: all ones");
        
        // Test random patterns
        for (i = 0; i < 100; i = i + 1) begin
            in = $random;
            #10;
            expected = $countones(in);
            if (count !== expected) 
                $display("FAIL: in=%h, expected=%d, got=%d", in, expected, count);
        end
        
        $display("Testbench complete");
        $finish;
    end
endmodule

`endif
