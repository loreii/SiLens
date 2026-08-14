// =============================================================================
// Ternary MAC Array (64-wide SIMD) - Level 1 Synthesis Block
// =============================================================================
// 64-parallel ternary multiply-accumulate unit.
// This is the core compute primitive for SiLens.
//
// Operation: acc += sum(w[i] * x[i]) for i in 0..63
// Where w[i] is ternary: -1, 0, +1 (encoded as 2 bits)
//
// Target: ~1mm² on SKY130
// Latency: 1 cycle (combinational MAC, registered output)
// Throughput: 64 MACs per cycle
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module ternary_mac_array_64 #(
    parameter LANES     = 64,       // SIMD width
    parameter ACT_WIDTH = 8,        // Activation bits
    parameter ACC_WIDTH = 32        // Accumulator bits
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Control
    input  wire                         valid_in,
    input  wire                         acc_clear,      // Clear accumulator
    input  wire                         acc_enable,     // Enable accumulation
    
    // Data inputs
    input  wire [LANES*ACT_WIDTH-1:0]   x_in,           // Activations
    input  wire [LANES*2-1:0]           w_in,           // Ternary weights
    
    // Accumulator output
    output reg  [ACC_WIDTH-1:0]         acc_out,
    output reg                          valid_out
);

    // =========================================================================
    // Weight encoding
    // =========================================================================
    // 00 = 0
    // 01 = +1
    // 10 = -1
    // 11 = reserved (treat as 0)
    
    // =========================================================================
    // Stage 1: Parallel ternary multiplies
    // =========================================================================
    // For ternary weights, multiply is just conditional negate/zero:
    // w=+1: result = x
    // w=-1: result = -x
    // w=0:  result = 0
    
    wire signed [ACT_WIDTH:0] products [0:LANES-1];
    
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : mult_gen
            wire [1:0] w_i = w_in[i*2 +: 2];
            wire signed [ACT_WIDTH-1:0] x_i = $signed(x_in[i*ACT_WIDTH +: ACT_WIDTH]);
            
            // Ternary multiply (no actual multiplier needed!)
            assign products[i] = (w_i == 2'b01) ?  {x_i[ACT_WIDTH-1], x_i} :  // +1: sign extend
                                 (w_i == 2'b10) ? -{x_i[ACT_WIDTH-1], x_i} :  // -1: negate
                                                   {(ACT_WIDTH+1){1'b0}};     // 0: zero
        end
    endgenerate
    
    // =========================================================================
    // Stage 2: Reduction tree (sum of 64 products)
    // =========================================================================
    // Use 6-level binary tree: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    // Each level adds 1 bit to prevent overflow
    
    // Level 1: 64 -> 32 (9-bit + 9-bit = 10-bit)
    wire signed [ACT_WIDTH+1:0] sum_l1 [0:31];
    generate
        for (i = 0; i < 32; i = i + 1) begin : l1_gen
            assign sum_l1[i] = $signed(products[i*2]) + $signed(products[i*2+1]);
        end
    endgenerate
    
    // Level 2: 32 -> 16 (10-bit + 10-bit = 11-bit)
    wire signed [ACT_WIDTH+2:0] sum_l2 [0:15];
    generate
        for (i = 0; i < 16; i = i + 1) begin : l2_gen
            assign sum_l2[i] = $signed(sum_l1[i*2]) + $signed(sum_l1[i*2+1]);
        end
    endgenerate
    
    // Level 3: 16 -> 8 (11-bit + 11-bit = 12-bit)
    wire signed [ACT_WIDTH+3:0] sum_l3 [0:7];
    generate
        for (i = 0; i < 8; i = i + 1) begin : l3_gen
            assign sum_l3[i] = $signed(sum_l2[i*2]) + $signed(sum_l2[i*2+1]);
        end
    endgenerate
    
    // Level 4: 8 -> 4 (12-bit + 12-bit = 13-bit)
    wire signed [ACT_WIDTH+4:0] sum_l4 [0:3];
    generate
        for (i = 0; i < 4; i = i + 1) begin : l4_gen
            assign sum_l4[i] = $signed(sum_l3[i*2]) + $signed(sum_l3[i*2+1]);
        end
    endgenerate
    
    // Level 5: 4 -> 2 (13-bit + 13-bit = 14-bit)
    wire signed [ACT_WIDTH+5:0] sum_l5 [0:1];
    assign sum_l5[0] = $signed(sum_l4[0]) + $signed(sum_l4[1]);
    assign sum_l5[1] = $signed(sum_l4[2]) + $signed(sum_l4[3]);
    
    // Level 6: 2 -> 1 (14-bit + 14-bit = 15-bit)
    wire signed [ACT_WIDTH+6:0] sum_total;
    assign sum_total = $signed(sum_l5[0]) + $signed(sum_l5[1]);
    
    // =========================================================================
    // Stage 3: Accumulator (registered)
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] acc_next;
    wire signed [ACC_WIDTH-1:0] sum_extended;
    
    // Sign-extend sum to accumulator width
    assign sum_extended = {{(ACC_WIDTH-ACT_WIDTH-7){sum_total[ACT_WIDTH+6]}}, sum_total};
    
    // Accumulate or load
    assign acc_next = acc_clear ? sum_extended : 
                      acc_enable ? ($signed(acc_out) + sum_extended) :
                      acc_out;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin
            if (valid_in) begin
                acc_out <= acc_next;
            end
            valid_out <= valid_in;
        end
    end

endmodule

`default_nettype wire
