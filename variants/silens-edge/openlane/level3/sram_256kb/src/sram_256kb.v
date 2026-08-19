// =============================================================================
// SiLens Edge - 256KB Dual-Port SRAM Macro
// =============================================================================
// Behavioral model for simulation. In production, replace with OpenRAM-generated
// SRAM or SKY130 SRAM macros.
//
// Configuration:
//   - Depth: 32K words
//   - Width: 64 bits
//   - Total: 32K × 64 = 256KB
//   - Dual-port: Port A (R/W), Port B (R only)
//
// Target area: ~10mm² on SKY130
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module sram_256kb #(
    parameter ADDR_WIDTH = 15,          // 2^15 = 32K words
    parameter DATA_WIDTH = 64           // 64-bit data
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // =========================================================================
    // Port A: Vision Encoder (Read/Write)
    // =========================================================================
    input  wire [ADDR_WIDTH-1:0]    a_addr,
    input  wire [DATA_WIDTH-1:0]    a_wdata,
    output reg  [DATA_WIDTH-1:0]    a_rdata,
    input  wire                     a_we,       // Write enable
    input  wire                     a_re,       // Read enable
    output wire                     a_ready,
    
    // =========================================================================
    // Port B: Classifier (Read Only)
    // =========================================================================
    input  wire [ADDR_WIDTH-1:0]    b_addr,
    input  wire [DATA_WIDTH-1:0]    b_wdata,    // Unused, for interface compatibility
    output reg  [DATA_WIDTH-1:0]    b_rdata,
    input  wire                     b_we,       // Unused, tied to 0 externally
    input  wire                     b_re,       // Read enable
    output wire                     b_ready
);

    // =========================================================================
    // Memory Array
    // =========================================================================
    // In synthesis, this would be replaced with SRAM macro instantiation
    // For simulation/behavioral: inferred memory
    
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
    
    // =========================================================================
    // Port A: Read/Write Access
    // =========================================================================
    
    always @(posedge clk) begin
        if (a_we) begin
            mem[a_addr] <= a_wdata;
        end
        if (a_re) begin
            a_rdata <= mem[a_addr];
        end
    end
    
    // =========================================================================
    // Port B: Read-Only Access
    // =========================================================================
    
    always @(posedge clk) begin
        if (b_re) begin
            b_rdata <= mem[b_addr];
        end
    end
    
    // =========================================================================
    // Ready Signals (always ready in this simple model)
    // =========================================================================
    // In a real SRAM macro, there might be wait states for certain operations
    
    assign a_ready = 1'b1;
    assign b_ready = 1'b1;

    // =========================================================================
    // Optional: Initialize memory to zero on reset (for simulation)
    // =========================================================================
    `ifdef SIMULATION
    integer i;
    initial begin
        for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
            mem[i] = {DATA_WIDTH{1'b0}};
        end
    end
    `endif

endmodule

`default_nettype wire
