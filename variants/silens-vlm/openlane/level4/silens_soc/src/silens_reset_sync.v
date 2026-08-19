// =============================================================================
// SiLens Reset Synchronizer
// =============================================================================
// 
// Synchronizes async reset to clock domain with proper metastability protection.
// Uses 2-stage synchronizer with async assert, sync deassert.
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_reset_sync (
    input  wire     clk,
    input  wire     rst_async_n,
    output wire     rst_sync_n
);

    // 2-stage synchronizer for metastability protection
    reg [1:0] sync_ff;
    
    always @(posedge clk or negedge rst_async_n) begin
        if (!rst_async_n) begin
            // Async assert (immediate)
            sync_ff <= 2'b00;
        end else begin
            // Sync deassert (through flip-flops)
            sync_ff <= {sync_ff[0], 1'b1};
        end
    end
    
    assign rst_sync_n = sync_ff[1];

endmodule

`default_nettype wire
