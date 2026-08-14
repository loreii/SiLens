// =============================================================================
// SiLens Reset Synchronizer
// =============================================================================
// Synchronizes asynchronous reset to clock domain.
// Implements proper reset synchronization to avoid metastability.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_reset_sync #(
    parameter STAGES = 3  // Number of synchronization stages
)(
    input  wire clk,
    input  wire rst_async_n,    // Asynchronous reset (active low)
    output wire rst_sync_n      // Synchronized reset (active low)
);

    // Synchronizer chain
    (* ASYNC_REG = "TRUE" *)
    reg [STAGES-1:0] sync_chain;
    
    always @(posedge clk or negedge rst_async_n) begin
        if (!rst_async_n) begin
            // Asynchronous assert
            sync_chain <= {STAGES{1'b0}};
        end else begin
            // Synchronous deassert
            sync_chain <= {sync_chain[STAGES-2:0], 1'b1};
        end
    end
    
    assign rst_sync_n = sync_chain[STAGES-1];

endmodule

`default_nettype wire
