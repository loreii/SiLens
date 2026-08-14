// =============================================================================
// SiLens Clock Generator
// =============================================================================
// PLL-based clock generation for the SoC.
// Generates core clock, DDR clock, and phase-shifted DDR clock.
//
// For SKY130: Uses ring oscillator + dividers as fallback
// For FPGA: Uses vendor-specific PLL
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_clock_gen #(
    parameter REF_CLK_MHZ  = 100,
    parameter CORE_CLK_MHZ = 100,
    parameter DDR_CLK_MHZ  = 533
)(
    input  wire clk_ref,
    input  wire rst_n,
    output wire clk_core,
    output wire clk_ddr,
    output wire clk_ddr_90,
    output wire locked
);

`ifdef SYNTHESIS_SKY130
    // =========================================================================
    // SKY130 Implementation: Simple clock distribution
    // =========================================================================
    // For SKY130, we use external oscillator for core clock
    // DDR clock would need dedicated PLL (not available in open PDK)
    // Fallback: Use core clock with timing constraints relaxed
    
    // Core clock = reference clock (for 100MHz designs)
    assign clk_core = clk_ref;
    
    // DDR clock = reference clock (will limit DDR speed)
    // In production, use external DDR clock or hard macro PLL
    assign clk_ddr = clk_ref;
    
    // 90° phase shift via delay line (approximate)
    // In production, use DLL or PLL for precise phase
    wire clk_ddr_delayed;
    
    // Chain of inverters for ~2.5ns delay (90° at 100MHz)
    (* keep = "true" *)
    wire [15:0] delay_chain;
    assign delay_chain[0] = clk_ref;
    
    genvar i;
    generate
        for (i = 1; i < 16; i = i + 1) begin : delay_gen
            (* keep = "true" *)
            sky130_fd_sc_hd__inv_2 u_inv (
                .A(delay_chain[i-1]),
                .Y(delay_chain[i])
            );
        end
    endgenerate
    
    assign clk_ddr_90 = delay_chain[8];  // Tap at middle for ~90°
    
    // Lock signal: after reset stabilization
    reg [7:0] lock_cnt;
    reg locked_r;
    
    always @(posedge clk_ref or negedge rst_n) begin
        if (!rst_n) begin
            lock_cnt <= 0;
            locked_r <= 0;
        end else if (!locked_r) begin
            if (lock_cnt == 8'hFF)
                locked_r <= 1;
            else
                lock_cnt <= lock_cnt + 1;
        end
    end
    
    assign locked = locked_r;

`else
    // =========================================================================
    // Generic/FPGA Implementation: Behavioral PLL model
    // =========================================================================
    
    // Core clock = reference clock (1:1 ratio)
    assign clk_core = clk_ref;
    
    // DDR clock generation (for simulation/FPGA)
    // In real FPGA, use MMCM/PLL primitive
    
    `ifdef SIMULATION
        // Simulation: Generate DDR clock at specified frequency
        reg clk_ddr_r = 0;
        real ddr_period = 1000.0 / DDR_CLK_MHZ;  // Period in ns
        
        initial begin
            forever begin
                #(ddr_period/2) clk_ddr_r = ~clk_ddr_r;
            end
        end
        
        assign clk_ddr = clk_ddr_r;
        
        // 90° phase shifted version
        reg clk_ddr_90_r = 0;
        initial begin
            #(ddr_period/4);  // 90° delay
            forever begin
                #(ddr_period/2) clk_ddr_90_r = ~clk_ddr_90_r;
            end
        end
        
        assign clk_ddr_90 = clk_ddr_90_r;
        
        // Lock after 100ns
        reg locked_r = 0;
        initial begin
            #100 locked_r = 1;
        end
        assign locked = locked_r;
        
    `else
        // FPGA: Use reference clock for both (constrain appropriately)
        assign clk_ddr = clk_ref;
        assign clk_ddr_90 = clk_ref;  // Would use MMCM phase shift
        
        // Simple lock counter
        reg [7:0] lock_cnt;
        reg locked_r;
        
        always @(posedge clk_ref or negedge rst_n) begin
            if (!rst_n) begin
                lock_cnt <= 0;
                locked_r <= 0;
            end else if (!locked_r) begin
                if (lock_cnt == 8'hFF)
                    locked_r <= 1;
                else
                    lock_cnt <= lock_cnt + 1;
            end
        end
        
        assign locked = locked_r;
    `endif

`endif

endmodule

`default_nettype wire
