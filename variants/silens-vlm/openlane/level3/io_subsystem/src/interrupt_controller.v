// =============================================================================
// SiLens Interrupt Controller
// =============================================================================
// 16-source interrupt controller with level/edge triggering, priority encoder,
// and interrupt masking. Part of IO Subsystem (Level 3).
//
// Features:
//   - 16 interrupt sources
//   - Level or edge triggered (configurable per source)
//   - Priority encoder with programmable priorities
//   - Interrupt status and mask registers
//   - Interrupt acknowledge mechanism
//
// Register Map (directly accessible):
//   0x00: IRQ_STATUS    (RO)   - Current interrupt status (pending)
//   0x04: IRQ_RAW       (RO)   - Raw interrupt inputs (before edge detect)
//   0x08: IRQ_ENABLE    (RW)   - Interrupt enable mask
//   0x0C: IRQ_EDGE_CFG  (RW)   - Edge trigger config (1=edge, 0=level)
//   0x10: IRQ_POLARITY  (RW)   - Polarity config (1=rising/high, 0=falling/low)
//   0x14: IRQ_CLEAR     (WO)   - Clear pending interrupts (write-1-to-clear)
//   0x18: IRQ_PRIORITY  (RW)   - Priority register (4-bit per source, 16 sources)
//   0x1C: IRQ_HIGHEST   (RO)   - Highest priority pending interrupt number
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module interrupt_controller #(
    parameter NUM_SOURCES = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Interrupt sources (directly active-high, active-low inverted by polarity config)
    input  wire [NUM_SOURCES-1:0]   irq_sources,
    
    // Combined interrupt output to host
    output wire                     irq_out,
    
    // Register interface
    input  wire [4:0]               reg_addr,
    input  wire [31:0]              reg_wdata,
    output reg  [31:0]              reg_rdata,
    input  wire                     reg_wr,
    input  wire                     reg_rd
);

    // =========================================================================
    // Register Addresses
    // =========================================================================
    
    localparam REG_IRQ_STATUS   = 5'h00;
    localparam REG_IRQ_RAW      = 5'h04;
    localparam REG_IRQ_ENABLE   = 5'h08;
    localparam REG_IRQ_EDGE_CFG = 5'h0C;
    localparam REG_IRQ_POLARITY = 5'h10;
    localparam REG_IRQ_CLEAR    = 5'h14;
    localparam REG_IRQ_PRIORITY = 5'h18;
    localparam REG_IRQ_HIGHEST  = 5'h1C;
    
    // =========================================================================
    // Configuration Registers
    // =========================================================================
    
    reg [NUM_SOURCES-1:0] irq_enable;       // Enable mask
    reg [NUM_SOURCES-1:0] irq_edge_cfg;     // 1 = edge triggered, 0 = level
    reg [NUM_SOURCES-1:0] irq_polarity;     // 1 = rising/active-high, 0 = falling/active-low
    reg [63:0]            irq_priority_reg; // 4 bits per source
    
    // =========================================================================
    // Status Registers
    // =========================================================================
    
    reg [NUM_SOURCES-1:0] irq_pending;      // Pending interrupts
    reg [NUM_SOURCES-1:0] irq_sources_d;    // Delayed for edge detection
    
    // =========================================================================
    // Edge Detection and Trigger Logic
    // =========================================================================
    
    wire [NUM_SOURCES-1:0] irq_adjusted;    // Polarity-adjusted sources
    wire [NUM_SOURCES-1:0] irq_edge;        // Edge detected
    wire [NUM_SOURCES-1:0] irq_triggered;   // Final trigger (edge or level)
    
    // Apply polarity inversion
    assign irq_adjusted = irq_sources ^ ~irq_polarity;
    
    // Edge detection (rising edge of adjusted signal)
    assign irq_edge = irq_adjusted & ~irq_sources_d;
    
    // Select edge or level based on configuration
    genvar i;
    generate
        for (i = 0; i < NUM_SOURCES; i = i + 1) begin : gen_trigger
            assign irq_triggered[i] = irq_edge_cfg[i] ? irq_edge[i] : irq_adjusted[i];
        end
    endgenerate
    
    // Delay for edge detection
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            irq_sources_d <= {NUM_SOURCES{1'b0}};
        end else begin
            irq_sources_d <= irq_adjusted;
        end
    end
    
    // =========================================================================
    // Pending Interrupt Logic
    // =========================================================================
    
    wire [NUM_SOURCES-1:0] irq_clear;
    
    // Pending register: set by trigger, cleared by software
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            irq_pending <= {NUM_SOURCES{1'b0}};
        end else begin
            // Set on trigger (for edge-triggered) or maintain for level
            irq_pending <= (irq_pending | (irq_triggered & irq_edge_cfg)) & ~irq_clear;
            // For level-triggered, pending = current level & enable
        end
    end
    
    // Effective pending (combines latched edge + current level triggers)
    wire [NUM_SOURCES-1:0] irq_effective_pending;
    assign irq_effective_pending = (irq_pending | (irq_triggered & ~irq_edge_cfg)) & irq_enable;
    
    // =========================================================================
    // Priority Encoder
    // =========================================================================
    
    // Find highest priority pending interrupt
    reg [3:0]  highest_irq;
    reg        irq_valid;
    
    // Simple priority encoder - lower number = higher priority
    // In production, use programmable priority from irq_priority_reg
    integer j;
    always @(*) begin
        highest_irq = 4'd0;
        irq_valid = 1'b0;
        
        for (j = NUM_SOURCES - 1; j >= 0; j = j - 1) begin
            if (irq_effective_pending[j]) begin
                highest_irq = j[3:0];
                irq_valid = 1'b1;
            end
        end
    end
    
    // =========================================================================
    // Interrupt Output
    // =========================================================================
    
    assign irq_out = irq_valid;
    
    // =========================================================================
    // Register Interface
    // =========================================================================
    
    // Clear register (directly used, not stored)
    assign irq_clear = (reg_wr && reg_addr == REG_IRQ_CLEAR) ? reg_wdata[NUM_SOURCES-1:0] : {NUM_SOURCES{1'b0}};
    
    // Register writes
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            irq_enable       <= {NUM_SOURCES{1'b0}};
            irq_edge_cfg     <= {NUM_SOURCES{1'b0}};  // Default: level triggered
            irq_polarity     <= {NUM_SOURCES{1'b1}};  // Default: active high
            irq_priority_reg <= 64'h0;
        end else if (reg_wr) begin
            case (reg_addr)
                REG_IRQ_ENABLE:   irq_enable       <= reg_wdata[NUM_SOURCES-1:0];
                REG_IRQ_EDGE_CFG: irq_edge_cfg     <= reg_wdata[NUM_SOURCES-1:0];
                REG_IRQ_POLARITY: irq_polarity     <= reg_wdata[NUM_SOURCES-1:0];
                REG_IRQ_PRIORITY: irq_priority_reg <= {reg_wdata, irq_priority_reg[31:0]};
                // Note: For full 64-bit priority, need two writes or wider bus
            endcase
        end
    end
    
    // Register reads
    always @(*) begin
        reg_rdata = 32'h0;
        
        if (reg_rd) begin
            case (reg_addr)
                REG_IRQ_STATUS:   reg_rdata = {{(32-NUM_SOURCES){1'b0}}, irq_effective_pending};
                REG_IRQ_RAW:      reg_rdata = {{(32-NUM_SOURCES){1'b0}}, irq_sources};
                REG_IRQ_ENABLE:   reg_rdata = {{(32-NUM_SOURCES){1'b0}}, irq_enable};
                REG_IRQ_EDGE_CFG: reg_rdata = {{(32-NUM_SOURCES){1'b0}}, irq_edge_cfg};
                REG_IRQ_POLARITY: reg_rdata = {{(32-NUM_SOURCES){1'b0}}, irq_polarity};
                REG_IRQ_PRIORITY: reg_rdata = irq_priority_reg[31:0];
                REG_IRQ_HIGHEST:  reg_rdata = {27'b0, irq_valid, highest_irq};
                default:          reg_rdata = 32'hDEAD_BEEF;
            endcase
        end
    end

endmodule

`default_nettype wire
