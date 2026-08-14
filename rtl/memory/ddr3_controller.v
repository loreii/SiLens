// =============================================================================
// SiLens DDR3 Memory Controller
// =============================================================================
// DDR3-1066 SDRAM controller for external KV cache memory.
// Supports 32-bit data width (4.3 GB/s bandwidth).
//
// For SKY130: This is a simplified controller suitable for DDR3-800/1066.
// Production version would need extensive timing characterization.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_ddr3_controller #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 28,      // 256MB address space
    parameter BURST_LEN  = 8,       // BL8 for DDR3
    
    // Timing parameters (for DDR3-1066, tCK = 1.875ns)
    parameter tRCD       = 8,       // RAS to CAS delay (cycles)
    parameter tRP        = 8,       // Precharge period
    parameter tRAS       = 20,      // Active to precharge
    parameter tRC        = 28,      // Row cycle time
    parameter tRFC       = 86,      // Refresh to active
    parameter tREFI      = 4160,    // Refresh interval (7.8us @ 533MHz)
    parameter CL         = 7,       // CAS latency
    parameter CWL        = 6        // CAS write latency
)(
    // System clocks
    input  wire                     clk_core,
    input  wire                     clk_ddr,
    input  wire                     clk_ddr_90,
    input  wire                     rst_n,
    
    // User interface (core clock domain)
    input  wire [ADDR_WIDTH-1:0]    user_addr,
    input  wire [DATA_WIDTH-1:0]    user_wdata,
    output reg  [DATA_WIDTH-1:0]    user_rdata,
    input  wire                     user_rd,
    input  wire                     user_wr,
    output reg                      user_ready,
    
    // DDR3 PHY interface
    output reg  [13:0]              ddr3_addr,
    output reg  [2:0]               ddr3_ba,
    output reg                      ddr3_cas_n,
    output wire                     ddr3_ck_p,
    output wire                     ddr3_ck_n,
    output reg                      ddr3_cke,
    output reg                      ddr3_cs_n,
    output reg  [DATA_WIDTH/8-1:0]  ddr3_dm,
    inout  wire [DATA_WIDTH-1:0]    ddr3_dq,
    inout  wire [DATA_WIDTH/8-1:0]  ddr3_dqs_p,
    inout  wire [DATA_WIDTH/8-1:0]  ddr3_dqs_n,
    output reg                      ddr3_odt,
    output reg                      ddr3_ras_n,
    output reg                      ddr3_reset_n,
    output reg                      ddr3_we_n,
    
    // Status
    output reg                      init_done,
    output reg                      cal_complete
);

    // =========================================================================
    // DDR3 Clock Output
    // =========================================================================
    
    assign ddr3_ck_p = clk_ddr;
    assign ddr3_ck_n = ~clk_ddr;
    
    // =========================================================================
    // State Machine
    // =========================================================================
    
    localparam ST_RESET     = 4'd0;
    localparam ST_CKE_LOW   = 4'd1;
    localparam ST_MRS       = 4'd2;
    localparam ST_ZQCL      = 4'd3;
    localparam ST_IDLE      = 4'd4;
    localparam ST_ACTIVATE  = 4'd5;
    localparam ST_READ      = 4'd6;
    localparam ST_WRITE     = 4'd7;
    localparam ST_PRECHARGE = 4'd8;
    localparam ST_REFRESH   = 4'd9;
    localparam ST_READ_DATA = 4'd10;
    
    reg [3:0] state;
    reg [15:0] wait_cnt;
    reg [15:0] refresh_cnt;
    reg refresh_pending;
    
    // =========================================================================
    // Address Decoding
    // =========================================================================
    
    // Address mapping: {row[13:0], bank[2:0], col[9:0], burst_addr[2:0]}
    wire [13:0] row_addr   = user_addr[27:14];
    wire [2:0]  bank_addr  = user_addr[13:11];
    wire [9:0]  col_addr   = user_addr[10:1];
    
    reg [13:0] active_row [0:7];  // Track active row per bank
    reg [7:0]  bank_active;
    
    // =========================================================================
    // DQ/DQS Tristate Control
    // =========================================================================
    
    reg [DATA_WIDTH-1:0] dq_out;
    reg dq_oe;
    reg [DATA_WIDTH/8-1:0] dqs_out;
    reg dqs_oe;
    
    assign ddr3_dq = dq_oe ? dq_out : {DATA_WIDTH{1'bz}};
    assign ddr3_dqs_p = dqs_oe ? dqs_out : {(DATA_WIDTH/8){1'bz}};
    assign ddr3_dqs_n = dqs_oe ? ~dqs_out : {(DATA_WIDTH/8){1'bz}};
    
    // =========================================================================
    // Initialization Sequence
    // =========================================================================
    
    reg [3:0] init_step;
    
    always @(posedge clk_ddr or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_RESET;
            wait_cnt <= 0;
            init_done <= 0;
            cal_complete <= 0;
            init_step <= 0;
            
            // Default pin states
            ddr3_reset_n <= 0;
            ddr3_cke <= 0;
            ddr3_cs_n <= 1;
            ddr3_ras_n <= 1;
            ddr3_cas_n <= 1;
            ddr3_we_n <= 1;
            ddr3_odt <= 0;
            ddr3_addr <= 0;
            ddr3_ba <= 0;
            ddr3_dm <= {(DATA_WIDTH/8){1'b0}};
            
            dq_oe <= 0;
            dqs_oe <= 0;
            
            refresh_cnt <= 0;
            refresh_pending <= 0;
            bank_active <= 0;
            
            user_ready <= 0;
            user_rdata <= 0;
            
        end else begin
            // Default: deassert command
            ddr3_cs_n <= 1;
            ddr3_ras_n <= 1;
            ddr3_cas_n <= 1;
            ddr3_we_n <= 1;
            user_ready <= 0;
            
            // Refresh counter
            if (init_done) begin
                if (refresh_cnt >= tREFI) begin
                    refresh_pending <= 1;
                    refresh_cnt <= 0;
                end else begin
                    refresh_cnt <= refresh_cnt + 1;
                end
            end
            
            case (state)
                // =============================================================
                // Initialization
                // =============================================================
                ST_RESET: begin
                    if (wait_cnt < 200) begin  // 200 cycles reset
                        wait_cnt <= wait_cnt + 1;
                    end else begin
                        ddr3_reset_n <= 1;
                        wait_cnt <= 0;
                        state <= ST_CKE_LOW;
                    end
                end
                
                ST_CKE_LOW: begin
                    if (wait_cnt < 500) begin  // 500us CKE low
                        wait_cnt <= wait_cnt + 1;
                    end else begin
                        ddr3_cke <= 1;
                        wait_cnt <= 0;
                        state <= ST_MRS;
                        init_step <= 0;
                    end
                end
                
                ST_MRS: begin
                    // Mode Register Set sequence
                    if (wait_cnt > 0) begin
                        wait_cnt <= wait_cnt - 1;
                    end else begin
                        case (init_step)
                            0: begin
                                // MRS2: CWL=6
                                ddr3_cs_n <= 0;
                                ddr3_ras_n <= 0;
                                ddr3_cas_n <= 0;
                                ddr3_we_n <= 0;
                                ddr3_ba <= 3'b010;
                                ddr3_addr <= 14'b00_0_000_0_001_000;  // CWL=6
                                wait_cnt <= tRP;
                                init_step <= 1;
                            end
                            1: begin
                                // MRS3
                                ddr3_cs_n <= 0;
                                ddr3_ras_n <= 0;
                                ddr3_cas_n <= 0;
                                ddr3_we_n <= 0;
                                ddr3_ba <= 3'b011;
                                ddr3_addr <= 14'b0;
                                wait_cnt <= tRP;
                                init_step <= 2;
                            end
                            2: begin
                                // MRS1: DLL enable, ODT
                                ddr3_cs_n <= 0;
                                ddr3_ras_n <= 0;
                                ddr3_cas_n <= 0;
                                ddr3_we_n <= 0;
                                ddr3_ba <= 3'b001;
                                ddr3_addr <= 14'b00_0_0_0_0_0_1_0_0_0_0_0_0;
                                wait_cnt <= tRP;
                                init_step <= 3;
                            end
                            3: begin
                                // MRS0: DLL reset, CL=7, BL=8
                                ddr3_cs_n <= 0;
                                ddr3_ras_n <= 0;
                                ddr3_cas_n <= 0;
                                ddr3_we_n <= 0;
                                ddr3_ba <= 3'b000;
                                ddr3_addr <= 14'b0_0_1_0_0_011_1_0_00;  // CL=7, BL=8
                                wait_cnt <= tRP;
                                init_step <= 4;
                            end
                            4: begin
                                state <= ST_ZQCL;
                                wait_cnt <= 0;
                            end
                        endcase
                    end
                end
                
                ST_ZQCL: begin
                    // ZQ calibration
                    if (wait_cnt == 0) begin
                        ddr3_cs_n <= 0;
                        ddr3_ras_n <= 1;
                        ddr3_cas_n <= 1;
                        ddr3_we_n <= 0;
                        ddr3_addr[10] <= 1;  // ZQCL long
                        wait_cnt <= 512;  // tZQinit
                    end else if (wait_cnt > 1) begin
                        wait_cnt <= wait_cnt - 1;
                    end else begin
                        init_done <= 1;
                        cal_complete <= 1;
                        state <= ST_IDLE;
                    end
                end
                
                // =============================================================
                // Normal Operation
                // =============================================================
                ST_IDLE: begin
                    user_ready <= 1;
                    
                    // Priority: Refresh > User request
                    if (refresh_pending) begin
                        // Precharge all banks first
                        if (|bank_active) begin
                            ddr3_cs_n <= 0;
                            ddr3_ras_n <= 0;
                            ddr3_cas_n <= 1;
                            ddr3_we_n <= 0;
                            ddr3_addr[10] <= 1;  // All banks
                            bank_active <= 0;
                            wait_cnt <= tRP;
                            state <= ST_PRECHARGE;
                        end else begin
                            state <= ST_REFRESH;
                        end
                    end else if (user_rd || user_wr) begin
                        user_ready <= 0;
                        // Check if row is already active
                        if (bank_active[bank_addr] && active_row[bank_addr] == row_addr) begin
                            // Row hit - go directly to read/write
                            state <= user_rd ? ST_READ : ST_WRITE;
                        end else if (bank_active[bank_addr]) begin
                            // Row miss - precharge then activate
                            ddr3_cs_n <= 0;
                            ddr3_ras_n <= 0;
                            ddr3_cas_n <= 1;
                            ddr3_we_n <= 0;
                            ddr3_ba <= bank_addr;
                            ddr3_addr[10] <= 0;  // Single bank
                            bank_active[bank_addr] <= 0;
                            wait_cnt <= tRP;
                            state <= ST_PRECHARGE;
                        end else begin
                            // Bank idle - activate
                            state <= ST_ACTIVATE;
                        end
                    end
                end
                
                ST_PRECHARGE: begin
                    if (wait_cnt > 1) begin
                        wait_cnt <= wait_cnt - 1;
                    end else begin
                        if (refresh_pending)
                            state <= ST_REFRESH;
                        else
                            state <= ST_ACTIVATE;
                    end
                end
                
                ST_ACTIVATE: begin
                    // Issue ACTIVATE command
                    ddr3_cs_n <= 0;
                    ddr3_ras_n <= 0;
                    ddr3_cas_n <= 1;
                    ddr3_we_n <= 1;
                    ddr3_ba <= bank_addr;
                    ddr3_addr <= row_addr;
                    
                    bank_active[bank_addr] <= 1;
                    active_row[bank_addr] <= row_addr;
                    
                    wait_cnt <= tRCD;
                    state <= user_rd ? ST_READ : ST_WRITE;
                end
                
                ST_READ: begin
                    if (wait_cnt > 1) begin
                        wait_cnt <= wait_cnt - 1;
                    end else begin
                        // Issue READ command
                        ddr3_cs_n <= 0;
                        ddr3_ras_n <= 1;
                        ddr3_cas_n <= 0;
                        ddr3_we_n <= 1;
                        ddr3_ba <= bank_addr;
                        ddr3_addr <= {4'b0, col_addr};
                        ddr3_addr[10] <= 0;  // No auto-precharge
                        
                        wait_cnt <= CL + BURST_LEN/2;
                        state <= ST_READ_DATA;
                    end
                end
                
                ST_READ_DATA: begin
                    if (wait_cnt > 1) begin
                        wait_cnt <= wait_cnt - 1;
                    end else begin
                        // Capture read data (simplified - real impl needs DQS alignment)
                        user_rdata <= ddr3_dq;
                        user_ready <= 1;
                        state <= ST_IDLE;
                    end
                end
                
                ST_WRITE: begin
                    if (wait_cnt > 1) begin
                        wait_cnt <= wait_cnt - 1;
                    end else begin
                        // Issue WRITE command
                        ddr3_cs_n <= 0;
                        ddr3_ras_n <= 1;
                        ddr3_cas_n <= 0;
                        ddr3_we_n <= 0;
                        ddr3_ba <= bank_addr;
                        ddr3_addr <= {4'b0, col_addr};
                        ddr3_addr[10] <= 0;
                        
                        // Drive data
                        dq_oe <= 1;
                        dqs_oe <= 1;
                        dq_out <= user_wdata;
                        dqs_out <= {(DATA_WIDTH/8){1'b1}};
                        
                        wait_cnt <= CWL + BURST_LEN/2 + 2;
                        state <= ST_IDLE;
                    end
                end
                
                ST_REFRESH: begin
                    // Issue REFRESH command
                    ddr3_cs_n <= 0;
                    ddr3_ras_n <= 0;
                    ddr3_cas_n <= 0;
                    ddr3_we_n <= 1;
                    
                    refresh_pending <= 0;
                    wait_cnt <= tRFC;
                    state <= ST_IDLE;
                end
                
                default: state <= ST_IDLE;
            endcase
            
            // Clear DQ/DQS output enable after write
            if (state != ST_WRITE) begin
                dq_oe <= 0;
                dqs_oe <= 0;
            end
        end
    end

endmodule

`default_nettype wire
