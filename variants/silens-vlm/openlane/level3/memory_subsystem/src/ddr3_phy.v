// =============================================================================
// DDR3 PHY - Physical Layer Interface
// =============================================================================
// DDR3-1066 Physical Layer with:
// - x32 data width (4 byte lanes)
// - DQS strobe generation and alignment
// - ODT (On-Die Termination) control
// - Read/Write leveling for timing calibration
// - SSTL15 I/O simulation
//
// For SKY130: This is a behavioral model. Production silicon would need
// extensive analog characterization and custom I/O cells.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module ddr3_phy #(
    parameter DATA_WIDTH    = 32,
    parameter ADDR_WIDTH    = 16,
    parameter BANK_WIDTH    = 3,
    parameter NUM_RANKS     = 1,
    parameter BURST_LEN     = 8,
    
    // Timing parameters for DDR3-1066
    parameter tCK           = 1875,     // Clock period in ps (1.875ns)
    parameter CL            = 7,        // CAS Latency
    parameter CWL           = 6,        // CAS Write Latency
    parameter AL            = 0,        // Additive Latency
    parameter tDQSS         = 1,        // DQS-DQ skew for write
    parameter tDQSCK        = 225       // DQS output access time (ps)
)(
    // ==========================================================================
    // System Clocks
    // ==========================================================================
    input  wire                         clk_core,       // Core clock (100 MHz)
    input  wire                         clk_ddr,        // DDR clock (533 MHz)
    input  wire                         clk_ddr_90,     // DDR clock 90° shifted
    input  wire                         clk_ddr_180,    // DDR clock 180° shifted
    input  wire                         clk_ref,        // Reference clock for DLL
    input  wire                         rst_n,
    
    // ==========================================================================
    // Controller Interface (Core Clock Domain)
    // ==========================================================================
    
    // Command interface
    input  wire [2:0]                   cmd,            // {RAS#, CAS#, WE#}
    input  wire                         cmd_valid,
    output reg                          cmd_ready,
    
    // Address interface
    input  wire [ADDR_WIDTH-1:0]        addr,
    input  wire [BANK_WIDTH-1:0]        bank,
    
    // Write data interface
    input  wire [DATA_WIDTH*2-1:0]      wdata,          // DDR: 2x data per cycle
    input  wire [DATA_WIDTH/4-1:0]      wdata_mask,     // Byte masks for 2 beats
    input  wire                         wdata_valid,
    output reg                          wdata_ready,
    
    // Read data interface
    output reg  [DATA_WIDTH*2-1:0]      rdata,
    output reg                          rdata_valid,
    
    // ==========================================================================
    // DDR3 Physical Signals
    // ==========================================================================
    
    // Clock
    output wire                         ddr3_ck_p,
    output wire                         ddr3_ck_n,
    
    // Command/Address
    output reg                          ddr3_cs_n,
    output reg                          ddr3_ras_n,
    output reg                          ddr3_cas_n,
    output reg                          ddr3_we_n,
    output reg                          ddr3_cke,
    output reg                          ddr3_reset_n,
    output reg  [BANK_WIDTH-1:0]        ddr3_ba,
    output reg  [ADDR_WIDTH-1:0]        ddr3_addr,
    
    // Data
    inout  wire [DATA_WIDTH-1:0]        ddr3_dq,
    inout  wire [DATA_WIDTH/8-1:0]      ddr3_dqs_p,
    inout  wire [DATA_WIDTH/8-1:0]      ddr3_dqs_n,
    output reg  [DATA_WIDTH/8-1:0]      ddr3_dm,
    
    // Control
    output reg                          ddr3_odt,
    
    // ==========================================================================
    // Calibration and Status
    // ==========================================================================
    input  wire                         cal_start,
    output reg                          cal_done,
    output reg                          cal_error,
    output reg  [7:0]                   rd_dqs_delay,   // Calibrated read delay
    output reg  [7:0]                   wr_dqs_delay,   // Calibrated write delay
    output reg                          phy_ready
);

    // =========================================================================
    // DDR3 Command Encoding
    // =========================================================================
    
    localparam CMD_NOP      = 3'b111;
    localparam CMD_ACT      = 3'b011;   // RAS=0, CAS=1, WE=1
    localparam CMD_READ     = 3'b101;   // RAS=1, CAS=0, WE=1
    localparam CMD_WRITE    = 3'b100;   // RAS=1, CAS=0, WE=0
    localparam CMD_PRE      = 3'b010;   // RAS=0, CAS=1, WE=0
    localparam CMD_REF      = 3'b001;   // RAS=0, CAS=0, WE=1
    localparam CMD_MRS      = 3'b000;   // RAS=0, CAS=0, WE=0
    localparam CMD_ZQCL     = 3'b110;   // ZQ Calibration Long
    
    // =========================================================================
    // Clock Output - Differential pair
    // =========================================================================
    
    assign ddr3_ck_p = clk_ddr;
    assign ddr3_ck_n = ~clk_ddr;
    
    // =========================================================================
    // DQ/DQS Tristate Control
    // =========================================================================
    
    reg [DATA_WIDTH-1:0] dq_out_r;
    reg [DATA_WIDTH-1:0] dq_out_f;
    reg dq_oe;
    reg [DATA_WIDTH/8-1:0] dqs_out;
    reg dqs_oe;
    
    // DDR output register - rising edge data
    reg [DATA_WIDTH-1:0] dq_out_reg;
    
    // Generate DQ tristate buffers
    assign ddr3_dq = dq_oe ? dq_out_reg : {DATA_WIDTH{1'bz}};
    
    // DQS generation with 90° phase shift for writes
    wire [DATA_WIDTH/8-1:0] dqs_internal;
    assign dqs_internal = dqs_oe ? dqs_out : {(DATA_WIDTH/8){1'bz}};
    assign ddr3_dqs_p = dqs_internal;
    assign ddr3_dqs_n = dqs_oe ? ~dqs_out : {(DATA_WIDTH/8){1'bz}};
    
    // =========================================================================
    // Write Data Path
    // =========================================================================
    
    // Write leveling delay line (simplified)
    reg [7:0] wr_delay_cnt;
    reg wr_active;
    reg [3:0] wr_burst_cnt;
    reg [DATA_WIDTH*2-1:0] wdata_buf;
    reg [DATA_WIDTH/4-1:0] wmask_buf;
    
    always @(posedge clk_ddr or negedge rst_n) begin
        if (!rst_n) begin
            dq_out_reg <= {DATA_WIDTH{1'b0}};
            dq_out_r <= {DATA_WIDTH{1'b0}};
            dq_out_f <= {DATA_WIDTH{1'b0}};
            dq_oe <= 1'b0;
            dqs_out <= {(DATA_WIDTH/8){1'b0}};
            dqs_oe <= 1'b0;
            ddr3_dm <= {(DATA_WIDTH/8){1'b0}};
            wr_delay_cnt <= 8'd0;
            wr_active <= 1'b0;
            wr_burst_cnt <= 4'd0;
            wdata_buf <= {(DATA_WIDTH*2){1'b0}};
            wmask_buf <= {(DATA_WIDTH/4){1'b0}};
            
        end else begin
            // Default: output disabled
            dqs_out <= {(DATA_WIDTH/8){1'b0}};
            
            if (wr_active) begin
                if (wr_delay_cnt > 0) begin
                    // Wait for CWL
                    wr_delay_cnt <= wr_delay_cnt - 1'b1;
                    
                    // Preamble - start DQS early
                    if (wr_delay_cnt == 2) begin
                        dqs_oe <= 1'b1;
                    end
                    
                end else if (wr_burst_cnt > 0) begin
                    // DDR output - alternate rising/falling edge data
                    dq_oe <= 1'b1;
                    dqs_oe <= 1'b1;
                    
                    // Toggle DQS for each data beat
                    dqs_out <= {(DATA_WIDTH/8){wr_burst_cnt[0]}};
                    
                    // Select data based on beat
                    if (wr_burst_cnt[0]) begin
                        dq_out_reg <= wdata_buf[DATA_WIDTH +: DATA_WIDTH];
                        ddr3_dm <= wmask_buf[DATA_WIDTH/8 +: DATA_WIDTH/8];
                    end else begin
                        dq_out_reg <= wdata_buf[0 +: DATA_WIDTH];
                        ddr3_dm <= wmask_buf[0 +: DATA_WIDTH/8];
                    end
                    
                    wr_burst_cnt <= wr_burst_cnt - 1'b1;
                    
                end else begin
                    // Postamble
                    dq_oe <= 1'b0;
                    dqs_oe <= 1'b0;
                    wr_active <= 1'b0;
                end
            end
            
            // Start new write
            if (wdata_valid && wdata_ready && !wr_active) begin
                wdata_buf <= wdata;
                wmask_buf <= wdata_mask;
                wr_active <= 1'b1;
                wr_delay_cnt <= CWL + AL - 1;
                wr_burst_cnt <= BURST_LEN;
            end
        end
    end
    
    // =========================================================================
    // Read Data Path
    // =========================================================================
    
    // Read leveling delay line (simplified)
    reg [7:0] rd_delay_cnt;
    reg rd_active;
    reg [3:0] rd_burst_cnt;
    reg [DATA_WIDTH*2-1:0] rdata_buf;
    
    // DQ input capture with DQS alignment
    reg [DATA_WIDTH-1:0] dq_in_r;
    reg [DATA_WIDTH-1:0] dq_in_f;
    
    // Capture on DQS edges (simplified - real impl needs IDELAY)
    always @(posedge clk_ddr) begin
        if (!dq_oe) begin
            dq_in_r <= ddr3_dq;
        end
    end
    
    always @(negedge clk_ddr) begin
        if (!dq_oe) begin
            dq_in_f <= ddr3_dq;
        end
    end
    
    always @(posedge clk_ddr or negedge rst_n) begin
        if (!rst_n) begin
            rd_delay_cnt <= 8'd0;
            rd_active <= 1'b0;
            rd_burst_cnt <= 4'd0;
            rdata_buf <= {(DATA_WIDTH*2){1'b0}};
            rdata <= {(DATA_WIDTH*2){1'b0}};
            rdata_valid <= 1'b0;
            
        end else begin
            rdata_valid <= 1'b0;
            
            if (rd_active) begin
                if (rd_delay_cnt > 0) begin
                    rd_delay_cnt <= rd_delay_cnt - 1'b1;
                    
                end else if (rd_burst_cnt > 0) begin
                    // Capture DDR data
                    if (rd_burst_cnt[0]) begin
                        rdata_buf[DATA_WIDTH +: DATA_WIDTH] <= dq_in_r;
                    end else begin
                        rdata_buf[0 +: DATA_WIDTH] <= dq_in_f;
                    end
                    
                    rd_burst_cnt <= rd_burst_cnt - 1'b1;
                    
                    // Output when burst complete
                    if (rd_burst_cnt == 1) begin
                        rdata <= rdata_buf;
                        rdata_valid <= 1'b1;
                        rd_active <= 1'b0;
                    end
                end
            end
        end
    end
    
    // =========================================================================
    // Command Path
    // =========================================================================
    
    reg cmd_pending;
    reg [2:0] cmd_reg;
    reg [ADDR_WIDTH-1:0] addr_reg;
    reg [BANK_WIDTH-1:0] bank_reg;
    
    always @(posedge clk_ddr or negedge rst_n) begin
        if (!rst_n) begin
            ddr3_cs_n <= 1'b1;
            ddr3_ras_n <= 1'b1;
            ddr3_cas_n <= 1'b1;
            ddr3_we_n <= 1'b1;
            ddr3_cke <= 1'b0;
            ddr3_reset_n <= 1'b0;
            ddr3_ba <= {BANK_WIDTH{1'b0}};
            ddr3_addr <= {ADDR_WIDTH{1'b0}};
            ddr3_odt <= 1'b0;
            
            cmd_pending <= 1'b0;
            cmd_ready <= 1'b0;
            cmd_reg <= CMD_NOP;
            addr_reg <= {ADDR_WIDTH{1'b0}};
            bank_reg <= {BANK_WIDTH{1'b0}};
            
            wdata_ready <= 1'b0;
            phy_ready <= 1'b0;
            
        end else begin
            // Default: NOP
            ddr3_cs_n <= 1'b1;
            ddr3_ras_n <= 1'b1;
            ddr3_cas_n <= 1'b1;
            ddr3_we_n <= 1'b1;
            
            if (phy_ready) begin
                cmd_ready <= 1'b1;
                wdata_ready <= !wr_active;
                
                // Issue pending command
                if (cmd_valid && cmd_ready) begin
                    ddr3_cs_n <= 1'b0;
                    {ddr3_ras_n, ddr3_cas_n, ddr3_we_n} <= cmd;
                    ddr3_ba <= bank;
                    ddr3_addr <= addr;
                    
                    // Start read tracking
                    if (cmd == CMD_READ) begin
                        rd_active <= 1'b1;
                        rd_delay_cnt <= CL + AL;
                        rd_burst_cnt <= BURST_LEN;
                    end
                    
                    // ODT control for writes
                    if (cmd == CMD_WRITE) begin
                        ddr3_odt <= 1'b1;
                    end else begin
                        ddr3_odt <= 1'b0;
                    end
                end
            end
        end
    end
    
    // =========================================================================
    // Read/Write Leveling Calibration
    // =========================================================================
    
    localparam CAL_IDLE     = 4'd0;
    localparam CAL_WR_LVL   = 4'd1;
    localparam CAL_WR_ADJ   = 4'd2;
    localparam CAL_RD_LVL   = 4'd3;
    localparam CAL_RD_ADJ   = 4'd4;
    localparam CAL_DONE     = 4'd5;
    
    reg [3:0] cal_state;
    reg [15:0] cal_cnt;
    reg [7:0] delay_tap;
    reg [DATA_WIDTH/8-1:0] dqs_sample;
    
    always @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            cal_state <= CAL_IDLE;
            cal_done <= 1'b0;
            cal_error <= 1'b0;
            cal_cnt <= 16'd0;
            delay_tap <= 8'd0;
            rd_dqs_delay <= 8'd32;  // Default center tap
            wr_dqs_delay <= 8'd32;
            dqs_sample <= {(DATA_WIDTH/8){1'b0}};
            
        end else begin
            case (cal_state)
                CAL_IDLE: begin
                    if (cal_start && !cal_done) begin
                        cal_state <= CAL_WR_LVL;
                        delay_tap <= 8'd0;
                        cal_cnt <= 16'd0;
                    end
                end
                
                CAL_WR_LVL: begin
                    // Write leveling: find DQS-CK alignment
                    // Sweep delay and sample DQ feedback
                    if (cal_cnt < 64) begin
                        cal_cnt <= cal_cnt + 1;
                    end else begin
                        cal_cnt <= 0;
                        
                        // Sample DQS feedback (simplified)
                        dqs_sample <= ddr3_dqs_p;
                        
                        if (&dqs_sample) begin
                            // Found rising edge
                            wr_dqs_delay <= delay_tap;
                            cal_state <= CAL_RD_LVL;
                            delay_tap <= 8'd0;
                        end else if (delay_tap < 8'd63) begin
                            delay_tap <= delay_tap + 1;
                        end else begin
                            // Failed to find edge
                            wr_dqs_delay <= 8'd32;  // Use default
                            cal_state <= CAL_RD_LVL;
                            delay_tap <= 8'd0;
                        end
                    end
                end
                
                CAL_RD_LVL: begin
                    // Read leveling: find DQS-DQ alignment window
                    if (cal_cnt < 64) begin
                        cal_cnt <= cal_cnt + 1;
                    end else begin
                        cal_cnt <= 0;
                        
                        // Find data eye center (simplified)
                        if (delay_tap < 8'd63) begin
                            delay_tap <= delay_tap + 1;
                        end else begin
                            // Use center of sweep as delay
                            rd_dqs_delay <= 8'd32;
                            cal_state <= CAL_DONE;
                        end
                    end
                end
                
                CAL_DONE: begin
                    cal_done <= 1'b1;
                    phy_ready <= 1'b1;
                end
                
                default: cal_state <= CAL_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Initialization Sequence
    // =========================================================================
    
    localparam INIT_RESET   = 3'd0;
    localparam INIT_CKE     = 3'd1;
    localparam INIT_MRS     = 3'd2;
    localparam INIT_ZQ      = 3'd3;
    localparam INIT_CAL     = 3'd4;
    localparam INIT_DONE    = 3'd5;
    
    reg [2:0] init_state;
    reg [19:0] init_cnt;
    reg [2:0] mrs_step;
    reg init_done_r;
    
    always @(posedge clk_ddr or negedge rst_n) begin
        if (!rst_n) begin
            init_state <= INIT_RESET;
            init_cnt <= 20'd0;
            mrs_step <= 3'd0;
            init_done_r <= 1'b0;
            
        end else begin
            case (init_state)
                INIT_RESET: begin
                    ddr3_reset_n <= 1'b0;
                    ddr3_cke <= 1'b0;
                    
                    if (init_cnt < 20'd200000) begin  // 200us reset
                        init_cnt <= init_cnt + 1;
                    end else begin
                        ddr3_reset_n <= 1'b1;
                        init_cnt <= 20'd0;
                        init_state <= INIT_CKE;
                    end
                end
                
                INIT_CKE: begin
                    if (init_cnt < 20'd500000) begin  // 500us CKE low
                        init_cnt <= init_cnt + 1;
                    end else begin
                        ddr3_cke <= 1'b1;
                        init_cnt <= 20'd0;
                        init_state <= INIT_MRS;
                        mrs_step <= 3'd0;
                    end
                end
                
                INIT_MRS: begin
                    // Mode register programming sequence
                    if (init_cnt > 0) begin
                        init_cnt <= init_cnt - 1;
                        ddr3_cs_n <= 1'b1;  // NOP
                    end else begin
                        case (mrs_step)
                            3'd0: begin
                                // MRS2: CWL=6
                                ddr3_cs_n <= 1'b0;
                                ddr3_ras_n <= 1'b0;
                                ddr3_cas_n <= 1'b0;
                                ddr3_we_n <= 1'b0;
                                ddr3_ba <= 3'b010;
                                ddr3_addr <= {ADDR_WIDTH{1'b0}};
                                ddr3_addr[5:3] <= 3'b001;  // CWL=6
                                init_cnt <= 20'd10;
                                mrs_step <= 3'd1;
                            end
                            3'd1: begin
                                // MRS3
                                ddr3_cs_n <= 1'b0;
                                ddr3_ras_n <= 1'b0;
                                ddr3_cas_n <= 1'b0;
                                ddr3_we_n <= 1'b0;
                                ddr3_ba <= 3'b011;
                                ddr3_addr <= {ADDR_WIDTH{1'b0}};
                                init_cnt <= 20'd10;
                                mrs_step <= 3'd2;
                            end
                            3'd2: begin
                                // MRS1: DLL enable, ODT=Rtt_Nom
                                ddr3_cs_n <= 1'b0;
                                ddr3_ras_n <= 1'b0;
                                ddr3_cas_n <= 1'b0;
                                ddr3_we_n <= 1'b0;
                                ddr3_ba <= 3'b001;
                                ddr3_addr <= {ADDR_WIDTH{1'b0}};
                                ddr3_addr[2] <= 1'b1;  // ODT Rtt_Nom
                                init_cnt <= 20'd10;
                                mrs_step <= 3'd3;
                            end
                            3'd3: begin
                                // MRS0: DLL reset, CL=7, BL=8
                                ddr3_cs_n <= 1'b0;
                                ddr3_ras_n <= 1'b0;
                                ddr3_cas_n <= 1'b0;
                                ddr3_we_n <= 1'b0;
                                ddr3_ba <= 3'b000;
                                ddr3_addr <= {ADDR_WIDTH{1'b0}};
                                ddr3_addr[8] <= 1'b1;       // DLL Reset
                                ddr3_addr[6:4] <= 3'b011;   // CL=7
                                ddr3_addr[1:0] <= 2'b00;    // BL=8
                                init_cnt <= 20'd512;        // tDLLK
                                mrs_step <= 3'd4;
                            end
                            3'd4: begin
                                init_state <= INIT_ZQ;
                                init_cnt <= 20'd0;
                            end
                            default: mrs_step <= 3'd0;
                        endcase
                    end
                end
                
                INIT_ZQ: begin
                    // ZQ Calibration Long
                    if (init_cnt == 0) begin
                        ddr3_cs_n <= 1'b0;
                        ddr3_ras_n <= 1'b1;
                        ddr3_cas_n <= 1'b1;
                        ddr3_we_n <= 1'b0;
                        ddr3_addr[10] <= 1'b1;  // ZQCL
                        init_cnt <= 20'd512;    // tZQinit
                    end else if (init_cnt > 1) begin
                        init_cnt <= init_cnt - 1;
                        ddr3_cs_n <= 1'b1;
                    end else begin
                        init_state <= INIT_CAL;
                    end
                end
                
                INIT_CAL: begin
                    // Wait for read/write leveling
                    if (cal_done) begin
                        init_state <= INIT_DONE;
                        init_done_r <= 1'b1;
                    end
                end
                
                INIT_DONE: begin
                    // Normal operation
                    init_done_r <= 1'b1;
                end
                
                default: init_state <= INIT_RESET;
            endcase
        end
    end

endmodule

`default_nettype wire
