// =============================================================================
// AXI4 Arbiter - 4-Port Memory Access Arbiter
// =============================================================================
// Round-robin arbiter with priority support for 4 AXI4 master ports:
// - Port 0: LLM Read (highest priority for inference latency)
// - Port 1: LLM Write (KV cache updates)
// - Port 2: Vision Read (frame processing)
// - Port 3: Host DMA (lowest priority, background transfers)
//
// Features:
// - Round-robin with priority override
// - Burst transaction support (up to 256 beats)
// - Read/Write channel independence
// - Command reordering for bandwidth optimization
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module axi_arbiter #(
    parameter NUM_PORTS     = 4,
    parameter DATA_WIDTH    = 64,
    parameter ADDR_WIDTH    = 32,
    parameter ID_WIDTH      = 4,
    parameter LEN_WIDTH     = 8
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // ==========================================================================
    // AXI4 Slave Ports (from masters: LLM, Vision, Host)
    // ==========================================================================
    
    // Write Address Channel
    input  wire [NUM_PORTS*ID_WIDTH-1:0]    s_axi_awid,
    input  wire [NUM_PORTS*ADDR_WIDTH-1:0]  s_axi_awaddr,
    input  wire [NUM_PORTS*LEN_WIDTH-1:0]   s_axi_awlen,
    input  wire [NUM_PORTS*3-1:0]           s_axi_awsize,
    input  wire [NUM_PORTS*2-1:0]           s_axi_awburst,
    input  wire [NUM_PORTS-1:0]             s_axi_awvalid,
    output reg  [NUM_PORTS-1:0]             s_axi_awready,
    
    // Write Data Channel
    input  wire [NUM_PORTS*DATA_WIDTH-1:0]  s_axi_wdata,
    input  wire [NUM_PORTS*DATA_WIDTH/8-1:0] s_axi_wstrb,
    input  wire [NUM_PORTS-1:0]             s_axi_wlast,
    input  wire [NUM_PORTS-1:0]             s_axi_wvalid,
    output reg  [NUM_PORTS-1:0]             s_axi_wready,
    
    // Write Response Channel
    output reg  [NUM_PORTS*ID_WIDTH-1:0]    s_axi_bid,
    output reg  [NUM_PORTS*2-1:0]           s_axi_bresp,
    output reg  [NUM_PORTS-1:0]             s_axi_bvalid,
    input  wire [NUM_PORTS-1:0]             s_axi_bready,
    
    // Read Address Channel
    input  wire [NUM_PORTS*ID_WIDTH-1:0]    s_axi_arid,
    input  wire [NUM_PORTS*ADDR_WIDTH-1:0]  s_axi_araddr,
    input  wire [NUM_PORTS*LEN_WIDTH-1:0]   s_axi_arlen,
    input  wire [NUM_PORTS*3-1:0]           s_axi_arsize,
    input  wire [NUM_PORTS*2-1:0]           s_axi_arburst,
    input  wire [NUM_PORTS-1:0]             s_axi_arvalid,
    output reg  [NUM_PORTS-1:0]             s_axi_arready,
    
    // Read Data Channel
    output reg  [NUM_PORTS*ID_WIDTH-1:0]    s_axi_rid,
    output reg  [NUM_PORTS*DATA_WIDTH-1:0]  s_axi_rdata,
    output reg  [NUM_PORTS*2-1:0]           s_axi_rresp,
    output reg  [NUM_PORTS-1:0]             s_axi_rlast,
    output reg  [NUM_PORTS-1:0]             s_axi_rvalid,
    input  wire [NUM_PORTS-1:0]             s_axi_rready,
    
    // ==========================================================================
    // AXI4 Master Port (to DDR3 Controller)
    // ==========================================================================
    
    // Write Address Channel
    output reg  [ID_WIDTH-1:0]              m_axi_awid,
    output reg  [ADDR_WIDTH-1:0]            m_axi_awaddr,
    output reg  [LEN_WIDTH-1:0]             m_axi_awlen,
    output reg  [2:0]                       m_axi_awsize,
    output reg  [1:0]                       m_axi_awburst,
    output reg                              m_axi_awvalid,
    input  wire                             m_axi_awready,
    
    // Write Data Channel
    output reg  [DATA_WIDTH-1:0]            m_axi_wdata,
    output reg  [DATA_WIDTH/8-1:0]          m_axi_wstrb,
    output reg                              m_axi_wlast,
    output reg                              m_axi_wvalid,
    input  wire                             m_axi_wready,
    
    // Write Response Channel
    input  wire [ID_WIDTH-1:0]              m_axi_bid,
    input  wire [1:0]                       m_axi_bresp,
    input  wire                             m_axi_bvalid,
    output reg                              m_axi_bready,
    
    // Read Address Channel
    output reg  [ID_WIDTH-1:0]              m_axi_arid,
    output reg  [ADDR_WIDTH-1:0]            m_axi_araddr,
    output reg  [LEN_WIDTH-1:0]             m_axi_arlen,
    output reg  [2:0]                       m_axi_arsize,
    output reg  [1:0]                       m_axi_arburst,
    output reg                              m_axi_arvalid,
    input  wire                             m_axi_arready,
    
    // Read Data Channel
    input  wire [ID_WIDTH-1:0]              m_axi_rid,
    input  wire [DATA_WIDTH-1:0]            m_axi_rdata,
    input  wire [1:0]                       m_axi_rresp,
    input  wire                             m_axi_rlast,
    input  wire                             m_axi_rvalid,
    output reg                              m_axi_rready,
    
    // ==========================================================================
    // Status and Debug
    // ==========================================================================
    output wire [NUM_PORTS-1:0]             port_active,
    output wire [1:0]                       current_read_port,
    output wire [1:0]                       current_write_port
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    
    localparam PORT_LLM_RD  = 2'd0;
    localparam PORT_LLM_WR  = 2'd1;
    localparam PORT_VIS_RD  = 2'd2;
    localparam PORT_HOST    = 2'd3;
    
    // AXI Response codes
    localparam RESP_OKAY    = 2'b00;
    localparam RESP_SLVERR  = 2'b10;
    
    // =========================================================================
    // Arbiter State Machines
    // =========================================================================
    
    // Read channel arbitration
    reg [1:0] rd_grant;
    reg [1:0] rd_grant_next;
    reg       rd_busy;
    reg [LEN_WIDTH-1:0] rd_beat_cnt;
    
    // Write channel arbitration
    reg [1:0] wr_grant;
    reg [1:0] wr_grant_next;
    reg       wr_addr_busy;
    reg       wr_data_busy;
    reg [LEN_WIDTH-1:0] wr_beat_cnt;
    
    // Pending transaction tracking
    reg [NUM_PORTS-1:0] rd_pending;
    reg [NUM_PORTS-1:0] wr_pending;
    
    // Transaction info storage
    reg [ID_WIDTH-1:0]   rd_id_store   [0:NUM_PORTS-1];
    reg [ADDR_WIDTH-1:0] rd_addr_store [0:NUM_PORTS-1];
    reg [LEN_WIDTH-1:0]  rd_len_store  [0:NUM_PORTS-1];
    reg [2:0]            rd_size_store [0:NUM_PORTS-1];
    reg [1:0]            rd_burst_store[0:NUM_PORTS-1];
    
    reg [ID_WIDTH-1:0]   wr_id_store   [0:NUM_PORTS-1];
    reg [ADDR_WIDTH-1:0] wr_addr_store [0:NUM_PORTS-1];
    reg [LEN_WIDTH-1:0]  wr_len_store  [0:NUM_PORTS-1];
    reg [2:0]            wr_size_store [0:NUM_PORTS-1];
    reg [1:0]            wr_burst_store[0:NUM_PORTS-1];
    
    // =========================================================================
    // Priority Encoder with Round-Robin
    // =========================================================================
    
    // Read arbitration - priority based round-robin
    always @(*) begin
        rd_grant_next = rd_grant;
        
        if (!rd_busy) begin
            // Check ports in priority order starting from last granted + 1
            case (rd_grant)
                2'd0: begin
                    if (s_axi_arvalid[1])      rd_grant_next = 2'd1;
                    else if (s_axi_arvalid[2]) rd_grant_next = 2'd2;
                    else if (s_axi_arvalid[3]) rd_grant_next = 2'd3;
                    else if (s_axi_arvalid[0]) rd_grant_next = 2'd0;
                end
                2'd1: begin
                    if (s_axi_arvalid[2])      rd_grant_next = 2'd2;
                    else if (s_axi_arvalid[3]) rd_grant_next = 2'd3;
                    else if (s_axi_arvalid[0]) rd_grant_next = 2'd0;
                    else if (s_axi_arvalid[1]) rd_grant_next = 2'd1;
                end
                2'd2: begin
                    if (s_axi_arvalid[3])      rd_grant_next = 2'd3;
                    else if (s_axi_arvalid[0]) rd_grant_next = 2'd0;
                    else if (s_axi_arvalid[1]) rd_grant_next = 2'd1;
                    else if (s_axi_arvalid[2]) rd_grant_next = 2'd2;
                end
                2'd3: begin
                    if (s_axi_arvalid[0])      rd_grant_next = 2'd0;
                    else if (s_axi_arvalid[1]) rd_grant_next = 2'd1;
                    else if (s_axi_arvalid[2]) rd_grant_next = 2'd2;
                    else if (s_axi_arvalid[3]) rd_grant_next = 2'd3;
                end
            endcase
            
            // Priority override: LLM read has highest priority
            if (s_axi_arvalid[PORT_LLM_RD] && rd_grant != PORT_LLM_RD) begin
                rd_grant_next = PORT_LLM_RD;
            end
        end
    end
    
    // Write arbitration - similar priority based round-robin
    always @(*) begin
        wr_grant_next = wr_grant;
        
        if (!wr_addr_busy) begin
            case (wr_grant)
                2'd0: begin
                    if (s_axi_awvalid[1])      wr_grant_next = 2'd1;
                    else if (s_axi_awvalid[2]) wr_grant_next = 2'd2;
                    else if (s_axi_awvalid[3]) wr_grant_next = 2'd3;
                    else if (s_axi_awvalid[0]) wr_grant_next = 2'd0;
                end
                2'd1: begin
                    if (s_axi_awvalid[2])      wr_grant_next = 2'd2;
                    else if (s_axi_awvalid[3]) wr_grant_next = 2'd3;
                    else if (s_axi_awvalid[0]) wr_grant_next = 2'd0;
                    else if (s_axi_awvalid[1]) wr_grant_next = 2'd1;
                end
                2'd2: begin
                    if (s_axi_awvalid[3])      wr_grant_next = 2'd3;
                    else if (s_axi_awvalid[0]) wr_grant_next = 2'd0;
                    else if (s_axi_awvalid[1]) wr_grant_next = 2'd1;
                    else if (s_axi_awvalid[2]) wr_grant_next = 2'd2;
                end
                2'd3: begin
                    if (s_axi_awvalid[0])      wr_grant_next = 2'd0;
                    else if (s_axi_awvalid[1]) wr_grant_next = 2'd1;
                    else if (s_axi_awvalid[2]) wr_grant_next = 2'd2;
                    else if (s_axi_awvalid[3]) wr_grant_next = 2'd3;
                end
            endcase
            
            // Priority override: LLM write has priority
            if (s_axi_awvalid[PORT_LLM_WR] && wr_grant != PORT_LLM_WR) begin
                wr_grant_next = PORT_LLM_WR;
            end
        end
    end
    
    // =========================================================================
    // Read Channel Logic
    // =========================================================================
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_grant <= 2'd0;
            rd_busy <= 1'b0;
            rd_beat_cnt <= {LEN_WIDTH{1'b0}};
            rd_pending <= {NUM_PORTS{1'b0}};
            
            m_axi_arid <= {ID_WIDTH{1'b0}};
            m_axi_araddr <= {ADDR_WIDTH{1'b0}};
            m_axi_arlen <= {LEN_WIDTH{1'b0}};
            m_axi_arsize <= 3'b0;
            m_axi_arburst <= 2'b01;
            m_axi_arvalid <= 1'b0;
            m_axi_rready <= 1'b0;
            
            s_axi_arready <= {NUM_PORTS{1'b0}};
            
            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                s_axi_rid[i*ID_WIDTH +: ID_WIDTH] <= {ID_WIDTH{1'b0}};
                s_axi_rdata[i*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                s_axi_rresp[i*2 +: 2] <= RESP_OKAY;
                s_axi_rlast[i] <= 1'b0;
                s_axi_rvalid[i] <= 1'b0;
            end
            
            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                rd_id_store[i] <= {ID_WIDTH{1'b0}};
                rd_addr_store[i] <= {ADDR_WIDTH{1'b0}};
                rd_len_store[i] <= {LEN_WIDTH{1'b0}};
                rd_size_store[i] <= 3'b0;
                rd_burst_store[i] <= 2'b0;
            end
            
        end else begin
            // Default ready signals
            s_axi_arready <= {NUM_PORTS{1'b0}};
            
            // Clear read valid when accepted
            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                if (s_axi_rvalid[i] && s_axi_rready[i]) begin
                    s_axi_rvalid[i] <= 1'b0;
                end
            end
            
            // State machine
            if (!rd_busy) begin
                // Accept new read request
                rd_grant <= rd_grant_next;
                
                if (s_axi_arvalid[rd_grant_next]) begin
                    // Store transaction info
                    rd_id_store[rd_grant_next] <= s_axi_arid[rd_grant_next*ID_WIDTH +: ID_WIDTH];
                    rd_addr_store[rd_grant_next] <= s_axi_araddr[rd_grant_next*ADDR_WIDTH +: ADDR_WIDTH];
                    rd_len_store[rd_grant_next] <= s_axi_arlen[rd_grant_next*LEN_WIDTH +: LEN_WIDTH];
                    rd_size_store[rd_grant_next] <= s_axi_arsize[rd_grant_next*3 +: 3];
                    rd_burst_store[rd_grant_next] <= s_axi_arburst[rd_grant_next*2 +: 2];
                    
                    // Forward to master port
                    m_axi_arid <= s_axi_arid[rd_grant_next*ID_WIDTH +: ID_WIDTH];
                    m_axi_araddr <= s_axi_araddr[rd_grant_next*ADDR_WIDTH +: ADDR_WIDTH];
                    m_axi_arlen <= s_axi_arlen[rd_grant_next*LEN_WIDTH +: LEN_WIDTH];
                    m_axi_arsize <= s_axi_arsize[rd_grant_next*3 +: 3];
                    m_axi_arburst <= s_axi_arburst[rd_grant_next*2 +: 2];
                    m_axi_arvalid <= 1'b1;
                    
                    s_axi_arready[rd_grant_next] <= 1'b1;
                    rd_busy <= 1'b1;
                    rd_beat_cnt <= s_axi_arlen[rd_grant_next*LEN_WIDTH +: LEN_WIDTH];
                    rd_pending[rd_grant_next] <= 1'b1;
                end
                
            end else begin
                // Address phase complete
                if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b1;
                end
                
                // Data phase - forward read data to correct slave port
                if (m_axi_rvalid && m_axi_rready) begin
                    s_axi_rid[rd_grant*ID_WIDTH +: ID_WIDTH] <= m_axi_rid;
                    s_axi_rdata[rd_grant*DATA_WIDTH +: DATA_WIDTH] <= m_axi_rdata;
                    s_axi_rresp[rd_grant*2 +: 2] <= m_axi_rresp;
                    s_axi_rvalid[rd_grant] <= 1'b1;
                    
                    if (rd_beat_cnt == 0) begin
                        s_axi_rlast[rd_grant] <= 1'b1;
                    end else begin
                        s_axi_rlast[rd_grant] <= 1'b0;
                    end
                    
                    if (m_axi_rlast) begin
                        rd_busy <= 1'b0;
                        m_axi_rready <= 1'b0;
                        rd_pending[rd_grant] <= 1'b0;
                    end else begin
                        rd_beat_cnt <= rd_beat_cnt - 1'b1;
                    end
                end
            end
        end
    end
    
    // =========================================================================
    // Write Channel Logic
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_grant <= 2'd0;
            wr_addr_busy <= 1'b0;
            wr_data_busy <= 1'b0;
            wr_beat_cnt <= {LEN_WIDTH{1'b0}};
            wr_pending <= {NUM_PORTS{1'b0}};
            
            m_axi_awid <= {ID_WIDTH{1'b0}};
            m_axi_awaddr <= {ADDR_WIDTH{1'b0}};
            m_axi_awlen <= {LEN_WIDTH{1'b0}};
            m_axi_awsize <= 3'b0;
            m_axi_awburst <= 2'b01;
            m_axi_awvalid <= 1'b0;
            
            m_axi_wdata <= {DATA_WIDTH{1'b0}};
            m_axi_wstrb <= {(DATA_WIDTH/8){1'b0}};
            m_axi_wlast <= 1'b0;
            m_axi_wvalid <= 1'b0;
            
            m_axi_bready <= 1'b0;
            
            s_axi_awready <= {NUM_PORTS{1'b0}};
            s_axi_wready <= {NUM_PORTS{1'b0}};
            
            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                s_axi_bid[i*ID_WIDTH +: ID_WIDTH] <= {ID_WIDTH{1'b0}};
                s_axi_bresp[i*2 +: 2] <= RESP_OKAY;
                s_axi_bvalid[i] <= 1'b0;
            end
            
            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                wr_id_store[i] <= {ID_WIDTH{1'b0}};
                wr_addr_store[i] <= {ADDR_WIDTH{1'b0}};
                wr_len_store[i] <= {LEN_WIDTH{1'b0}};
                wr_size_store[i] <= 3'b0;
                wr_burst_store[i] <= 2'b0;
            end
            
        end else begin
            // Default ready signals
            s_axi_awready <= {NUM_PORTS{1'b0}};
            s_axi_wready <= {NUM_PORTS{1'b0}};
            
            // Clear write response valid when accepted
            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                if (s_axi_bvalid[i] && s_axi_bready[i]) begin
                    s_axi_bvalid[i] <= 1'b0;
                end
            end
            
            // Write address phase
            if (!wr_addr_busy) begin
                wr_grant <= wr_grant_next;
                
                if (s_axi_awvalid[wr_grant_next]) begin
                    // Store transaction info
                    wr_id_store[wr_grant_next] <= s_axi_awid[wr_grant_next*ID_WIDTH +: ID_WIDTH];
                    wr_addr_store[wr_grant_next] <= s_axi_awaddr[wr_grant_next*ADDR_WIDTH +: ADDR_WIDTH];
                    wr_len_store[wr_grant_next] <= s_axi_awlen[wr_grant_next*LEN_WIDTH +: LEN_WIDTH];
                    wr_size_store[wr_grant_next] <= s_axi_awsize[wr_grant_next*3 +: 3];
                    wr_burst_store[wr_grant_next] <= s_axi_awburst[wr_grant_next*2 +: 2];
                    
                    // Forward to master port
                    m_axi_awid <= s_axi_awid[wr_grant_next*ID_WIDTH +: ID_WIDTH];
                    m_axi_awaddr <= s_axi_awaddr[wr_grant_next*ADDR_WIDTH +: ADDR_WIDTH];
                    m_axi_awlen <= s_axi_awlen[wr_grant_next*LEN_WIDTH +: LEN_WIDTH];
                    m_axi_awsize <= s_axi_awsize[wr_grant_next*3 +: 3];
                    m_axi_awburst <= s_axi_awburst[wr_grant_next*2 +: 2];
                    m_axi_awvalid <= 1'b1;
                    
                    s_axi_awready[wr_grant_next] <= 1'b1;
                    wr_addr_busy <= 1'b1;
                    wr_data_busy <= 1'b1;
                    wr_beat_cnt <= s_axi_awlen[wr_grant_next*LEN_WIDTH +: LEN_WIDTH];
                    wr_pending[wr_grant_next] <= 1'b1;
                end
                
            end else begin
                // Address handshake complete
                if (m_axi_awvalid && m_axi_awready) begin
                    m_axi_awvalid <= 1'b0;
                end
            end
            
            // Write data phase
            if (wr_data_busy) begin
                // Forward write data from granted port
                if (s_axi_wvalid[wr_grant] && !m_axi_wvalid) begin
                    m_axi_wdata <= s_axi_wdata[wr_grant*DATA_WIDTH +: DATA_WIDTH];
                    m_axi_wstrb <= s_axi_wstrb[wr_grant*(DATA_WIDTH/8) +: DATA_WIDTH/8];
                    m_axi_wlast <= s_axi_wlast[wr_grant];
                    m_axi_wvalid <= 1'b1;
                    s_axi_wready[wr_grant] <= 1'b1;
                end
                
                // Data handshake
                if (m_axi_wvalid && m_axi_wready) begin
                    m_axi_wvalid <= 1'b0;
                    
                    if (m_axi_wlast) begin
                        wr_data_busy <= 1'b0;
                        m_axi_bready <= 1'b1;
                    end
                end
            end
            
            // Write response phase
            if (m_axi_bvalid && m_axi_bready) begin
                s_axi_bid[wr_grant*ID_WIDTH +: ID_WIDTH] <= m_axi_bid;
                s_axi_bresp[wr_grant*2 +: 2] <= m_axi_bresp;
                s_axi_bvalid[wr_grant] <= 1'b1;
                
                m_axi_bready <= 1'b0;
                wr_addr_busy <= 1'b0;
                wr_pending[wr_grant] <= 1'b0;
            end
        end
    end
    
    // =========================================================================
    // Status Outputs
    // =========================================================================
    
    assign port_active = rd_pending | wr_pending;
    assign current_read_port = rd_grant;
    assign current_write_port = wr_grant;

endmodule

`default_nettype wire
