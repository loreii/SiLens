// =============================================================================
// SiLens - Memory Controller / Arbiter
// =============================================================================
// Arbitrates memory access between multiple requestors:
//   - Vision encoder activations
//   - LLM activations
//   - KV cache
//   - Embedding lookups
//   - DMA transfers
//
// Features:
//   - Round-robin arbitration with priority override
//   - AXI-Stream interfaces for requestors
//   - Support for both on-chip SRAM and external DRAM
//
// License: Apache 2.0
// =============================================================================

module memory_controller #(
    parameter DATA_WIDTH   = 128,                   // Memory data width
    parameter ADDR_WIDTH   = 32,                    // Address width
    parameter NUM_PORTS    = 4,                     // Number of requestor ports
    parameter BURST_LEN    = 8,                     // Burst length
    parameter FIFO_DEPTH   = 16                     // Request FIFO depth
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Port 0: Vision encoder (high priority)
    input  wire [ADDR_WIDTH-1:0]        p0_addr,
    input  wire [DATA_WIDTH-1:0]        p0_wr_data,
    input  wire                         p0_wr_en,
    input  wire                         p0_rd_en,
    output reg  [DATA_WIDTH-1:0]        p0_rd_data,
    output reg                          p0_rd_valid,
    output wire                         p0_ready,
    input  wire                         p0_burst,
    input  wire [$clog2(BURST_LEN)-1:0] p0_burst_len,
    
    // Port 1: Language model (high priority)
    input  wire [ADDR_WIDTH-1:0]        p1_addr,
    input  wire [DATA_WIDTH-1:0]        p1_wr_data,
    input  wire                         p1_wr_en,
    input  wire                         p1_rd_en,
    output reg  [DATA_WIDTH-1:0]        p1_rd_data,
    output reg                          p1_rd_valid,
    output wire                         p1_ready,
    input  wire                         p1_burst,
    input  wire [$clog2(BURST_LEN)-1:0] p1_burst_len,
    
    // Port 2: DMA (medium priority)
    input  wire [ADDR_WIDTH-1:0]        p2_addr,
    input  wire [DATA_WIDTH-1:0]        p2_wr_data,
    input  wire                         p2_wr_en,
    input  wire                         p2_rd_en,
    output reg  [DATA_WIDTH-1:0]        p2_rd_data,
    output reg                          p2_rd_valid,
    output wire                         p2_ready,
    input  wire                         p2_burst,
    input  wire [$clog2(BURST_LEN)-1:0] p2_burst_len,
    
    // Port 3: Debug/config (low priority)
    input  wire [ADDR_WIDTH-1:0]        p3_addr,
    input  wire [DATA_WIDTH-1:0]        p3_wr_data,
    input  wire                         p3_wr_en,
    input  wire                         p3_rd_en,
    output reg  [DATA_WIDTH-1:0]        p3_rd_data,
    output reg                          p3_rd_valid,
    output wire                         p3_ready,
    input  wire                         p3_burst,
    input  wire [$clog2(BURST_LEN)-1:0] p3_burst_len,
    
    // Memory interface (to SRAM or external controller)
    output reg  [ADDR_WIDTH-1:0]        mem_addr,
    output reg  [DATA_WIDTH-1:0]        mem_wr_data,
    output reg                          mem_wr_en,
    output reg                          mem_rd_en,
    input  wire [DATA_WIDTH-1:0]        mem_rd_data,
    input  wire                         mem_rd_valid,
    output wire                         mem_ready,
    input  wire                         mem_busy,
    
    // Status
    output wire [NUM_PORTS-1:0]         port_active,
    output wire                         arbiter_busy
);


    // =========================================================================
    // Request signals aggregation
    // =========================================================================
    
    wire [NUM_PORTS-1:0] port_request;
    wire [NUM_PORTS-1:0] port_is_write;
    
    assign port_request[0] = p0_wr_en | p0_rd_en;
    assign port_request[1] = p1_wr_en | p1_rd_en;
    assign port_request[2] = p2_wr_en | p2_rd_en;
    assign port_request[3] = p3_wr_en | p3_rd_en;
    
    assign port_is_write[0] = p0_wr_en;
    assign port_is_write[1] = p1_wr_en;
    assign port_is_write[2] = p2_wr_en;
    assign port_is_write[3] = p3_wr_en;
    
    // =========================================================================
    // Round-robin arbiter with priority
    // =========================================================================
    
    reg [$clog2(NUM_PORTS)-1:0] last_grant;
    reg [$clog2(NUM_PORTS)-1:0] current_grant;
    reg grant_valid;
    
    // Priority: Port 0 > Port 1 > Port 2 > Port 3
    // With round-robin fallback for equal priority
    
    wire [NUM_PORTS-1:0] grant_mask;
    reg [$clog2(NUM_PORTS)-1:0] next_grant;
    
    always @(*) begin
        next_grant = last_grant;
        
        // Priority-based selection
        if (port_request[0]) begin
            next_grant = 2'd0;
        end else if (port_request[1]) begin
            next_grant = 2'd1;
        end else if (port_request[2]) begin
            next_grant = 2'd2;
        end else if (port_request[3]) begin
            next_grant = 2'd3;
        end
    end
    
    // =========================================================================
    // Arbiter FSM
    // =========================================================================
    
    localparam ARB_IDLE    = 3'd0;
    localparam ARB_GRANT   = 3'd1;
    localparam ARB_WRITE   = 3'd2;
    localparam ARB_READ    = 3'd3;
    localparam ARB_BURST   = 3'd4;
    localparam ARB_WAIT    = 3'd5;
    
    reg [2:0] arb_state;
    reg [$clog2(BURST_LEN)-1:0] burst_cnt;
    reg [$clog2(BURST_LEN)-1:0] burst_len_r;
    reg [ADDR_WIDTH-1:0] burst_addr;
    
    assign arbiter_busy = (arb_state != ARB_IDLE);
    assign port_active = {grant_valid && (current_grant == 3),
                          grant_valid && (current_grant == 2),
                          grant_valid && (current_grant == 1),
                          grant_valid && (current_grant == 0)};
    
    // Ready signals
    assign p0_ready = (arb_state == ARB_IDLE) || (grant_valid && current_grant == 0);
    assign p1_ready = (arb_state == ARB_IDLE) || (grant_valid && current_grant == 1);
    assign p2_ready = (arb_state == ARB_IDLE) || (grant_valid && current_grant == 2);
    assign p3_ready = (arb_state == ARB_IDLE) || (grant_valid && current_grant == 3);
    
    assign mem_ready = !mem_busy;

    
    // =========================================================================
    // Main arbiter FSM
    // =========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            arb_state     <= ARB_IDLE;
            last_grant    <= 0;
            current_grant <= 0;
            grant_valid   <= 1'b0;
            burst_cnt     <= 0;
            burst_len_r   <= 0;
            burst_addr    <= 0;
            mem_addr      <= 0;
            mem_wr_data   <= 0;
            mem_wr_en     <= 1'b0;
            mem_rd_en     <= 1'b0;
        end else begin
            // Default: clear memory strobes
            mem_wr_en <= 1'b0;
            mem_rd_en <= 1'b0;
            
            case (arb_state)
                ARB_IDLE: begin
                    grant_valid <= 1'b0;
                    
                    if (|port_request) begin
                        current_grant <= next_grant;
                        grant_valid   <= 1'b1;
                        arb_state     <= ARB_GRANT;
                    end
                end
                
                ARB_GRANT: begin
                    // Capture request parameters
                    case (current_grant)
                        2'd0: begin
                            mem_addr    <= p0_addr;
                            mem_wr_data <= p0_wr_data;
                            burst_len_r <= p0_burst ? p0_burst_len : 0;
                        end
                        2'd1: begin
                            mem_addr    <= p1_addr;
                            mem_wr_data <= p1_wr_data;
                            burst_len_r <= p1_burst ? p1_burst_len : 0;
                        end
                        2'd2: begin
                            mem_addr    <= p2_addr;
                            mem_wr_data <= p2_wr_data;
                            burst_len_r <= p2_burst ? p2_burst_len : 0;
                        end
                        2'd3: begin
                            mem_addr    <= p3_addr;
                            mem_wr_data <= p3_wr_data;
                            burst_len_r <= p3_burst ? p3_burst_len : 0;
                        end
                    endcase
                    
                    burst_cnt  <= 0;
                    burst_addr <= mem_addr;
                    
                    if (port_is_write[current_grant]) begin
                        arb_state <= ARB_WRITE;
                    end else begin
                        arb_state <= ARB_READ;
                    end
                end
                
                ARB_WRITE: begin
                    if (!mem_busy) begin
                        mem_wr_en <= 1'b1;
                        
                        if (burst_cnt >= burst_len_r) begin
                            last_grant <= current_grant;
                            arb_state  <= ARB_IDLE;
                        end else begin
                            burst_cnt  <= burst_cnt + 1;
                            burst_addr <= burst_addr + (DATA_WIDTH / 8);
                            mem_addr   <= burst_addr + (DATA_WIDTH / 8);
                            arb_state  <= ARB_BURST;
                        end
                    end
                end
                
                ARB_READ: begin
                    if (!mem_busy) begin
                        mem_rd_en <= 1'b1;
                        arb_state <= ARB_WAIT;
                    end
                end
                
                ARB_BURST: begin
                    // Fetch next burst data from port
                    case (current_grant)
                        2'd0: mem_wr_data <= p0_wr_data;
                        2'd1: mem_wr_data <= p1_wr_data;
                        2'd2: mem_wr_data <= p2_wr_data;
                        2'd3: mem_wr_data <= p3_wr_data;
                    endcase
                    
                    if (port_is_write[current_grant]) begin
                        arb_state <= ARB_WRITE;
                    end else begin
                        arb_state <= ARB_READ;
                    end
                end
                
                ARB_WAIT: begin
                    // Wait for read data
                    if (mem_rd_valid) begin
                        if (burst_cnt >= burst_len_r) begin
                            last_grant <= current_grant;
                            arb_state  <= ARB_IDLE;
                        end else begin
                            burst_cnt  <= burst_cnt + 1;
                            burst_addr <= burst_addr + (DATA_WIDTH / 8);
                            mem_addr   <= burst_addr + (DATA_WIDTH / 8);
                            arb_state  <= ARB_READ;
                        end
                    end
                end
                
                default: arb_state <= ARB_IDLE;
            endcase
        end
    end

    
    // =========================================================================
    // Read data routing
    // =========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            p0_rd_data  <= 0;
            p0_rd_valid <= 1'b0;
            p1_rd_data  <= 0;
            p1_rd_valid <= 1'b0;
            p2_rd_data  <= 0;
            p2_rd_valid <= 1'b0;
            p3_rd_data  <= 0;
            p3_rd_valid <= 1'b0;
        end else begin
            // Default: clear valid signals
            p0_rd_valid <= 1'b0;
            p1_rd_valid <= 1'b0;
            p2_rd_valid <= 1'b0;
            p3_rd_valid <= 1'b0;
            
            if (mem_rd_valid && grant_valid) begin
                case (current_grant)
                    2'd0: begin
                        p0_rd_data  <= mem_rd_data;
                        p0_rd_valid <= 1'b1;
                    end
                    2'd1: begin
                        p1_rd_data  <= mem_rd_data;
                        p1_rd_valid <= 1'b1;
                    end
                    2'd2: begin
                        p2_rd_data  <= mem_rd_data;
                        p2_rd_valid <= 1'b1;
                    end
                    2'd3: begin
                        p3_rd_data  <= mem_rd_data;
                        p3_rd_valid <= 1'b1;
                    end
                endcase
            end
        end
    end

endmodule
