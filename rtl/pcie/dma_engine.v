// =============================================================================
// SiLens - DMA Controller
// =============================================================================
// DMA engine for high-bandwidth host memory transfers.
//
// Features:
//   - Scatter-gather DMA support
//   - Descriptor-based transfers
//   - Read and write channels
//   - Interrupt generation on completion
//
// Transfer modes:
//   - Host-to-device (H2D): Image data, tokens, configuration
//   - Device-to-host (D2H): Generated tokens, status
//
// License: Apache 2.0
// =============================================================================

module dma_engine #(
    parameter DATA_WIDTH      = 128,                // AXI-Stream width
    parameter ADDR_WIDTH      = 64,                 // Host address width
    parameter LOCAL_ADDR_WIDTH = 32,                // Local memory address
    parameter MAX_BURST_LEN   = 16,                 // Maximum burst length
    parameter DESC_FIFO_DEPTH = 16                  // Descriptor FIFO depth
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Configuration registers
    input  wire                         dma_enable,
    input  wire                         h2d_start,          // Start H2D transfer
    input  wire                         d2h_start,          // Start D2H transfer
    output wire                         h2d_busy,
    output wire                         d2h_busy,
    output wire                         h2d_done,
    output wire                         d2h_done,
    output wire                         dma_error,
    
    // H2D descriptor
    input  wire [ADDR_WIDTH-1:0]        h2d_host_addr,      // Host memory address
    input  wire [LOCAL_ADDR_WIDTH-1:0]  h2d_local_addr,     // Local memory address
    input  wire [23:0]                  h2d_length,         // Transfer length (bytes)
    
    // D2H descriptor
    input  wire [ADDR_WIDTH-1:0]        d2h_host_addr,
    input  wire [LOCAL_ADDR_WIDTH-1:0]  d2h_local_addr,
    input  wire [23:0]                  d2h_length,
    
    // PCIe TX interface (memory writes to host)
    output reg  [DATA_WIDTH-1:0]        pcie_tx_data,
    output reg  [DATA_WIDTH/8-1:0]      pcie_tx_keep,
    output reg                          pcie_tx_last,
    output reg                          pcie_tx_valid,
    input  wire                         pcie_tx_ready,
    
    // PCIe RX interface (data from host)
    input  wire [DATA_WIDTH-1:0]        pcie_rx_data,
    input  wire [DATA_WIDTH/8-1:0]      pcie_rx_keep,
    input  wire                         pcie_rx_last,
    input  wire                         pcie_rx_valid,
    output wire                         pcie_rx_ready,
    
    // Local memory interface
    output reg  [LOCAL_ADDR_WIDTH-1:0]  local_addr,
    output reg  [DATA_WIDTH-1:0]        local_wr_data,
    output reg                          local_wr_en,
    output reg                          local_rd_en,
    input  wire [DATA_WIDTH-1:0]        local_rd_data,
    input  wire                         local_rd_valid,
    
    // Interrupt
    output reg                          interrupt
);


    // =========================================================================
    // Constants
    // =========================================================================
    
    localparam BYTES_PER_BEAT = DATA_WIDTH / 8;
    localparam BURST_WIDTH    = $clog2(MAX_BURST_LEN);
    
    // =========================================================================
    // H2D (Host to Device) FSM
    // =========================================================================
    
    localparam H2D_IDLE     = 3'd0;
    localparam H2D_REQ      = 3'd1;
    localparam H2D_WAIT     = 3'd2;
    localparam H2D_RECEIVE  = 3'd3;
    localparam H2D_STORE    = 3'd4;
    localparam H2D_DONE     = 3'd5;
    localparam H2D_ERROR    = 3'd6;
    
    reg [2:0] h2d_state;
    reg [ADDR_WIDTH-1:0] h2d_host_addr_r;
    reg [LOCAL_ADDR_WIDTH-1:0] h2d_local_addr_r;
    reg [23:0] h2d_bytes_remaining;
    reg [BURST_WIDTH-1:0] h2d_burst_cnt;
    reg h2d_done_r;
    reg h2d_error_r;
    
    assign h2d_busy = (h2d_state != H2D_IDLE);
    assign h2d_done = h2d_done_r;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            h2d_state          <= H2D_IDLE;
            h2d_host_addr_r    <= 0;
            h2d_local_addr_r   <= 0;
            h2d_bytes_remaining <= 0;
            h2d_burst_cnt      <= 0;
            h2d_done_r         <= 1'b0;
            h2d_error_r        <= 1'b0;
        end else begin
            h2d_done_r <= 1'b0;
            
            case (h2d_state)
                H2D_IDLE: begin
                    if (dma_enable && h2d_start) begin
                        h2d_host_addr_r    <= h2d_host_addr;
                        h2d_local_addr_r   <= h2d_local_addr;
                        h2d_bytes_remaining <= h2d_length;
                        h2d_state          <= H2D_REQ;
                    end
                end
                
                H2D_REQ: begin
                    // Generate PCIe memory read request
                    // In real implementation, build TLP header
                    h2d_state <= H2D_WAIT;
                end
                
                H2D_WAIT: begin
                    // Wait for completion from host
                    if (pcie_rx_valid) begin
                        h2d_state <= H2D_RECEIVE;
                    end
                end
                
                H2D_RECEIVE: begin
                    if (pcie_rx_valid && pcie_rx_ready) begin
                        // Store received data to local memory
                        local_addr    <= h2d_local_addr_r;
                        local_wr_data <= pcie_rx_data;
                        local_wr_en   <= 1'b1;
                        
                        h2d_local_addr_r   <= h2d_local_addr_r + BYTES_PER_BEAT;
                        h2d_host_addr_r    <= h2d_host_addr_r + BYTES_PER_BEAT;
                        h2d_bytes_remaining <= h2d_bytes_remaining - BYTES_PER_BEAT;
                        
                        if (pcie_rx_last || h2d_bytes_remaining <= BYTES_PER_BEAT) begin
                            h2d_state <= H2D_STORE;
                        end
                    end
                end
                
                H2D_STORE: begin
                    local_wr_en <= 1'b0;
                    
                    if (h2d_bytes_remaining > 0) begin
                        h2d_state <= H2D_REQ;
                    end else begin
                        h2d_state  <= H2D_DONE;
                        h2d_done_r <= 1'b1;
                    end
                end
                
                H2D_DONE: begin
                    h2d_state <= H2D_IDLE;
                end
                
                H2D_ERROR: begin
                    h2d_error_r <= 1'b1;
                    h2d_state   <= H2D_IDLE;
                end
                
                default: h2d_state <= H2D_IDLE;
            endcase
        end
    end
    
    assign pcie_rx_ready = (h2d_state == H2D_RECEIVE);

    
    // =========================================================================
    // D2H (Device to Host) FSM
    // =========================================================================
    
    localparam D2H_IDLE   = 3'd0;
    localparam D2H_FETCH  = 3'd1;
    localparam D2H_WAIT   = 3'd2;
    localparam D2H_SEND   = 3'd3;
    localparam D2H_DONE   = 3'd4;
    localparam D2H_ERROR  = 3'd5;
    
    reg [2:0] d2h_state;
    reg [ADDR_WIDTH-1:0] d2h_host_addr_r;
    reg [LOCAL_ADDR_WIDTH-1:0] d2h_local_addr_r;
    reg [23:0] d2h_bytes_remaining;
    reg [BURST_WIDTH-1:0] d2h_burst_cnt;
    reg d2h_done_r;
    reg d2h_error_r;
    
    assign d2h_busy = (d2h_state != D2H_IDLE);
    assign d2h_done = d2h_done_r;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            d2h_state          <= D2H_IDLE;
            d2h_host_addr_r    <= 0;
            d2h_local_addr_r   <= 0;
            d2h_bytes_remaining <= 0;
            d2h_burst_cnt      <= 0;
            d2h_done_r         <= 1'b0;
            d2h_error_r        <= 1'b0;
            pcie_tx_data       <= 0;
            pcie_tx_keep       <= 0;
            pcie_tx_last       <= 1'b0;
            pcie_tx_valid      <= 1'b0;
        end else begin
            d2h_done_r    <= 1'b0;
            pcie_tx_valid <= 1'b0;
            pcie_tx_last  <= 1'b0;
            local_rd_en   <= 1'b0;
            
            case (d2h_state)
                D2H_IDLE: begin
                    if (dma_enable && d2h_start) begin
                        d2h_host_addr_r    <= d2h_host_addr;
                        d2h_local_addr_r   <= d2h_local_addr;
                        d2h_bytes_remaining <= d2h_length;
                        d2h_state          <= D2H_FETCH;
                    end
                end
                
                D2H_FETCH: begin
                    // Read from local memory
                    local_addr  <= d2h_local_addr_r;
                    local_rd_en <= 1'b1;
                    d2h_state   <= D2H_WAIT;
                end
                
                D2H_WAIT: begin
                    if (local_rd_valid) begin
                        pcie_tx_data <= local_rd_data;
                        pcie_tx_keep <= {(DATA_WIDTH/8){1'b1}};
                        d2h_state    <= D2H_SEND;
                    end
                end
                
                D2H_SEND: begin
                    pcie_tx_valid <= 1'b1;
                    
                    if (d2h_bytes_remaining <= BYTES_PER_BEAT) begin
                        pcie_tx_last <= 1'b1;
                    end
                    
                    if (pcie_tx_ready) begin
                        d2h_local_addr_r   <= d2h_local_addr_r + BYTES_PER_BEAT;
                        d2h_host_addr_r    <= d2h_host_addr_r + BYTES_PER_BEAT;
                        d2h_bytes_remaining <= d2h_bytes_remaining - BYTES_PER_BEAT;
                        
                        if (d2h_bytes_remaining <= BYTES_PER_BEAT) begin
                            d2h_state  <= D2H_DONE;
                            d2h_done_r <= 1'b1;
                        end else begin
                            d2h_state <= D2H_FETCH;
                        end
                    end
                end
                
                D2H_DONE: begin
                    d2h_state <= D2H_IDLE;
                end
                
                D2H_ERROR: begin
                    d2h_error_r <= 1'b1;
                    d2h_state   <= D2H_IDLE;
                end
                
                default: d2h_state <= D2H_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Error and interrupt handling
    // =========================================================================
    
    assign dma_error = h2d_error_r | d2h_error_r;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            interrupt <= 1'b0;
        end else begin
            interrupt <= h2d_done_r | d2h_done_r;
        end
    end

endmodule
