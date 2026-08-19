// =============================================================================
// SiLens IO Subsystem - Level 3
// =============================================================================
// Host communication and control interfaces for SiLens SoC.
// Integrates parallel host interface, SPI slave, interrupt controller,
// GPIO controller, and internal register file.
//
// Target: ~30mm² (5500µm × 5500µm) on SKY130
//
// Architecture:
//   External Host → Parallel Host Interface (high bandwidth, 16-bit data)
//                        ↓
//   Internal AXI Bus ← → Register File
//        ↓                    ↓
//   SPI Slave              GPIO/IRQ
//   (config)               (status/control)
//
// External Interfaces:
//   - Parallel Host: 16-bit data bus, 8-bit address, control signals
//   - SPI Slave: SCLK, MOSI, MISO, CS# (Mode 0, up to 25MHz)
//   - GPIO: 8 bidirectional pins with pull control
//   - IRQ: Active-low interrupt output to host
//
// Internal Interface:
//   - AXI-Lite Master: 32-bit data, 32-bit address
//
// Register Map:
//   0x0000-0x00FF: Host Interface registers (via parallel bus)
//   0x0100-0x01FF: SPI Slave control/status
//   0x0200-0x02FF: Interrupt Controller
//   0x0300-0x03FF: GPIO Controller
//   0x0400-0x04FF: System Control & Status
//   0x0500-0x05FF: Performance Counters
//   0x0600-0x06FF: Debug Access
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module io_subsystem #(
    parameter HOST_DATA_WIDTH = 16,
    parameter HOST_ADDR_WIDTH = 8,
    parameter AXI_DATA_WIDTH  = 32,
    parameter AXI_ADDR_WIDTH  = 32,
    parameter NUM_IRQ_SOURCES = 16,
    parameter NUM_GPIO_PINS   = 8
)(
    // =========================================================================
    // Clock and Reset
    // =========================================================================
    input  wire                         clk,            // Main system clock (100 MHz)
    input  wire                         rst_n,          // Active-low reset
    
    // Host clock domain (may be different from system clock)
    input  wire                         host_clk,       // Host interface clock
    input  wire                         host_rst_n,     // Host reset
    
    // =========================================================================
    // Parallel Host Interface (External - to FPGA bridge)
    // =========================================================================
    input  wire [HOST_DATA_WIDTH-1:0]   host_data_in,   // Bidirectional data in
    output wire [HOST_DATA_WIDTH-1:0]   host_data_out,  // Bidirectional data out
    output wire                         host_data_oe,   // Data output enable
    input  wire [HOST_ADDR_WIDTH-1:0]   host_addr,      // Address bus
    input  wire                         host_wr_n,      // Write strobe (active low)
    input  wire                         host_rd_n,      // Read strobe (active low)
    input  wire                         host_cs_n,      // Chip select (active low)
    output wire                         host_ready,     // Ready/wait signal
    output wire                         host_irq_n,     // Interrupt to host (active low)
    
    // =========================================================================
    // SPI Slave Interface (External - configuration port)
    // =========================================================================
    input  wire                         spi_clk,        // SPI clock (up to 25 MHz)
    input  wire                         spi_mosi,       // Master Out Slave In
    output wire                         spi_miso,       // Master In Slave Out
    input  wire                         spi_cs_n,       // Chip select (active low)
    
    // =========================================================================
    // GPIO Interface (External)
    // =========================================================================
    input  wire [NUM_GPIO_PINS-1:0]     gpio_in,        // GPIO input
    output wire [NUM_GPIO_PINS-1:0]     gpio_out,       // GPIO output
    output wire [NUM_GPIO_PINS-1:0]     gpio_oe,        // GPIO output enable
    output wire [NUM_GPIO_PINS-1:0]     gpio_pull_en,   // Pull enable
    output wire [NUM_GPIO_PINS-1:0]     gpio_pull_sel,  // Pull select (1=up)
    
    // =========================================================================
    // AXI-Lite Master Interface (Internal - to other subsystems)
    // =========================================================================
    output reg  [AXI_ADDR_WIDTH-1:0]    m_axi_awaddr,   // Write address
    output reg                          m_axi_awvalid,  // Write address valid
    input  wire                         m_axi_awready,  // Write address ready
    
    output reg  [AXI_DATA_WIDTH-1:0]    m_axi_wdata,    // Write data
    output reg  [3:0]                   m_axi_wstrb,    // Write strobes
    output reg                          m_axi_wvalid,   // Write data valid
    input  wire                         m_axi_wready,   // Write data ready
    
    input  wire [1:0]                   m_axi_bresp,    // Write response
    input  wire                         m_axi_bvalid,   // Write response valid
    output reg                          m_axi_bready,   // Write response ready
    
    output reg  [AXI_ADDR_WIDTH-1:0]    m_axi_araddr,   // Read address
    output reg                          m_axi_arvalid,  // Read address valid
    input  wire                         m_axi_arready,  // Read address ready
    
    input  wire [AXI_DATA_WIDTH-1:0]    m_axi_rdata,    // Read data
    input  wire [1:0]                   m_axi_rresp,    // Read response
    input  wire                         m_axi_rvalid,   // Read data valid
    output reg                          m_axi_rready,   // Read data ready
    
    // =========================================================================
    // Status/Control to/from other subsystems
    // =========================================================================
    // External interrupt sources from other subsystems
    input  wire [7:0]                   ext_irq_sources,
    
    // Control outputs (directly from host interface to other subsystems)
    output wire                         frame_start,
    output wire                         seq_start,
    output wire                         gen_start,
    output wire                         abort,
    
    // Status inputs from other subsystems
    input  wire [31:0]                  core_status,
    input  wire                         ddr_init_done
);

    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // Host interface internal signals
    wire [31:0] host_data_out_32;
    wire        host_data_oe_int;
    wire        host_ready_int;
    wire        host_irq_int;
    
    // Token streaming (from host interface)
    wire [15:0] token_in;
    wire        token_in_valid;
    wire        token_in_ready;
    wire [15:0] token_out;
    wire        token_out_valid;
    wire        token_out_ready;
    
    // Pixel streaming
    wire [23:0] pixel_out;
    wire        pixel_out_valid;
    wire        pixel_out_ready;
    
    // SPI register interface
    wire [7:0]  spi_reg_addr;
    wire [7:0]  spi_reg_wdata;
    wire [7:0]  spi_reg_rdata;
    wire        spi_reg_wr;
    
    // Internal register bus (from address decode)
    wire [11:0] int_reg_addr;
    wire [31:0] int_reg_wdata;
    wire [31:0] int_reg_rdata_irqc;
    wire [31:0] int_reg_rdata_gpio;
    reg  [31:0] int_reg_rdata_sys;
    reg  [31:0] int_reg_rdata;
    wire        int_reg_wr;
    wire        int_reg_rd;
    
    // Interrupt signals
    wire [NUM_IRQ_SOURCES-1:0] irq_sources_all;
    wire        irq_combined;
    wire        gpio_irq;
    
    // =========================================================================
    // Address Decode Logic
    // =========================================================================
    
    // Internal address spaces
    localparam ADDR_IRQC_BASE = 12'h200;
    localparam ADDR_GPIO_BASE = 12'h300;
    localparam ADDR_SYS_BASE  = 12'h400;
    localparam ADDR_PERF_BASE = 12'h500;
    localparam ADDR_DBG_BASE  = 12'h600;
    
    wire sel_irqc = (int_reg_addr[11:8] == 4'h2);
    wire sel_gpio = (int_reg_addr[11:8] == 4'h3);
    wire sel_sys  = (int_reg_addr[11:8] == 4'h4);
    wire sel_perf = (int_reg_addr[11:8] == 4'h5);
    wire sel_dbg  = (int_reg_addr[11:8] == 4'h6);
    
    // =========================================================================
    // Host Interface Instance
    // =========================================================================
    
    silens_host_interface #(
        .DATA_WIDTH(32),
        .ADDR_WIDTH(16),
        .TOKEN_FIFO_DEPTH(256),
        .PIXEL_FIFO_DEPTH(1024)
    ) u_host_interface (
        // Host side
        .host_clk       (host_clk),
        .host_rst_n     (host_rst_n),
        .host_data_in   ({16'b0, host_data_in}),
        .host_data_out  (host_data_out_32),
        .host_data_oe   (host_data_oe_int),
        .host_addr      ({8'b0, host_addr}),
        .host_rd_n      (host_rd_n),
        .host_wr_n      (host_wr_n),
        .host_cs_n      (host_cs_n),
        .host_ready     (host_ready_int),
        .host_irq       (host_irq_int),
        
        // Core side
        .core_clk       (clk),
        .core_rst_n     (rst_n),
        
        // Control outputs
        .frame_start    (frame_start),
        .seq_start      (seq_start),
        .gen_start      (gen_start),
        .abort          (abort),
        
        // Token streaming
        .token_in       (token_in),
        .token_in_valid (token_in_valid),
        .token_in_ready (token_in_ready),
        .token_out      (token_out),
        .token_out_valid(token_out_valid),
        .token_out_ready(token_out_ready),
        
        // Pixel streaming
        .pixel_out      (pixel_out),
        .pixel_out_valid(pixel_out_valid),
        .pixel_out_ready(pixel_out_ready),
        
        // Status
        .status         (core_status),
        .ddr_init_done  (ddr_init_done)
    );
    
    // Extract 16-bit output from 32-bit interface
    assign host_data_out = host_data_out_32[HOST_DATA_WIDTH-1:0];
    assign host_data_oe  = host_data_oe_int;
    assign host_ready    = host_ready_int;
    assign host_irq_n    = ~(host_irq_int | irq_combined);
    
    // Token/pixel interfaces directly tied for now (no external connection)
    assign token_in_ready  = 1'b1;
    assign token_out       = 16'h0;
    assign token_out_valid = 1'b0;
    assign pixel_out_ready = 1'b1;
    
    // =========================================================================
    // SPI Slave Instance
    // =========================================================================
    
    silens_spi_slave u_spi_slave (
        .clk        (clk),
        .rst_n      (rst_n),
        
        // SPI pins
        .spi_clk    (spi_clk),
        .spi_mosi   (spi_mosi),
        .spi_miso   (spi_miso),
        .spi_cs_n   (spi_cs_n),
        
        // Register interface
        .reg_addr   (spi_reg_addr),
        .reg_wdata  (spi_reg_wdata),
        .reg_rdata  (spi_reg_rdata),
        .reg_wr     (spi_reg_wr)
    );
    
    // =========================================================================
    // Interrupt Controller Instance
    // =========================================================================
    
    // Combine interrupt sources
    // [7:0]:   External interrupts from other subsystems
    // [8]:     GPIO change interrupt
    // [9]:     SPI transaction complete
    // [15:10]: Reserved
    assign irq_sources_all = {
        6'b0,                   // [15:10] Reserved
        1'b0,                   // [9] SPI (placeholder)
        gpio_irq,               // [8] GPIO
        ext_irq_sources         // [7:0] External
    };
    
    interrupt_controller #(
        .NUM_SOURCES(NUM_IRQ_SOURCES)
    ) u_irq_ctrl (
        .clk         (clk),
        .rst_n       (rst_n),
        
        .irq_sources (irq_sources_all),
        .irq_out     (irq_combined),
        
        // Register interface
        .reg_addr    (int_reg_addr[4:0]),
        .reg_wdata   (int_reg_wdata),
        .reg_rdata   (int_reg_rdata_irqc),
        .reg_wr      (int_reg_wr & sel_irqc),
        .reg_rd      (int_reg_rd & sel_irqc)
    );
    
    // =========================================================================
    // GPIO Controller Instance
    // =========================================================================
    
    gpio_controller #(
        .NUM_PINS(NUM_GPIO_PINS),
        .DEBOUNCE_CYCLES(4)
    ) u_gpio_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        
        // GPIO pins
        .gpio_in      (gpio_in),
        .gpio_out     (gpio_out),
        .gpio_oe      (gpio_oe),
        .gpio_pull_en (gpio_pull_en),
        .gpio_pull_sel(gpio_pull_sel),
        
        // Interrupt
        .gpio_irq     (gpio_irq),
        
        // Register interface
        .reg_addr     (int_reg_addr[4:0]),
        .reg_wdata    (int_reg_wdata),
        .reg_rdata    (int_reg_rdata_gpio),
        .reg_wr       (int_reg_wr & sel_gpio),
        .reg_rd       (int_reg_rd & sel_gpio)
    );
    
    // =========================================================================
    // System Control Registers
    // =========================================================================
    
    // System registers
    reg [31:0] sys_ctrl_reg;
    reg [31:0] sys_scratch_reg;
    
    // Performance counters
    reg [31:0] perf_host_rd_cnt;
    reg [31:0] perf_host_wr_cnt;
    reg [31:0] perf_irq_cnt;
    
    // System control register addresses
    localparam SYS_REG_CTRL     = 5'h00;
    localparam SYS_REG_STATUS   = 5'h04;
    localparam SYS_REG_VERSION  = 5'h08;
    localparam SYS_REG_SCRATCH  = 5'h0C;
    
    // Version constant
    localparam VERSION = 32'h0001_0000;  // v1.0.0
    
    // System register read data
    always @(*) begin
        int_reg_rdata_sys = 32'h0;
        
        if (int_reg_rd && sel_sys) begin
            case (int_reg_addr[4:0])
                SYS_REG_CTRL:    int_reg_rdata_sys = sys_ctrl_reg;
                SYS_REG_STATUS:  int_reg_rdata_sys = core_status;
                SYS_REG_VERSION: int_reg_rdata_sys = VERSION;
                SYS_REG_SCRATCH: int_reg_rdata_sys = sys_scratch_reg;
                default:         int_reg_rdata_sys = 32'hDEAD_BEEF;
            endcase
        end
    end
    
    // System register writes
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sys_ctrl_reg    <= 32'h0;
            sys_scratch_reg <= 32'h0;
        end else if (int_reg_wr && sel_sys) begin
            case (int_reg_addr[4:0])
                SYS_REG_CTRL:    sys_ctrl_reg    <= int_reg_wdata;
                SYS_REG_SCRATCH: sys_scratch_reg <= int_reg_wdata;
            endcase
        end
    end
    
    // =========================================================================
    // Performance Counters
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_host_rd_cnt <= 32'h0;
            perf_host_wr_cnt <= 32'h0;
            perf_irq_cnt     <= 32'h0;
        end else begin
            // Count host transactions (simplified - would need CDC in real design)
            if (irq_combined)
                perf_irq_cnt <= perf_irq_cnt + 1;
        end
    end
    
    // =========================================================================
    // SPI Register Bridge
    // =========================================================================
    
    // Map 8-bit SPI addresses to internal register space
    // SPI can access system control registers (0x400-0x4FF range)
    assign spi_reg_rdata = int_reg_rdata_sys[7:0];
    
    // =========================================================================
    // Internal Register Bus Mux
    // =========================================================================
    
    // Address and data from host interface (directly mapped)
    assign int_reg_addr  = {4'b0, host_addr};
    assign int_reg_wdata = {16'b0, host_data_in};
    assign int_reg_wr    = ~host_cs_n & ~host_wr_n;
    assign int_reg_rd    = ~host_cs_n & ~host_rd_n;
    
    // Read data mux
    always @(*) begin
        if (sel_irqc)
            int_reg_rdata = int_reg_rdata_irqc;
        else if (sel_gpio)
            int_reg_rdata = int_reg_rdata_gpio;
        else if (sel_sys)
            int_reg_rdata = int_reg_rdata_sys;
        else
            int_reg_rdata = 32'hDEAD_BEEF;
    end
    
    // =========================================================================
    // AXI-Lite Master (Bridge from host to internal AXI bus)
    // =========================================================================
    
    // Simple AXI-Lite master state machine
    localparam AXI_IDLE    = 3'd0;
    localparam AXI_WR_ADDR = 3'd1;
    localparam AXI_WR_DATA = 3'd2;
    localparam AXI_WR_RESP = 3'd3;
    localparam AXI_RD_ADDR = 3'd4;
    localparam AXI_RD_DATA = 3'd5;
    
    reg [2:0] axi_state;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_state     <= AXI_IDLE;
            m_axi_awaddr  <= 32'h0;
            m_axi_awvalid <= 1'b0;
            m_axi_wdata   <= 32'h0;
            m_axi_wstrb   <= 4'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_bready  <= 1'b0;
            m_axi_araddr  <= 32'h0;
            m_axi_arvalid <= 1'b0;
            m_axi_rready  <= 1'b0;
        end else begin
            case (axi_state)
                AXI_IDLE: begin
                    m_axi_awvalid <= 1'b0;
                    m_axi_wvalid  <= 1'b0;
                    m_axi_bready  <= 1'b0;
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready  <= 1'b0;
                    
                    // Check for external AXI transaction request
                    // (triggered by specific control register writes)
                    if (sys_ctrl_reg[0]) begin
                        // AXI write request
                        m_axi_awaddr  <= {20'h0, int_reg_addr};
                        m_axi_awvalid <= 1'b1;
                        m_axi_wdata   <= int_reg_wdata;
                        m_axi_wstrb   <= 4'b1111;
                        axi_state     <= AXI_WR_ADDR;
                    end else if (sys_ctrl_reg[1]) begin
                        // AXI read request
                        m_axi_araddr  <= {20'h0, int_reg_addr};
                        m_axi_arvalid <= 1'b1;
                        axi_state     <= AXI_RD_ADDR;
                    end
                end
                
                AXI_WR_ADDR: begin
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b1;
                        axi_state     <= AXI_WR_DATA;
                    end
                end
                
                AXI_WR_DATA: begin
                    if (m_axi_wready) begin
                        m_axi_wvalid <= 1'b0;
                        m_axi_bready <= 1'b1;
                        axi_state    <= AXI_WR_RESP;
                    end
                end
                
                AXI_WR_RESP: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 1'b0;
                        axi_state    <= AXI_IDLE;
                    end
                end
                
                AXI_RD_ADDR: begin
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        axi_state     <= AXI_RD_DATA;
                    end
                end
                
                AXI_RD_DATA: begin
                    if (m_axi_rvalid) begin
                        m_axi_rready <= 1'b0;
                        // m_axi_rdata available for reading
                        axi_state    <= AXI_IDLE;
                    end
                end
                
                default: axi_state <= AXI_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
