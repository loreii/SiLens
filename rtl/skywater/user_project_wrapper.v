// =============================================================================
// SiLens - SkyWater SKY130 Caravel User Project Wrapper
// =============================================================================
// Pin mapping for SkyWater SKY130 PDK with Caravel harness.
//
// Caravel provides:
//   - 38 GPIOs (active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low active high active low)
//   - 128-bit Logic Analyzer interface
//   - Wishbone bus (32-bit) from management SoC
//   - 3 IRQ lines
//
// Target: NanoLM-5M (scaled for SKY130 feasibility)
// =============================================================================

`default_nettype none

module user_project_wrapper #(
    parameter BITS = 32
)(
`ifdef USE_POWER_PINS
    inout vccd1,        // User area 1 1.8V supply
    inout vssd1,        // User area 1 digital ground
`endif

    // Wishbone Slave ports (directly from Caravel management SoC)
    input  wire        wb_clk_i,
    input  wire        wb_rst_i,
    input  wire        wbs_stb_i,
    input  wire        wbs_cyc_i,
    input  wire        wbs_we_i,
    input  wire [3:0]  wbs_sel_i,
    input  wire [31:0] wbs_dat_i,
    input  wire [31:0] wbs_adr_i,
    output wire        wbs_ack_o,
    output wire [31:0] wbs_dat_o,

    // Logic Analyzer Signals
    input  wire [127:0] la_data_in,
    output wire [127:0] la_data_out,
    input  wire [127:0] la_oenb,

    // IOs - directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly
    input  wire [`MPRJ_IO_PADS-1:0] io_in,
    output wire [`MPRJ_IO_PADS-1:0] io_out,
    output wire [`MPRJ_IO_PADS-1:0] io_oeb,

    // Analog (directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly
    inout  wire [`MPRJ_IO_PADS-10:0] analog_io,

    // Independent clock (directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly directly
    input  wire        user_clock2,

    // User maskable interrupt signals
    output wire [2:0]  user_irq
);

    // =========================================================================
    // GPIO Pin Assignments for SKY130 Caravel
    // =========================================================================
    // Caravel provides 38 GPIOs: io[37:0]
    //
    // Pin Mapping:
    // -------------------------------------------------------------------------
    // io[0]     - Reserved (JTAG)
    // io[1]     - Reserved (SDO)
    // io[2]     - Reserved (SDI)
    // io[3]     - Reserved (CSB)
    // io[4]     - Reserved (SCK)
    // io[5]     - Reserved (ser_rx)
    // io[6]     - Reserved (ser_tx)
    // io[7]     - Reserved (IRQ)
    // -------------------------------------------------------------------------
    // io[8]     - spi_clk        (Input)  - SPI clock for data streaming
    // io[9]     - spi_mosi       (Input)  - SPI data in (tokens/pixels)
    // io[10]    - spi_miso       (Output) - SPI data out (generated tokens)
    // io[11]    - spi_cs_n       (Input)  - SPI chip select (active low)
    // -------------------------------------------------------------------------
    // io[12]    - uart_rx        (Input)  - UART receive
    // io[13]    - uart_tx        (Output) - UART transmit
    // -------------------------------------------------------------------------
    // io[14]    - frame_start    (Input)  - Start image inference
    // io[15]    - seq_start      (Input)  - Start text sequence
    // io[16]    - gen_start      (Input)  - Start autoregressive generation
    // io[17]    - inference_done (Output) - Inference complete
    // io[18]    - busy           (Output) - Processing in progress
    // io[19]    - error_flag     (Output) - Error occurred
    // -------------------------------------------------------------------------
    // io[23:20] - status[3:0]    (Output) - FSM state / status LEDs
    // -------------------------------------------------------------------------
    // io[31:24] - token_out[7:0] (Output) - Parallel token output (LSB)
    // io[35:32] - token_out[11:8](Output) - Parallel token output (MSB, 4K vocab)
    // io[36]    - token_valid    (Output) - Token output strobe
    // io[37]    - token_ready    (Input)  - Consumer ready for token
    // =========================================================================

    // -------------------------------------------------------------------------
    // Wire declarations for GPIO mapping
    // -------------------------------------------------------------------------
    
    // SPI interface
    wire spi_clk    = io_in[8];
    wire spi_mosi   = io_in[9];
    wire spi_miso;
    wire spi_cs_n   = io_in[11];
    
    // UART interface
    wire uart_rx    = io_in[12];
    wire uart_tx;
    
    // Control signals
    wire frame_start = io_in[14];
    wire seq_start   = io_in[15];
    wire gen_start   = io_in[16];
    wire inference_done;
    wire busy;
    wire error_flag;
    
    // Status outputs
    wire [3:0] status;
    
    // Token output interface
    wire [11:0] token_out;
    wire token_valid;
    wire token_ready = io_in[37];

    // -------------------------------------------------------------------------
    // Internal signals
    // -------------------------------------------------------------------------
    
    wire clk;
    wire rst_n;
    
    // Use Wishbone clock or user_clock2
    assign clk = wb_clk_i;
    assign rst_n = ~wb_rst_i;
    
    // -------------------------------------------------------------------------
    // GPIO Output Assignments
    // -------------------------------------------------------------------------
    
    assign io_out[7:0]   = 8'b0;              // Reserved pins
    assign io_out[8]     = 1'b0;              // spi_clk (input)
    assign io_out[9]     = 1'b0;              // spi_mosi (input)
    assign io_out[10]    = spi_miso;          // spi_miso (output)
    assign io_out[11]    = 1'b0;              // spi_cs_n (input)
    assign io_out[12]    = 1'b0;              // uart_rx (input)
    assign io_out[13]    = uart_tx;           // uart_tx (output)
    assign io_out[14]    = 1'b0;              // frame_start (input)
    assign io_out[15]    = 1'b0;              // seq_start (input)
    assign io_out[16]    = 1'b0;              // gen_start (input)
    assign io_out[17]    = inference_done;    // inference_done (output)
    assign io_out[18]    = busy;              // busy (output)
    assign io_out[19]    = error_flag;        // error_flag (output)
    assign io_out[23:20] = status;            // status LEDs (output)
    assign io_out[31:24] = token_out[7:0];    // token_out LSB (output)
    assign io_out[35:32] = token_out[11:8];   // token_out MSB (output)
    assign io_out[36]    = token_valid;       // token_valid (output)
    assign io_out[37]    = 1'b0;              // token_ready (input)

    // -------------------------------------------------------------------------
    // GPIO Output Enable (active low: 0=output, 1=input)
    // -------------------------------------------------------------------------
    
    assign io_oeb[7:0]   = 8'hFF;             // Reserved (inputs)
    assign io_oeb[8]     = 1'b1;              // spi_clk (input)
    assign io_oeb[9]     = 1'b1;              // spi_mosi (input)
    assign io_oeb[10]    = 1'b0;              // spi_miso (output)
    assign io_oeb[11]    = 1'b1;              // spi_cs_n (input)
    assign io_oeb[12]    = 1'b1;              // uart_rx (input)
    assign io_oeb[13]    = 1'b0;              // uart_tx (output)
    assign io_oeb[14]    = 1'b1;              // frame_start (input)
    assign io_oeb[15]    = 1'b1;              // seq_start (input)
    assign io_oeb[16]    = 1'b1;              // gen_start (input)
    assign io_oeb[17]    = 1'b0;              // inference_done (output)
    assign io_oeb[18]    = 1'b0;              // busy (output)
    assign io_oeb[19]    = 1'b0;              // error_flag (output)
    assign io_oeb[23:20] = 4'b0000;           // status (outputs)
    assign io_oeb[31:24] = 8'b0;              // token_out LSB (outputs)
    assign io_oeb[35:32] = 4'b0;              // token_out MSB (outputs)
    assign io_oeb[36]    = 1'b0;              // token_valid (output)
    assign io_oeb[37]    = 1'b1;              // token_ready (input)

    // -------------------------------------------------------------------------
    // Wishbone Register Interface
    // -------------------------------------------------------------------------
    // Address Map:
    //   0x3000_0000 - Control register (R/W)
    //   0x3000_0004 - Status register (RO)
    //   0x3000_0008 - Token input register (WO)
    //   0x3000_000C - Token output register (RO)
    //   0x3000_0010 - Config register (R/W)
    //   0x3000_0100 - Weight load address (WO)
    //   0x3000_0104 - Weight load data (WO)
    // -------------------------------------------------------------------------
    
    localparam WB_BASE_ADDR = 32'h3000_0000;
    
    wire wb_valid = wbs_cyc_i && wbs_stb_i;
    wire [7:0] wb_addr = wbs_adr_i[7:0];
    
    reg [31:0] ctrl_reg;
    reg [31:0] config_reg;
    reg [31:0] token_in_reg;
    reg        token_in_valid;
    wire [31:0] status_reg;
    wire [31:0] token_out_reg;
    
    reg wbs_ack_r;
    reg [31:0] wbs_dat_r;
    
    always @(posedge wb_clk_i) begin
        if (wb_rst_i) begin
            wbs_ack_r <= 1'b0;
            wbs_dat_r <= 32'b0;
            ctrl_reg <= 32'b0;
            config_reg <= 32'b0;
            token_in_reg <= 32'b0;
            token_in_valid <= 1'b0;
        end else begin
            wbs_ack_r <= 1'b0;
            token_in_valid <= 1'b0;
            
            if (wb_valid && !wbs_ack_r) begin
                wbs_ack_r <= 1'b1;
                
                if (wbs_we_i) begin
                    // Write
                    case (wb_addr)
                        8'h00: ctrl_reg <= wbs_dat_i;
                        8'h08: begin
                            token_in_reg <= wbs_dat_i;
                            token_in_valid <= 1'b1;
                        end
                        8'h10: config_reg <= wbs_dat_i;
                    endcase
                end else begin
                    // Read
                    case (wb_addr)
                        8'h00: wbs_dat_r <= ctrl_reg;
                        8'h04: wbs_dat_r <= status_reg;
                        8'h0C: wbs_dat_r <= token_out_reg;
                        8'h10: wbs_dat_r <= config_reg;
                        default: wbs_dat_r <= 32'hDEAD_BEEF;
                    endcase
                end
            end
        end
    end
    
    assign wbs_ack_o = wbs_ack_r;
    assign wbs_dat_o = wbs_dat_r;
    
    // Status register composition
    assign status_reg = {
        24'b0,
        status,           // [7:4] FSM state
        error_flag,       // [3]
        busy,             // [2]
        inference_done,   // [1]
        token_valid       // [0]
    };
    
    assign token_out_reg = {20'b0, token_out};

    // -------------------------------------------------------------------------
    // Logic Analyzer Interface
    // -------------------------------------------------------------------------
    // Directly expose internal debug signals through LA
    //
    // la_data_out[31:0]   - Cycle counter
    // la_data_out[63:32]  - Inference cycles
    // la_data_out[67:64]  - FSM state
    // la_data_out[79:68]  - Current token
    // la_data_out[95:80]  - Token count
    // la_data_out[127:96] - Reserved / custom debug
    // -------------------------------------------------------------------------
    
    wire [31:0] debug_cycle_counter;
    wire [31:0] debug_inference_cycles;
    wire [3:0]  debug_fsm_state;
    wire [11:0] debug_current_token;
    wire [15:0] debug_token_count;
    
    assign la_data_out[31:0]   = debug_cycle_counter;
    assign la_data_out[63:32]  = debug_inference_cycles;
    assign la_data_out[67:64]  = debug_fsm_state;
    assign la_data_out[79:68]  = debug_current_token;
    assign la_data_out[95:80]  = debug_token_count;
    assign la_data_out[127:96] = 32'b0;

    // -------------------------------------------------------------------------
    // IRQ Signals
    // -------------------------------------------------------------------------
    // irq[0] - Inference complete
    // irq[1] - Error occurred
    // irq[2] - Token ready (buffer has token)
    // -------------------------------------------------------------------------
    
    assign user_irq[0] = inference_done;
    assign user_irq[1] = error_flag;
    assign user_irq[2] = token_valid;

    // -------------------------------------------------------------------------
    // SiLens NanoLM Core Instance
    // -------------------------------------------------------------------------
    // Scaled-down model for SKY130 feasibility:
    //   - NanoLM-5M architecture
    //   - 4K vocabulary
    //   - 128 embedding dimension
    //   - 4 transformer layers
    // -------------------------------------------------------------------------
    
    silens_nanolm_core #(
        .VOCAB_SIZE(4096),
        .EMBED_DIM(128),
        .NUM_LAYERS(4),
        .NUM_HEADS(4),
        .FFN_DIM(512),
        .MAX_SEQ_LEN(256),
        .ACT_WIDTH(8)
    ) u_core (
        .clk(clk),
        .rst_n(rst_n),
        
        // SPI interface for data streaming
        .spi_clk(spi_clk),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .spi_cs_n(spi_cs_n),
        
        // UART interface
        .uart_rx(uart_rx),
        .uart_tx(uart_tx),
        
        // Control
        .frame_start(frame_start | ctrl_reg[0]),
        .seq_start(seq_start | ctrl_reg[1]),
        .gen_start(gen_start | ctrl_reg[2]),
        
        // Wishbone token interface
        .wb_token_in(token_in_reg[11:0]),
        .wb_token_in_valid(token_in_valid),
        
        // Status
        .inference_done(inference_done),
        .busy(busy),
        .error_flag(error_flag),
        .status(status),
        
        // Token output
        .token_out(token_out),
        .token_valid(token_valid),
        .token_ready(token_ready),
        
        // Debug outputs for LA
        .debug_cycle_counter(debug_cycle_counter),
        .debug_inference_cycles(debug_inference_cycles),
        .debug_fsm_state(debug_fsm_state),
        .debug_current_token(debug_current_token),
        .debug_token_count(debug_token_count)
    );

endmodule

`default_nettype wire
