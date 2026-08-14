// =============================================================================
// SiLens SPI Slave Interface
// =============================================================================
// SPI slave for configuration and debug access.
// Provides simple register read/write interface for system configuration.
//
// Protocol:
//   - Mode 0 (CPOL=0, CPHA=0)
//   - MSB first
//   - Write: [0][7-bit addr][8-bit data] = 16 bits
//   - Read:  [1][7-bit addr] -> [8-bit data] = 16+8 bits
//
// For SKY130: Low-speed interface for configuration, max ~10 MHz
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_spi_slave (
    // System clock (for register interface)
    input  wire         clk,
    input  wire         rst_n,
    
    // SPI interface
    input  wire         spi_clk,
    input  wire         spi_mosi,
    output wire         spi_miso,
    input  wire         spi_cs_n,
    
    // Register interface (core clock domain)
    output reg  [7:0]   reg_addr,
    output reg  [7:0]   reg_wdata,
    input  wire [7:0]   reg_rdata,
    output reg          reg_wr
);

    // =========================================================================
    // SPI Clock Domain Registers
    // =========================================================================
    
    // Shift register for receiving data
    reg [15:0] shift_in;
    reg [4:0]  bit_cnt;
    
    // Shift register for transmitting data
    reg [7:0]  shift_out;
    
    // Command decoded
    reg        cmd_write;
    reg [6:0]  cmd_addr;
    
    // State machine
    localparam ST_IDLE    = 2'd0;
    localparam ST_COMMAND = 2'd1;
    localparam ST_DATA    = 2'd2;
    localparam ST_RESPOND = 2'd3;
    
    reg [1:0] state;
    
    // =========================================================================
    // SPI Receive (MOSI) - Sample on rising edge of SPI_CLK
    // =========================================================================
    
    always @(posedge spi_clk or posedge spi_cs_n) begin
        if (spi_cs_n) begin
            // Reset on CS deassert
            shift_in <= 16'b0;
            bit_cnt <= 0;
            state <= ST_IDLE;
            cmd_write <= 0;
            cmd_addr <= 0;
        end else begin
            // Shift in MOSI data
            shift_in <= {shift_in[14:0], spi_mosi};
            bit_cnt <= bit_cnt + 1;
            
            case (state)
                ST_IDLE: begin
                    if (bit_cnt == 0) begin
                        state <= ST_COMMAND;
                    end
                end
                
                ST_COMMAND: begin
                    if (bit_cnt == 7) begin
                        // Command byte complete: [R/W][6:0 addr]
                        cmd_write <= ~shift_in[6];  // Bit 7 will be R/W (0=write, 1=read)
                        cmd_addr <= {shift_in[5:0], spi_mosi};
                        state <= ST_DATA;
                    end
                end
                
                ST_DATA: begin
                    if (bit_cnt == 15) begin
                        // Data byte complete (for write) or start response (for read)
                        state <= ST_RESPOND;
                    end
                end
                
                ST_RESPOND: begin
                    // Sending response for read, or waiting for CS deassert
                end
            endcase
        end
    end
    
    // =========================================================================
    // SPI Transmit (MISO) - Change on falling edge of SPI_CLK
    // =========================================================================
    
    reg [7:0] miso_shift;
    reg miso_out;
    
    always @(negedge spi_clk or posedge spi_cs_n) begin
        if (spi_cs_n) begin
            miso_shift <= 8'hFF;
            miso_out <= 1'b1;
        end else begin
            if (bit_cnt == 8 && !cmd_write) begin
                // Load read data at start of response phase
                miso_shift <= reg_rdata;
                miso_out <= reg_rdata[7];
            end else if (state == ST_DATA || state == ST_RESPOND) begin
                // Shift out response
                miso_shift <= {miso_shift[6:0], 1'b1};
                miso_out <= miso_shift[6];
            end
        end
    end
    
    assign spi_miso = spi_cs_n ? 1'bz : miso_out;
    
    // =========================================================================
    // CDC: SPI domain -> Core clock domain
    // =========================================================================
    
    // Synchronize CS deassertion as transaction complete signal
    reg [2:0] cs_sync;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cs_sync <= 3'b111;
        end else begin
            cs_sync <= {cs_sync[1:0], spi_cs_n};
        end
    end
    
    wire transaction_done = cs_sync[1] & ~cs_sync[2];  // Rising edge of CS
    
    // Capture transaction data on CS rising edge
    reg [15:0] captured_data;
    reg        captured_write;
    reg [6:0]  captured_addr;
    reg        capture_valid;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            captured_data <= 0;
            captured_write <= 0;
            captured_addr <= 0;
            capture_valid <= 0;
        end else begin
            capture_valid <= 0;
            
            if (transaction_done) begin
                captured_data <= shift_in;
                captured_write <= cmd_write;
                captured_addr <= cmd_addr;
                capture_valid <= 1;
            end
        end
    end
    
    // =========================================================================
    // Register Interface
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_addr <= 0;
            reg_wdata <= 0;
            reg_wr <= 0;
        end else begin
            reg_wr <= 0;
            
            if (capture_valid) begin
                reg_addr <= {1'b0, captured_addr};
                
                if (captured_write) begin
                    // Write operation
                    reg_wdata <= captured_data[7:0];
                    reg_wr <= 1;
                end
                // Read operation: reg_addr is set, reg_rdata will be sampled by SPI
            end
        end
    end

endmodule

`default_nettype wire
