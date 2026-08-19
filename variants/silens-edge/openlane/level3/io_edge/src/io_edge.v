// =============================================================================
// SiLens Edge IO Subsystem - Level 3
// =============================================================================
// Minimal IO subsystem for embedded/industrial edge deployment.
// Designed as a "sensor peripheral" for microcontroller hosts.
//
// Target: ~5mm² (2250µm × 2250µm) on SKY130
//
// Interfaces:
//   - SPI Slave (Primary): Image data input, result readback (Mode 0, 50 MHz)
//   - I2C Slave: Configuration registers (100/400 kHz)
//   - GPIO: Trigger input, class output, status signals
//
// SPI Command Protocol:
//   Byte 0: Command [7:6]=OpCode, [5:0]=Address/Length
//     OpCode 00: Write config register
//     OpCode 01: Read config register  
//     OpCode 10: Write image data (burst)
//     OpCode 11: Read inference result
//
// Register Map (I2C/SPI):
//   0x00: CTRL      - [0]=START, [1]=ABORT, [2]=SOFT_RST
//   0x01: STATUS    - [0]=BUSY, [1]=DONE, [2]=ERROR, [7:4]=CLASS
//   0x02: CONFIG0   - Image format, trigger mode
//   0x03: CONFIG1   - Threshold settings
//   0x04: IMG_W_L   - Image width low byte
//   0x05: IMG_W_H   - Image width high byte  
//   0x06: IMG_H_L   - Image height low byte
//   0x07: IMG_H_H   - Image height high byte
//   0x08-0x0F: Reserved
//   0x10: CLASS_OUT - Classification result (0-15)
//   0x11: CONF_OUT  - Confidence score (0-255)
//   0x12: LATENCY_L - Inference latency (low)
//   0x13: LATENCY_H - Inference latency (high)
//   0xFE: VERSION   - Hardware version
//   0xFF: CHIP_ID   - Chip identification
//
// GPIO Pin Assignments:
//   [0]: TRIG_IN    - External trigger input (rising edge)
//   [1]: CLASS_0    - Classification bit 0
//   [2]: CLASS_1    - Classification bit 1
//   [3]: CLASS_2    - Classification bit 2
//   [4]: CLASS_3    - Classification bit 3
//   [5]: BUSY       - Inference in progress
//   [6]: ERROR      - Error indicator
//   [7]: IRQ_N      - Interrupt output (active low)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module io_edge #(
    parameter IMG_FIFO_DEPTH   = 512,    // 512 bytes image input FIFO
    parameter I2C_ADDR         = 7'h50   // Default I2C slave address
)(
    // =========================================================================
    // Clock and Reset
    // =========================================================================
    input  wire         clk,            // Core clock (200 MHz)
    input  wire         rst_n,          // Active-low reset
    
    // =========================================================================
    // SPI Slave Interface (Primary - to MCU host)
    // =========================================================================
    input  wire         spi_clk,        // SPI clock (up to 50 MHz)
    input  wire         spi_mosi,       // Master Out Slave In
    output wire         spi_miso,       // Master In Slave Out
    input  wire         spi_cs_n,       // Chip select (active low)
    
    // =========================================================================
    // I2C Slave Interface (Configuration)
    // =========================================================================
    input  wire         i2c_scl,        // I2C clock
    inout  wire         i2c_sda,        // I2C data (open-drain)
    
    // =========================================================================
    // GPIO Interface
    // =========================================================================
    input  wire [7:0]   gpio_in,        // GPIO input (directly from pads)
    output wire [7:0]   gpio_out,       // GPIO output
    output wire [7:0]   gpio_oe,        // GPIO output enable
    
    // =========================================================================
    // Internal Interface (to inference engine)
    // =========================================================================
    // Image data output (to vision encoder)
    output wire [7:0]   img_data,       // Pixel data (grayscale or packed RGB)
    output wire         img_valid,      // Data valid
    input  wire         img_ready,      // Backpressure from engine
    output wire         img_sof,        // Start of frame
    output wire         img_eof,        // End of frame
    
    // Control signals to inference engine
    output wire         inf_start,      // Start inference
    output wire         inf_abort,      // Abort inference
    
    // Status from inference engine
    input  wire         inf_busy,       // Inference in progress
    input  wire         inf_done,       // Inference complete
    input  wire         inf_error,      // Error occurred
    input  wire [3:0]   inf_class,      // Classification result
    input  wire [7:0]   inf_confidence  // Confidence score
);

    // =========================================================================
    // Constants
    // =========================================================================
    localparam VERSION  = 8'h10;    // v1.0
    localparam CHIP_ID  = 8'hED;    // 'ED' for Edge
    
    // SPI Command opcodes
    localparam OP_WR_REG   = 2'b00;
    localparam OP_RD_REG   = 2'b01;
    localparam OP_WR_IMG   = 2'b10;
    localparam OP_RD_RSLT  = 2'b11;
    
    // Register addresses
    localparam REG_CTRL      = 8'h00;
    localparam REG_STATUS    = 8'h01;
    localparam REG_CONFIG0   = 8'h02;
    localparam REG_CONFIG1   = 8'h03;
    localparam REG_IMG_W_L   = 8'h04;
    localparam REG_IMG_W_H   = 8'h05;
    localparam REG_IMG_H_L   = 8'h06;
    localparam REG_IMG_H_H   = 8'h07;
    localparam REG_CLASS_OUT = 8'h10;
    localparam REG_CONF_OUT  = 8'h11;
    localparam REG_LAT_L     = 8'h12;
    localparam REG_LAT_H     = 8'h13;
    localparam REG_VERSION   = 8'hFE;
    localparam REG_CHIP_ID   = 8'hFF;

    // =========================================================================
    // Internal Registers
    // =========================================================================
    reg [7:0] reg_ctrl;
    reg [7:0] reg_config0;
    reg [7:0] reg_config1;
    reg [15:0] reg_img_width;
    reg [15:0] reg_img_height;
    reg [15:0] latency_counter;
    reg [15:0] latency_captured;
    
    // Control signals derived from registers
    wire ctrl_start    = reg_ctrl[0];
    wire ctrl_abort    = reg_ctrl[1];
    wire ctrl_soft_rst = reg_ctrl[2];
    wire trig_mode     = reg_config0[0];  // 0=SPI trigger, 1=GPIO trigger

    // =========================================================================
    // SPI Slave - Mode 0 (CPOL=0, CPHA=0), up to 50 MHz
    // =========================================================================
    
    // SPI shift register and state
    reg [7:0]  spi_shift_in;
    reg [7:0]  spi_shift_out;
    reg [2:0]  spi_bit_cnt;
    reg [1:0]  spi_byte_cnt;
    reg [1:0]  spi_opcode;
    reg [5:0]  spi_addr;
    reg        spi_miso_reg;
    
    // SPI states
    localparam SPI_CMD     = 2'd0;
    localparam SPI_DATA    = 2'd1;
    localparam SPI_BURST   = 2'd2;
    reg [1:0]  spi_state;
    
    // SPI transaction signals (core clock domain)
    reg        spi_wr_pulse;
    reg        spi_rd_pulse;
    reg [7:0]  spi_wr_addr;
    reg [7:0]  spi_wr_data;
    reg [7:0]  spi_rd_data;
    
    // Image FIFO write from SPI
    reg        img_fifo_wr;
    reg [7:0]  img_fifo_wdata;
    
    // SPI receive - sample on rising edge
    always @(posedge spi_clk or posedge spi_cs_n) begin
        if (spi_cs_n) begin
            spi_shift_in <= 8'h00;
            spi_bit_cnt  <= 3'd0;
            spi_byte_cnt <= 2'd0;
            spi_state    <= SPI_CMD;
            spi_opcode   <= 2'b00;
            spi_addr     <= 6'd0;
        end else begin
            spi_shift_in <= {spi_shift_in[6:0], spi_mosi};
            spi_bit_cnt  <= spi_bit_cnt + 1;
            
            if (spi_bit_cnt == 3'd7) begin
                // Full byte received
                case (spi_state)
                    SPI_CMD: begin
                        spi_opcode <= spi_shift_in[6:5];  // Will get bits 7:6 after shift
                        spi_addr   <= {spi_shift_in[4:0], spi_mosi};
                        spi_byte_cnt <= spi_byte_cnt + 1;
                        spi_state <= (spi_shift_in[6:5] == OP_WR_IMG) ? SPI_BURST : SPI_DATA;
                    end
                    SPI_DATA: begin
                        spi_byte_cnt <= spi_byte_cnt + 1;
                    end
                    SPI_BURST: begin
                        // Stay in burst mode until CS deasserts
                    end
                endcase
            end
        end
    end

    // SPI transmit - change on falling edge
    always @(negedge spi_clk or posedge spi_cs_n) begin
        if (spi_cs_n) begin
            spi_shift_out <= 8'hFF;
            spi_miso_reg  <= 1'b1;
        end else begin
            if (spi_bit_cnt == 3'd0 && spi_state == SPI_DATA) begin
                // Load response data at start of data phase
                spi_shift_out <= spi_rd_data;
                spi_miso_reg  <= spi_rd_data[7];
            end else begin
                spi_shift_out <= {spi_shift_out[6:0], 1'b1};
                spi_miso_reg  <= spi_shift_out[6];
            end
        end
    end
    
    assign spi_miso = spi_cs_n ? 1'bz : spi_miso_reg;
    
    // =========================================================================
    // CDC: SPI domain -> Core clock domain
    // =========================================================================
    
    reg [2:0] cs_sync;
    wire      spi_transaction_done;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cs_sync <= 3'b111;
        else
            cs_sync <= {cs_sync[1:0], spi_cs_n};
    end
    
    assign spi_transaction_done = cs_sync[1] & ~cs_sync[2];  // Rising edge of CS
    
    // Capture SPI transaction on CS deassertion
    reg [7:0]  cap_data;
    reg [1:0]  cap_opcode;
    reg [5:0]  cap_addr;
    reg        cap_valid;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cap_data   <= 8'h00;
            cap_opcode <= 2'b00;
            cap_addr   <= 6'd0;
            cap_valid  <= 1'b0;
        end else begin
            cap_valid <= 1'b0;
            if (spi_transaction_done) begin
                cap_data   <= spi_shift_in;
                cap_opcode <= spi_opcode;
                cap_addr   <= spi_addr;
                cap_valid  <= 1'b1;
            end
        end
    end

    // =========================================================================
    // I2C Slave - Simple implementation (100/400 kHz)
    // =========================================================================
    
    // I2C state machine
    localparam I2C_ST_IDLE      = 3'd0;
    localparam I2C_ST_ADDR      = 3'd1;
    localparam I2C_ST_ADDR_ACK  = 3'd2;
    localparam I2C_ST_REG_ADDR  = 3'd3;
    localparam I2C_ST_REG_ACK   = 3'd4;
    localparam I2C_ST_WR_DATA   = 3'd5;
    localparam I2C_ST_WR_ACK    = 3'd6;
    localparam I2C_ST_RD_DATA   = 3'd7;
    
    reg [2:0]  i2c_state;
    reg [7:0]  i2c_shift;
    reg [2:0]  i2c_bit_cnt;
    reg        i2c_rw;           // 0=write, 1=read
    reg [7:0]  i2c_reg_addr;
    reg        i2c_sda_out;
    reg        i2c_sda_oe;
    
    // Synchronize I2C signals
    reg [2:0]  scl_sync;
    reg [2:0]  sda_sync;
    wire       scl_rise, scl_fall;
    wire       sda_rise, sda_fall;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scl_sync <= 3'b111;
            sda_sync <= 3'b111;
        end else begin
            scl_sync <= {scl_sync[1:0], i2c_scl};
            sda_sync <= {sda_sync[1:0], i2c_sda};
        end
    end
    
    assign scl_rise = scl_sync[1] & ~scl_sync[2];
    assign scl_fall = ~scl_sync[1] & scl_sync[2];
    assign sda_rise = sda_sync[1] & ~sda_sync[2];
    assign sda_fall = ~sda_sync[1] & sda_sync[2];
    
    // START and STOP detection
    wire i2c_start = sda_fall & scl_sync[1];
    wire i2c_stop  = sda_rise & scl_sync[1];
    
    // I2C SDA open-drain output
    assign i2c_sda = i2c_sda_oe ? 1'b0 : 1'bz;

    // I2C register interface
    reg        i2c_wr_pulse;
    reg [7:0]  i2c_wr_data;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i2c_state    <= I2C_ST_IDLE;
            i2c_shift    <= 8'h00;
            i2c_bit_cnt  <= 3'd0;
            i2c_rw       <= 1'b0;
            i2c_reg_addr <= 8'h00;
            i2c_sda_out  <= 1'b1;
            i2c_sda_oe   <= 1'b0;
            i2c_wr_pulse <= 1'b0;
            i2c_wr_data  <= 8'h00;
        end else begin
            i2c_wr_pulse <= 1'b0;
            
            if (i2c_start) begin
                i2c_state   <= I2C_ST_ADDR;
                i2c_bit_cnt <= 3'd0;
                i2c_sda_oe  <= 1'b0;
            end else if (i2c_stop) begin
                i2c_state  <= I2C_ST_IDLE;
                i2c_sda_oe <= 1'b0;
            end else if (scl_rise) begin
                // Sample data on rising edge
                case (i2c_state)
                    I2C_ST_ADDR, I2C_ST_REG_ADDR, I2C_ST_WR_DATA: begin
                        i2c_shift   <= {i2c_shift[6:0], sda_sync[1]};
                        i2c_bit_cnt <= i2c_bit_cnt + 1;
                    end
                endcase
            end else if (scl_fall) begin
                // Change data on falling edge
                case (i2c_state)
                    I2C_ST_ADDR: begin
                        if (i2c_bit_cnt == 3'd0) begin
                            // Check if address matches
                            if (i2c_shift[7:1] == I2C_ST_ADDR) begin
                                i2c_rw     <= i2c_shift[0];
                                i2c_state  <= I2C_ST_ADDR_ACK;
                                i2c_sda_oe <= 1'b1;  // ACK
                            end else begin
                                i2c_state <= I2C_ST_IDLE;  // NACK
                            end
                        end
                    end
                    I2C_ST_ADDR_ACK: begin
                        i2c_sda_oe <= 1'b0;
                        i2c_state  <= i2c_rw ? I2C_ST_RD_DATA : I2C_ST_REG_ADDR;
                        i2c_bit_cnt <= 3'd0;
                        if (i2c_rw) begin
                            // Load read data
                            i2c_shift <= reg_read(i2c_reg_addr);
                        end
                    end
                    I2C_ST_REG_ADDR: begin
                        if (i2c_bit_cnt == 3'd0) begin
                            i2c_reg_addr <= i2c_shift;
                            i2c_state    <= I2C_ST_REG_ACK;
                            i2c_sda_oe   <= 1'b1;  // ACK
                        end
                    end
                    I2C_ST_REG_ACK: begin
                        i2c_sda_oe  <= 1'b0;
                        i2c_state   <= I2C_ST_WR_DATA;
                        i2c_bit_cnt <= 3'd0;
                    end
                    I2C_ST_WR_DATA: begin
                        if (i2c_bit_cnt == 3'd0) begin
                            i2c_wr_data  <= i2c_shift;
                            i2c_wr_pulse <= 1'b1;
                            i2c_state    <= I2C_ST_WR_ACK;
                            i2c_sda_oe   <= 1'b1;  // ACK
                            i2c_reg_addr <= i2c_reg_addr + 1;  // Auto-increment
                        end
                    end
                    I2C_ST_WR_ACK: begin
                        i2c_sda_oe  <= 1'b0;
                        i2c_state   <= I2C_ST_WR_DATA;
                        i2c_bit_cnt <= 3'd0;
                    end
                    I2C_ST_RD_DATA: begin
                        i2c_bit_cnt <= i2c_bit_cnt + 1;
                        if (i2c_bit_cnt == 3'd7) begin
                            i2c_reg_addr <= i2c_reg_addr + 1;  // Auto-increment
                            i2c_shift    <= reg_read(i2c_reg_addr + 1);
                        end
                        i2c_sda_oe <= ~i2c_shift[7];  // Output MSB
                        i2c_shift  <= {i2c_shift[6:0], 1'b1};
                    end
                endcase
            end
        end
    end

    // =========================================================================
    // Register Read Function
    // =========================================================================
    
    function [7:0] reg_read;
        input [7:0] addr;
        begin
            case (addr)
                REG_CTRL:      reg_read = reg_ctrl;
                REG_STATUS:    reg_read = {inf_class, inf_error, inf_done, inf_busy, 1'b0};
                REG_CONFIG0:   reg_read = reg_config0;
                REG_CONFIG1:   reg_read = reg_config1;
                REG_IMG_W_L:   reg_read = reg_img_width[7:0];
                REG_IMG_W_H:   reg_read = reg_img_width[15:8];
                REG_IMG_H_L:   reg_read = reg_img_height[7:0];
                REG_IMG_H_H:   reg_read = reg_img_height[15:8];
                REG_CLASS_OUT: reg_read = {4'b0, inf_class};
                REG_CONF_OUT:  reg_read = inf_confidence;
                REG_LAT_L:     reg_read = latency_captured[7:0];
                REG_LAT_H:     reg_read = latency_captured[15:8];
                REG_VERSION:   reg_read = VERSION;
                REG_CHIP_ID:   reg_read = CHIP_ID;
                default:       reg_read = 8'hFF;
            endcase
        end
    endfunction
    
    // Provide read data for SPI
    always @(*) begin
        spi_rd_data = reg_read({2'b0, cap_addr});
    end
    
    // =========================================================================
    // Register Write Logic
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || ctrl_soft_rst) begin
            reg_ctrl       <= 8'h00;
            reg_config0    <= 8'h00;
            reg_config1    <= 8'h00;
            reg_img_width  <= 16'd224;   // Default: 224x224
            reg_img_height <= 16'd224;
        end else begin
            // Auto-clear start and abort bits
            reg_ctrl[0] <= 1'b0;
            reg_ctrl[1] <= 1'b0;
            
            // SPI register write
            if (cap_valid && cap_opcode == OP_WR_REG) begin
                case ({2'b0, cap_addr})
                    REG_CTRL:    reg_ctrl    <= cap_data;
                    REG_CONFIG0: reg_config0 <= cap_data;
                    REG_CONFIG1: reg_config1 <= cap_data;
                    REG_IMG_W_L: reg_img_width[7:0]   <= cap_data;
                    REG_IMG_W_H: reg_img_width[15:8]  <= cap_data;
                    REG_IMG_H_L: reg_img_height[7:0]  <= cap_data;
                    REG_IMG_H_H: reg_img_height[15:8] <= cap_data;
                endcase
            end
            
            // I2C register write
            if (i2c_wr_pulse) begin
                case (i2c_reg_addr)
                    REG_CTRL:    reg_ctrl    <= i2c_wr_data;
                    REG_CONFIG0: reg_config0 <= i2c_wr_data;
                    REG_CONFIG1: reg_config1 <= i2c_wr_data;
                    REG_IMG_W_L: reg_img_width[7:0]   <= i2c_wr_data;
                    REG_IMG_W_H: reg_img_width[15:8]  <= i2c_wr_data;
                    REG_IMG_H_L: reg_img_height[7:0]  <= i2c_wr_data;
                    REG_IMG_H_H: reg_img_height[15:8] <= i2c_wr_data;
                endcase
            end
        end
    end

    // =========================================================================
    // Image Data FIFO (Simple synchronous FIFO)
    // =========================================================================
    
    reg [7:0]  img_fifo [0:IMG_FIFO_DEPTH-1];
    reg [8:0]  fifo_wr_ptr;
    reg [8:0]  fifo_rd_ptr;
    wire [8:0] fifo_count = fifo_wr_ptr - fifo_rd_ptr;
    wire       fifo_empty = (fifo_count == 0);
    wire       fifo_full  = (fifo_count >= IMG_FIFO_DEPTH);
    
    // Frame tracking
    reg [15:0] pixel_count;
    wire [15:0] total_pixels = reg_img_width * reg_img_height;
    reg        frame_active;
    
    // SPI burst write to FIFO (needs CDC)
    reg [2:0]  spi_byte_done_sync;
    wire       spi_byte_done_pulse;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            spi_byte_done_sync <= 3'b000;
        else
            spi_byte_done_sync <= {spi_byte_done_sync[1:0], (spi_bit_cnt == 3'd7 && spi_state == SPI_BURST)};
    end
    
    assign spi_byte_done_pulse = spi_byte_done_sync[1] & ~spi_byte_done_sync[2];
    
    // FIFO write
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || ctrl_soft_rst) begin
            fifo_wr_ptr <= 9'd0;
        end else if (spi_byte_done_pulse && !fifo_full) begin
            img_fifo[fifo_wr_ptr[8:0]] <= spi_shift_in;
            fifo_wr_ptr <= fifo_wr_ptr + 1;
        end
    end
    
    // FIFO read
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || ctrl_soft_rst) begin
            fifo_rd_ptr <= 9'd0;
        end else if (img_valid && img_ready) begin
            fifo_rd_ptr <= fifo_rd_ptr + 1;
        end
    end
    
    assign img_data  = img_fifo[fifo_rd_ptr[8:0]];
    assign img_valid = !fifo_empty && frame_active;

    // =========================================================================
    // Frame Control & Pixel Counter
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || ctrl_soft_rst) begin
            pixel_count  <= 16'd0;
            frame_active <= 1'b0;
        end else begin
            if (inf_start) begin
                pixel_count  <= 16'd0;
                frame_active <= 1'b1;
            end else if (img_valid && img_ready) begin
                pixel_count <= pixel_count + 1;
                if (pixel_count >= total_pixels - 1) begin
                    frame_active <= 1'b0;
                end
            end
        end
    end
    
    assign img_sof = (pixel_count == 0) && frame_active;
    assign img_eof = (pixel_count >= total_pixels - 1) && img_valid && img_ready;
    
    // =========================================================================
    // Latency Counter
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latency_counter  <= 16'd0;
            latency_captured <= 16'd0;
        end else begin
            if (inf_start) begin
                latency_counter <= 16'd0;
            end else if (inf_busy) begin
                latency_counter <= latency_counter + 1;
            end
            
            if (inf_done) begin
                latency_captured <= latency_counter;
            end
        end
    end
    
    // =========================================================================
    // GPIO Trigger Input (with debounce)
    // =========================================================================
    
    reg [2:0]  trig_sync;
    reg [7:0]  trig_debounce;
    reg        trig_stable;
    reg        trig_prev;
    wire       trig_rising;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            trig_sync     <= 3'b000;
            trig_debounce <= 8'd0;
            trig_stable   <= 1'b0;
            trig_prev     <= 1'b0;
        end else begin
            trig_sync <= {trig_sync[1:0], gpio_in[0]};  // TRIG_IN on GPIO[0]
            trig_prev <= trig_stable;
            
            // Simple debounce: require 8 consecutive same samples
            if (trig_sync[2] == trig_stable) begin
                trig_debounce <= 8'd0;
            end else begin
                trig_debounce <= trig_debounce + 1;
                if (trig_debounce == 8'hFF) begin
                    trig_stable <= trig_sync[2];
                end
            end
        end
    end
    
    assign trig_rising = trig_stable & ~trig_prev;

    // =========================================================================
    // Inference Control
    // =========================================================================
    
    // Start trigger: either from SPI/I2C CTRL register or GPIO trigger (if enabled)
    assign inf_start = ctrl_start | (trig_mode & trig_rising);
    assign inf_abort = ctrl_abort;
    
    // =========================================================================
    // GPIO Output Mapping
    // =========================================================================
    //   [0]: TRIG_IN    - Input only
    //   [1]: CLASS_0    - Output
    //   [2]: CLASS_1    - Output  
    //   [3]: CLASS_2    - Output
    //   [4]: CLASS_3    - Output
    //   [5]: BUSY       - Output
    //   [6]: ERROR      - Output
    //   [7]: IRQ_N      - Output (active low)
    
    assign gpio_out = {
        ~inf_done,      // [7] IRQ_N: active low when inference complete
        inf_error,      // [6] ERROR
        inf_busy,       // [5] BUSY
        inf_class       // [4:1] CLASS[3:0]
    };
    
    assign gpio_oe = 8'b1111_1110;  // All outputs except GPIO[0] (TRIG_IN)
    
endmodule

`default_nettype wire
