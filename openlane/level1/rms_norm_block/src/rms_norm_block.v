// =============================================================================
// RMS Normalization Block (576-dim) - Level 1 Synthesis Block
// =============================================================================
// Standalone RMS norm for LLM layers.
// Implements: y = x * rsqrt(mean(x^2) + eps) * gamma
//
// This is a synthesis-optimized version with hardcoded dimension.
// Uses iterative computation to reduce area.
//
// Target: ~0.5mm² on SKY130
// Latency: ~100 cycles
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module rms_norm_block #(
    parameter DIM       = 576,      // Hidden dimension (LLM)
    parameter ACT_WIDTH = 8,        // Activation bits
    parameter ACC_WIDTH = 32,       // Accumulator bits
    parameter FRAC_BITS = 8         // Fixed-point fractional bits
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Input interface
    input  wire [DIM*ACT_WIDTH-1:0] x_in,
    input  wire                     valid_in,
    output wire                     ready_in,
    
    // Gamma scale parameter (from weight ROM)
    input  wire [DIM*ACT_WIDTH-1:0] gamma,
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0] y_out,
    output reg                      valid_out,
    input  wire                     ready_out
);

    // =========================================================================
    // Parameters
    // =========================================================================
    
    localparam EPS = 1;  // Small constant for numerical stability
    localparam DIM_BITS = $clog2(DIM);
    
    // =========================================================================
    // FSM
    // =========================================================================
    
    localparam ST_IDLE      = 3'd0;
    localparam ST_LOAD      = 3'd1;
    localparam ST_SUM_SQ    = 3'd2;
    localparam ST_INV_SQRT  = 3'd3;
    localparam ST_NORMALIZE = 3'd4;
    localparam ST_OUTPUT    = 3'd5;
    
    reg [2:0] state;
    
    // =========================================================================
    // Registers
    // =========================================================================
    
    // Input buffer (double-buffered for throughput if needed)
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];
    
    // Computation registers
    reg [ACC_WIDTH-1:0] sum_sq;         // Sum of squares
    reg [ACC_WIDTH-1:0] inv_rms;        // Inverse RMS
    reg [DIM_BITS:0] idx;               // Element index
    reg [2:0] nr_iter;                  // Newton-Raphson iteration
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign ready_in = (state == ST_IDLE);
    
    // =========================================================================
    // Input loading
    // =========================================================================
    
    integer i;
    always @(posedge clk) begin
        if (state == ST_IDLE && valid_in) begin
            for (i = 0; i < DIM; i = i + 1) begin
                x_buf[i] <= $signed(x_in[i*ACT_WIDTH +: ACT_WIDTH]);
            end
        end
    end
    
    // =========================================================================
    // Sum of squares computation (iterative)
    // =========================================================================
    
    wire signed [ACT_WIDTH-1:0] x_curr = x_buf[idx];
    wire signed [2*ACT_WIDTH-1:0] x_sq = x_curr * x_curr;
    
    // =========================================================================
    // Newton-Raphson inverse square root
    // =========================================================================
    // y_{n+1} = y_n * (3 - x * y_n^2) / 2
    
    reg [ACC_WIDTH-1:0] y_nr;
    wire [ACC_WIDTH-1:0] y_sq = (y_nr * y_nr) >> FRAC_BITS;
    wire [ACC_WIDTH-1:0] x_y_sq = (sum_sq * y_sq) >> FRAC_BITS;
    wire [ACC_WIDTH-1:0] three_fp = 3 << FRAC_BITS;
    wire [ACC_WIDTH-1:0] y_next = (y_nr * (three_fp - x_y_sq)) >> (FRAC_BITS + 1);
    
    // =========================================================================
    // Normalization
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] x_scaled;
    wire signed [ACT_WIDTH-1:0] gamma_curr = $signed(gamma[idx*ACT_WIDTH +: ACT_WIDTH]);
    
    assign x_scaled = ($signed({x_curr[ACT_WIDTH-1], x_curr}) * $signed(inv_rms)) >>> FRAC_BITS;
    
    // Apply gamma and saturate
    wire signed [ACC_WIDTH-1:0] y_gamma = (x_scaled * $signed({1'b0, gamma_curr})) >>> FRAC_BITS;
    
    function signed [ACT_WIDTH-1:0] saturate;
        input signed [ACC_WIDTH-1:0] val;
        localparam MAX_VAL = (1 << (ACT_WIDTH-1)) - 1;
        localparam MIN_VAL = -(1 << (ACT_WIDTH-1));
        begin
            if (val > MAX_VAL)
                saturate = MAX_VAL;
            else if (val < MIN_VAL)
                saturate = MIN_VAL;
            else
                saturate = val[ACT_WIDTH-1:0];
        end
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            idx <= 0;
            sum_sq <= 0;
            inv_rms <= 0;
            y_nr <= 0;
            nr_iter <= 0;
            valid_out <= 0;
            y_out <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    valid_out <= 0;
                    if (valid_in) begin
                        state <= ST_LOAD;
                    end
                end
                
                ST_LOAD: begin
                    // Input loaded in combinational block above
                    state <= ST_SUM_SQ;
                    idx <= 0;
                    sum_sq <= 0;
                end
                
                ST_SUM_SQ: begin
                    // Accumulate x^2
                    sum_sq <= sum_sq + x_sq;
                    
                    if (idx == DIM - 1) begin
                        state <= ST_INV_SQRT;
                        nr_iter <= 0;
                        // Initial guess: 1.0 in fixed-point
                        y_nr <= 1 << FRAC_BITS;
                        // Add epsilon and compute mean
                        sum_sq <= ((sum_sq + x_sq) >> DIM_BITS) + EPS;
                    end else begin
                        idx <= idx + 1;
                    end
                end
                
                ST_INV_SQRT: begin
                    // Newton-Raphson iterations
                    y_nr <= y_next;
                    
                    if (nr_iter == 3) begin
                        inv_rms <= y_nr;
                        state <= ST_NORMALIZE;
                        idx <= 0;
                    end else begin
                        nr_iter <= nr_iter + 1;
                    end
                end
                
                ST_NORMALIZE: begin
                    // Apply normalization element by element
                    y_out[idx*ACT_WIDTH +: ACT_WIDTH] <= saturate(y_gamma);
                    
                    if (idx == DIM - 1) begin
                        state <= ST_OUTPUT;
                    end else begin
                        idx <= idx + 1;
                    end
                end
                
                ST_OUTPUT: begin
                    valid_out <= 1;
                    if (ready_out) begin
                        valid_out <= 0;
                        state <= ST_IDLE;
                    end
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
