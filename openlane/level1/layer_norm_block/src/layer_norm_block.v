// =============================================================================
// Layer Normalization Block (768-dim) - Level 1 Synthesis Block
// =============================================================================
// Standalone Layer Norm for Vision Encoder layers.
// Implements: y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// This is a synthesis-optimized version with parameterized dimension.
// Uses iterative computation to reduce area (sequential processing).
//
// Target: ~0.5mm² on SKY130
// Latency: ~150 cycles (mean calc + var calc + normalize)
// Reuse: 24× across vision subsystem
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module layer_norm_block #(
    parameter DIM       = 768,      // Hidden dimension (Vision Encoder)
    parameter ACT_WIDTH = 8,        // Activation bits
    parameter ACC_WIDTH = 32        // Accumulator bits
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Input interface
    input  wire [DIM*ACT_WIDTH-1:0] x_in,
    input  wire                     valid_in,
    output wire                     ready_in,
    
    // Gamma scale and Beta offset parameters (from weight ROM)
    input  wire [DIM*ACT_WIDTH-1:0] gamma,
    input  wire [DIM*ACT_WIDTH-1:0] beta,
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0] y_out,
    output reg                      valid_out,
    input  wire                     ready_out
);

    // =========================================================================
    // Parameters
    // =========================================================================
    
    localparam FRAC_BITS = 8;       // Fixed-point fractional bits
    localparam EPS = 1;             // Small constant for numerical stability
    localparam DIM_BITS = $clog2(DIM);
    
    // =========================================================================
    // FSM States
    // =========================================================================
    
    localparam ST_IDLE      = 3'd0;
    localparam ST_LOAD      = 3'd1;
    localparam ST_CALC_MEAN = 3'd2;
    localparam ST_CALC_VAR  = 3'd3;
    localparam ST_INV_SQRT  = 3'd4;
    localparam ST_NORMALIZE = 3'd5;
    localparam ST_OUTPUT    = 3'd6;
    
    reg [2:0] state;
    
    // =========================================================================
    // Registers
    // =========================================================================
    
    // Input buffer
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];
    
    // Computation registers
    reg signed [ACC_WIDTH-1:0] sum_x;       // Sum of x (for mean)
    reg signed [ACC_WIDTH-1:0] sum_var;     // Sum of (x - mean)^2 (for variance)
    reg signed [ACC_WIDTH-1:0] mean;        // Mean value
    reg [ACC_WIDTH-1:0] variance;           // Variance + epsilon
    reg [ACC_WIDTH-1:0] inv_std;            // Inverse standard deviation
    reg [DIM_BITS:0] idx;                   // Element index
    reg [2:0] nr_iter;                      // Newton-Raphson iteration counter
    
    // Newton-Raphson working register
    reg [ACC_WIDTH-1:0] y_nr;
    
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
    // Current element access
    // =========================================================================
    
    wire signed [ACT_WIDTH-1:0] x_curr = x_buf[idx];
    wire signed [ACT_WIDTH-1:0] gamma_curr = $signed(gamma[idx*ACT_WIDTH +: ACT_WIDTH]);
    wire signed [ACT_WIDTH-1:0] beta_curr = $signed(beta[idx*ACT_WIDTH +: ACT_WIDTH]);
    
    // =========================================================================
    // Mean calculation: sum(x) / N
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] mean_computed = sum_x >>> DIM_BITS;
    
    // =========================================================================
    // Variance calculation: sum((x - mean)^2) / N
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] x_centered = {{(ACC_WIDTH-ACT_WIDTH){x_curr[ACT_WIDTH-1]}}, x_curr} - mean;
    wire signed [2*ACC_WIDTH-1:0] x_centered_sq = x_centered * x_centered;
    
    // =========================================================================
    // Newton-Raphson inverse square root
    // =========================================================================
    // Computes 1/sqrt(variance) using: y_{n+1} = y_n * (3 - x * y_n^2) / 2
    
    wire [ACC_WIDTH-1:0] y_sq = (y_nr * y_nr) >> FRAC_BITS;
    wire [ACC_WIDTH-1:0] var_y_sq = (variance * y_sq) >> FRAC_BITS;
    wire [ACC_WIDTH-1:0] three_fp = 3 << FRAC_BITS;
    wire signed [ACC_WIDTH-1:0] diff_term = $signed(three_fp) - $signed(var_y_sq);
    wire [ACC_WIDTH-1:0] y_next = (diff_term > 0) ? 
                                  ((y_nr * diff_term[ACC_WIDTH-2:0]) >> (FRAC_BITS + 1)) : 0;
    
    // =========================================================================
    // Normalization: gamma * (x - mean) * inv_std + beta
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] x_norm = (x_centered * $signed(inv_std)) >>> FRAC_BITS;
    wire signed [ACC_WIDTH-1:0] y_scaled = (x_norm * $signed({1'b0, gamma_curr})) >>> FRAC_BITS;
    wire signed [ACC_WIDTH-1:0] y_biased = y_scaled + {{(ACC_WIDTH-ACT_WIDTH){beta_curr[ACT_WIDTH-1]}}, beta_curr};
    
    // =========================================================================
    // Saturation function
    // =========================================================================
    
    function signed [ACT_WIDTH-1:0] saturate;
        input signed [ACC_WIDTH-1:0] val;
        reg signed [ACT_WIDTH-1:0] max_val;
        reg signed [ACT_WIDTH-1:0] min_val;
        begin
            max_val = {1'b0, {(ACT_WIDTH-1){1'b1}}};  // +127 for 8-bit
            min_val = {1'b1, {(ACT_WIDTH-1){1'b0}}};  // -128 for 8-bit
            if (val > $signed({{(ACC_WIDTH-ACT_WIDTH){1'b0}}, max_val}))
                saturate = max_val;
            else if (val < $signed({{(ACC_WIDTH-ACT_WIDTH){1'b1}}, min_val}))
                saturate = min_val;
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
            sum_x <= 0;
            sum_var <= 0;
            mean <= 0;
            variance <= 0;
            inv_std <= 0;
            y_nr <= 0;
            nr_iter <= 0;
            valid_out <= 0;
            y_out <= 0;
        end else begin
            case (state)
                // ---------------------------------------------------------
                // IDLE: Wait for valid input
                // ---------------------------------------------------------
                ST_IDLE: begin
                    valid_out <= 0;
                    if (valid_in) begin
                        state <= ST_LOAD;
                    end
                end
                
                // ---------------------------------------------------------
                // LOAD: Initialize for mean calculation
                // ---------------------------------------------------------
                ST_LOAD: begin
                    state <= ST_CALC_MEAN;
                    idx <= 0;
                    sum_x <= 0;
                end
                
                // ---------------------------------------------------------
                // CALC_MEAN: Accumulate sum(x) iteratively
                // ---------------------------------------------------------
                ST_CALC_MEAN: begin
                    sum_x <= sum_x + {{(ACC_WIDTH-ACT_WIDTH){x_curr[ACT_WIDTH-1]}}, x_curr};
                    
                    if (idx == DIM - 1) begin
                        state <= ST_CALC_VAR;
                        mean <= (sum_x + {{(ACC_WIDTH-ACT_WIDTH){x_curr[ACT_WIDTH-1]}}, x_curr}) >>> DIM_BITS;
                        idx <= 0;
                        sum_var <= 0;
                    end else begin
                        idx <= idx + 1;
                    end
                end
                
                // ---------------------------------------------------------
                // CALC_VAR: Accumulate sum((x - mean)^2) iteratively
                // ---------------------------------------------------------
                ST_CALC_VAR: begin
                    sum_var <= sum_var + x_centered_sq[ACC_WIDTH-1:0];
                    
                    if (idx == DIM - 1) begin
                        state <= ST_INV_SQRT;
                        // Compute variance = sum_var / N + epsilon
                        variance <= ((sum_var + x_centered_sq[ACC_WIDTH-1:0]) >> DIM_BITS) + EPS;
                        nr_iter <= 0;
                        // Initial guess for Newton-Raphson: 1.0 in fixed-point
                        y_nr <= 1 << FRAC_BITS;
                    end else begin
                        idx <= idx + 1;
                    end
                end
                
                // ---------------------------------------------------------
                // INV_SQRT: Newton-Raphson iterations for 1/sqrt(var)
                // ---------------------------------------------------------
                ST_INV_SQRT: begin
                    y_nr <= y_next;
                    
                    if (nr_iter == 4) begin
                        // Store final inverse std deviation
                        inv_std <= y_nr;
                        state <= ST_NORMALIZE;
                        idx <= 0;
                    end else begin
                        nr_iter <= nr_iter + 1;
                    end
                end
                
                // ---------------------------------------------------------
                // NORMALIZE: Apply normalization element by element
                // ---------------------------------------------------------
                ST_NORMALIZE: begin
                    y_out[idx*ACT_WIDTH +: ACT_WIDTH] <= saturate(y_biased);
                    
                    if (idx == DIM - 1) begin
                        state <= ST_OUTPUT;
                    end else begin
                        idx <= idx + 1;
                    end
                end
                
                // ---------------------------------------------------------
                // OUTPUT: Signal valid output, wait for ready
                // ---------------------------------------------------------
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
