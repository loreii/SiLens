// =============================================================================
// SiLU Activation Unit - Level 1 Synthesis Block
// =============================================================================
// Gated MLP activation function: SiLU(x) = x * sigmoid(x)
// Also known as Swish activation.
//
// Hardware approximation using piecewise linear sigmoid:
//   sigmoid(x) ≈ PWL approximation with 8 segments
//   
// Piecewise linear regions (symmetric around 0):
//   x < -4.0:  sigmoid ≈ 0.0
//   x > +4.0:  sigmoid ≈ 1.0
//   Otherwise: linear interpolation between breakpoints
//
// Target: ~0.3mm² on SKY130 (550µm × 550µm)
// Reuse: 42× (30 in LLM MLP, 12 in vision MLP)
// Throughput: PARALLEL elements per cycle (pipelined)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silu_unit #(
    parameter WIDTH    = 8,         // Element bit width (signed fixed-point Q4.4)
    parameter PARALLEL = 16,        // Elements processed per cycle
    parameter FRAC     = 4          // Fractional bits for fixed-point
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface (AXI-Stream style)
    input  wire [PARALLEL*WIDTH-1:0]    x_in,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Output interface (AXI-Stream style)
    output reg  [PARALLEL*WIDTH-1:0]    y_out,
    output reg                          valid_out,
    input  wire                         ready_out
);

    // =========================================================================
    // Fixed-point constants (Q4.4 format)
    // =========================================================================
    // 1.0 = 16 (1 << FRAC)
    // 0.5 = 8
    // Breakpoints at x = -4, -2, -1, 0, 1, 2, 4
    
    localparam signed [WIDTH-1:0] ONE_FP   = (1 << FRAC);           // 1.0
    localparam signed [WIDTH-1:0] HALF_FP  = (1 << (FRAC - 1));     // 0.5
    
    // Breakpoints in fixed-point (scaled by 2^FRAC)
    localparam signed [WIDTH-1:0] BP_N4 = -4 * ONE_FP;   // -64 in Q4.4 (saturated to -128)
    localparam signed [WIDTH-1:0] BP_N2 = -2 * ONE_FP;   // -32 in Q4.4
    localparam signed [WIDTH-1:0] BP_N1 = -1 * ONE_FP;   // -16 in Q4.4
    localparam signed [WIDTH-1:0] BP_P1 =  1 * ONE_FP;   // +16 in Q4.4
    localparam signed [WIDTH-1:0] BP_P2 =  2 * ONE_FP;   // +32 in Q4.4
    localparam signed [WIDTH-1:0] BP_P4 =  4 * ONE_FP;   // +64 in Q4.4 (saturated to +127)
    
    // Sigmoid values at breakpoints (in Q0.8 format, scaled to Q4.4)
    // sigmoid(-4) ≈ 0.018, sigmoid(-2) ≈ 0.119, sigmoid(-1) ≈ 0.269
    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(2) ≈ 0.881, sigmoid(4) ≈ 0.982
    localparam signed [WIDTH-1:0] SIG_N4 = 0;    // ≈ 0.0
    localparam signed [WIDTH-1:0] SIG_N2 = 2;    // ≈ 0.125
    localparam signed [WIDTH-1:0] SIG_N1 = 4;    // ≈ 0.25
    localparam signed [WIDTH-1:0] SIG_0  = 8;    // = 0.5
    localparam signed [WIDTH-1:0] SIG_P1 = 12;   // ≈ 0.75
    localparam signed [WIDTH-1:0] SIG_P2 = 14;   // ≈ 0.875
    localparam signed [WIDTH-1:0] SIG_P4 = 16;   // ≈ 1.0
    
    // =========================================================================
    // Pipeline registers
    // =========================================================================
    
    // Stage 1: Input register & sigmoid computation
    reg signed [WIDTH-1:0]       x_s1     [0:PARALLEL-1];
    reg signed [WIDTH-1:0]       sig_s1   [0:PARALLEL-1];
    reg                          valid_s1;
    
    // Stage 2: Multiply x * sigmoid(x)
    reg signed [2*WIDTH-1:0]     prod_s2  [0:PARALLEL-1];
    reg                          valid_s2;
    
    // =========================================================================
    // Flow control
    // =========================================================================
    
    wire pipe_stall = valid_out && !ready_out;
    assign ready_in = !pipe_stall;
    
    // =========================================================================
    // Piecewise linear sigmoid function
    // =========================================================================
    // Using a function for synthesis efficiency
    
    function signed [WIDTH-1:0] pwl_sigmoid;
        input signed [WIDTH-1:0] x;
        reg signed [WIDTH-1:0] result;
        reg signed [2*WIDTH-1:0] interp;
        begin
            // Clamp to representable range
            if (x <= BP_N4 || x < -8'sd64) begin
                // x <= -4: sigmoid ≈ 0
                result = SIG_N4;
            end else if (x < BP_N2) begin
                // -4 < x < -2: linear from 0 to 0.125
                // slope = (SIG_N2 - SIG_N4) / (BP_N2 - BP_N4) = 2/32 = 1/16
                interp = (x - BP_N4) >>> 4;
                result = SIG_N4 + interp[WIDTH-1:0];
            end else if (x < BP_N1) begin
                // -2 < x < -1: linear from 0.125 to 0.25
                // slope = (SIG_N1 - SIG_N2) / (BP_N1 - BP_N2) = 2/16 = 1/8
                interp = (x - BP_N2) >>> 3;
                result = SIG_N2 + interp[WIDTH-1:0];
            end else if (x < 0) begin
                // -1 < x < 0: linear from 0.25 to 0.5
                // slope = (SIG_0 - SIG_N1) / (0 - BP_N1) = 4/16 = 1/4
                interp = (x - BP_N1) >>> 2;
                result = SIG_N1 + interp[WIDTH-1:0];
            end else if (x < BP_P1) begin
                // 0 <= x < 1: linear from 0.5 to 0.75
                // slope = (SIG_P1 - SIG_0) / (BP_P1 - 0) = 4/16 = 1/4
                interp = x >>> 2;
                result = SIG_0 + interp[WIDTH-1:0];
            end else if (x < BP_P2) begin
                // 1 <= x < 2: linear from 0.75 to 0.875
                // slope = (SIG_P2 - SIG_P1) / (BP_P2 - BP_P1) = 2/16 = 1/8
                interp = (x - BP_P1) >>> 3;
                result = SIG_P1 + interp[WIDTH-1:0];
            end else if (x < BP_P4 && x <= 8'sd63) begin
                // 2 <= x < 4: linear from 0.875 to 1.0
                // slope = (SIG_P4 - SIG_P2) / (BP_P4 - BP_P2) = 2/32 = 1/16
                interp = (x - BP_P2) >>> 4;
                result = SIG_P2 + interp[WIDTH-1:0];
            end else begin
                // x >= 4: sigmoid ≈ 1
                result = SIG_P4;
            end
            pwl_sigmoid = result;
        end
    endfunction
    
    // =========================================================================
    // Saturation function for output
    // =========================================================================
    
    function signed [WIDTH-1:0] saturate;
        input signed [2*WIDTH-1:0] val;
        localparam signed [2*WIDTH-1:0] MAX_VAL = (1 << (WIDTH-1)) - 1;
        localparam signed [2*WIDTH-1:0] MIN_VAL = -(1 << (WIDTH-1));
        begin
            if (val > MAX_VAL)
                saturate = MAX_VAL[WIDTH-1:0];
            else if (val < MIN_VAL)
                saturate = MIN_VAL[WIDTH-1:0];
            else
                saturate = val[WIDTH-1:0];
        end
    endfunction
    
    // =========================================================================
    // Pipeline Stage 1: Compute sigmoid(x) for all parallel elements
    // =========================================================================
    
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1 <= 1'b0;
            for (i = 0; i < PARALLEL; i = i + 1) begin
                x_s1[i]   <= {WIDTH{1'b0}};
                sig_s1[i] <= {WIDTH{1'b0}};
            end
        end else if (!pipe_stall) begin
            valid_s1 <= valid_in;
            for (i = 0; i < PARALLEL; i = i + 1) begin
                x_s1[i]   <= $signed(x_in[i*WIDTH +: WIDTH]);
                sig_s1[i] <= pwl_sigmoid($signed(x_in[i*WIDTH +: WIDTH]));
            end
        end
    end
    
    // =========================================================================
    // Pipeline Stage 2: Multiply x * sigmoid(x)
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2 <= 1'b0;
            for (i = 0; i < PARALLEL; i = i + 1) begin
                prod_s2[i] <= {(2*WIDTH){1'b0}};
            end
        end else if (!pipe_stall) begin
            valid_s2 <= valid_s1;
            for (i = 0; i < PARALLEL; i = i + 1) begin
                // x * sigmoid(x), result is in Q8.8 format
                prod_s2[i] <= x_s1[i] * sig_s1[i];
            end
        end
    end
    
    // =========================================================================
    // Pipeline Stage 3: Output with saturation
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            y_out     <= {(PARALLEL*WIDTH){1'b0}};
        end else if (!pipe_stall) begin
            valid_out <= valid_s2;
            for (i = 0; i < PARALLEL; i = i + 1) begin
                // Shift right by FRAC to normalize back to Q4.4, then saturate
                y_out[i*WIDTH +: WIDTH] <= saturate(prod_s2[i] >>> FRAC);
            end
        end
    end

endmodule

`default_nettype wire
