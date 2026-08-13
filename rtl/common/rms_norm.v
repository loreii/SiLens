// =============================================================================
// SiLens - RMS Normalization Module
// =============================================================================
// Implements Root Mean Square Layer Normalization.
//
// RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
//
// Unlike LayerNorm, RMSNorm does not center the input (no mean subtraction).
// This is used in LLaMA-style models like SmolLM2.
//
// License: Apache 2.0
// =============================================================================

module rms_norm #(
    parameter DIM         = 576,                    // Feature dimension
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits
    parameter PARALLEL    = 16,                     // Elements per cycle
    parameter EPS         = 1                       // Epsilon (fixed-point)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Learnable scale parameter
    input  wire [DIM*ACT_WIDTH-1:0]     gamma,
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0]     y_out,
    output reg                          valid_out,
    input  wire                         ready_out
);

    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE       = 2'd0;
    localparam STATE_COMPUTE_MS = 2'd1;  // Compute mean square
    localparam STATE_NORMALIZE  = 2'd2;  // Normalize and scale
    localparam STATE_DONE       = 2'd3;
    
    reg [1:0] state;
    
    // =========================================================================
    // Input buffer
    // =========================================================================
    
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];
    
    // =========================================================================
    // Statistics
    // =========================================================================
    
    reg signed [ACC_WIDTH-1:0] sum_sq;       // Sum of squares
    reg signed [ACC_WIDTH-1:0] mean_sq;      // Mean of squares
    reg signed [ACC_WIDTH-1:0] inv_rms;      // 1/sqrt(mean_sq + eps)
    
    // Newton-Raphson iteration counter
    reg [2:0] nr_iter;
    localparam NR_ITERATIONS = 3;
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    // =========================================================================
    // Input loading
    // =========================================================================
    
    integer load_i;
    always @(posedge clk) begin
        if (state == STATE_IDLE && valid_in) begin
            for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                x_buf[load_i] <= $signed(x_in[load_i*ACT_WIDTH +: ACT_WIDTH]);
            end
        end
    end
    
    // =========================================================================
    // Inverse square root using Newton-Raphson
    // =========================================================================
    // y = 1/sqrt(x)
    // Newton-Raphson: y_{n+1} = y_n * (3 - x * y_n^2) / 2
    
    function signed [ACC_WIDTH-1:0] newton_raphson_step;
        input signed [ACC_WIDTH-1:0] x;
        input signed [ACC_WIDTH-1:0] y;
        reg signed [ACC_WIDTH-1:0] y_sq;
        reg signed [ACC_WIDTH-1:0] x_y_sq;
        reg signed [ACC_WIDTH-1:0] three;
        begin
            three = 3 << FRAC_BITS;
            y_sq = (y * y) >>> FRAC_BITS;
            x_y_sq = (x * y_sq) >>> FRAC_BITS;
            newton_raphson_step = (y * (three - x_y_sq)) >>> (FRAC_BITS + 1);
        end
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer proc_i;
    reg signed [ACC_WIDTH-1:0] sq_sum;
    reg signed [ACC_WIDTH-1:0] y_nr;
    reg signed [ACC_WIDTH-1:0] norm_result;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            sum_sq <= 0;
            mean_sq <= 0;
            inv_rms <= 0;
            nr_iter <= 0;
            valid_out <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        state <= STATE_COMPUTE_MS;
                        nr_iter <= 0;
                    end
                end
                
                STATE_COMPUTE_MS: begin
                    // Compute sum of squares
                    sq_sum = 0;
                    for (proc_i = 0; proc_i < DIM; proc_i = proc_i + 1) begin
                        sq_sum = sq_sum + ($signed(x_buf[proc_i]) * $signed(x_buf[proc_i]));
                    end
                    
                    // Compute mean: sum / DIM
                    mean_sq <= (sq_sum / DIM) + EPS;
                    
                    // Initialize Newton-Raphson with y = 1.0
                    y_nr <= 1 << FRAC_BITS;
                    
                    state <= STATE_NORMALIZE;
                end
                
                STATE_NORMALIZE: begin
                    // Newton-Raphson iteration for 1/sqrt(mean_sq)
                    if (nr_iter < NR_ITERATIONS) begin
                        y_nr <= newton_raphson_step(mean_sq, y_nr);
                        nr_iter <= nr_iter + 1;
                    end else begin
                        // Apply normalization: y = x * inv_rms * gamma
                        inv_rms <= y_nr;
                        
                        for (proc_i = 0; proc_i < DIM; proc_i = proc_i + 1) begin
                            norm_result = ($signed(x_buf[proc_i]) * y_nr) >>> FRAC_BITS;
                            norm_result = (norm_result * $signed(gamma[proc_i*ACT_WIDTH +: ACT_WIDTH])) >>> FRAC_BITS;
                            y_out[proc_i*ACT_WIDTH +: ACT_WIDTH] <= saturate(norm_result);
                        end
                        
                        state <= STATE_DONE;
                    end
                end
                
                STATE_DONE: begin
                    valid_out <= 1'b1;
                    if (ready_out) begin
                        valid_out <= 1'b0;
                        state <= STATE_IDLE;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Saturation function
    // =========================================================================
    
    function signed [ACT_WIDTH-1:0] saturate;
        input signed [ACC_WIDTH-1:0] val;
        begin
            if (val > $signed({{(ACC_WIDTH-ACT_WIDTH+1){1'b0}}, {(ACT_WIDTH-1){1'b1}}}))
                saturate = {1'b0, {(ACT_WIDTH-1){1'b1}}};
            else if (val < $signed({{(ACC_WIDTH-ACT_WIDTH+1){1'b1}}, {(ACT_WIDTH-1){1'b0}}}))
                saturate = {1'b1, {(ACT_WIDTH-1){1'b0}}};
            else
                saturate = val[ACT_WIDTH-1:0];
        end
    endfunction

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module rms_norm_tb;
    parameter DIM = 16;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 4;
    
    reg clk, rst_n;
    reg [DIM*ACT_WIDTH-1:0] x_in;
    reg valid_in;
    wire ready_in;
    reg [DIM*ACT_WIDTH-1:0] gamma;
    wire [DIM*ACT_WIDTH-1:0] y_out;
    wire valid_out;
    reg ready_out;
    
    rms_norm #(
        .DIM(DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i;
    
    initial begin
        $display("RMS Norm Testbench");
        $display("==================");
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        valid_in = 0;
        ready_out = 1;
        gamma = {DIM{8'd16}};  // 1.0 in Q4.4
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Test with simple input
        for (i = 0; i < DIM; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = 8'd16;  // 1.0
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        repeat(100) begin
            @(posedge clk);
            if (valid_out) begin
                $display("Output received!");
                for (i = 0; i < DIM; i = i + 1) begin
                    $write("%3d ", $signed(y_out[i*ACT_WIDTH +: ACT_WIDTH]));
                end
                $display("");
                break;
            end
        end
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #10000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
