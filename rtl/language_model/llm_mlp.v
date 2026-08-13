// =============================================================================
// SiLens - Language Model MLP Block
// =============================================================================
// Implements the MLP (feedforward) block for the language model decoder.
//
// Architecture:
//   - Gate projection: 576 -> 1536
//   - Up projection: 576 -> 1536
//   - SiLU activation on gate
//   - Element-wise multiply: gate * up
//   - Down projection: 1536 -> 576
//
// This is SwiGLU variant:
//   output = down(silu(gate(x)) * up(x))
//
// SiLU (Sigmoid Linear Unit) = x * sigmoid(x)
//
// License: Apache 2.0
// =============================================================================

module llm_mlp #(
    parameter DIM         = 576,                    // Input/output dimension
    parameter HIDDEN_DIM  = 1536,                   // Hidden dimension
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // Hardwired ternary weights
    input  wire [DIM*HIDDEN_DIM*2-1:0]  w_gate,             // Gate projection
    input  wire [DIM*HIDDEN_DIM*2-1:0]  w_up,               // Up projection
    input  wire [HIDDEN_DIM*DIM*2-1:0]  w_down,             // Down projection
    
    // Output interface
    output reg  [DIM*ACT_WIDTH-1:0]     y_out,
    output reg                          valid_out,
    input  wire                         ready_out
);

    // =========================================================================
    // Weight encoding
    // =========================================================================
    
    localparam W_ZERO = 2'b00;
    localparam W_POS  = 2'b01;
    localparam W_NEG  = 2'b10;
    
    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE     = 3'd0;
    localparam STATE_GATE_UP  = 3'd1;   // Compute gate and up projections
    localparam STATE_SILU     = 3'd2;   // Apply SiLU to gate
    localparam STATE_MULT     = 3'd3;   // Element-wise multiply
    localparam STATE_DOWN     = 3'd4;   // Down projection
    localparam STATE_OUTPUT   = 3'd5;
    
    reg [2:0] state;
    
    // =========================================================================
    // Processing indices
    // =========================================================================
    
    reg [$clog2(HIDDEN_DIM)-1:0] hidden_idx;
    reg [$clog2(DIM)-1:0] out_idx;
    
    // =========================================================================
    // Buffers
    // =========================================================================
    
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];
    reg signed [ACT_WIDTH-1:0] gate_buf [0:HIDDEN_DIM-1];     // After SiLU
    reg signed [ACT_WIDTH-1:0] up_buf [0:HIDDEN_DIM-1];
    reg signed [ACT_WIDTH-1:0] hidden_buf [0:HIDDEN_DIM-1];   // gate * up
    
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
    // Ternary MAC helper
    // =========================================================================
    
    function signed [ACC_WIDTH-1:0] ternary_mac_single;
        input signed [ACT_WIDTH-1:0] act;
        input [1:0] weight;
        begin
            case (weight)
                W_POS:   ternary_mac_single = act;
                W_NEG:   ternary_mac_single = -act;
                default: ternary_mac_single = 0;
            endcase
        end
    endfunction
    
    // =========================================================================
    // SiLU activation (piece-wise linear approximation)
    // =========================================================================
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    //
    // Approximation:
    //   x < -4:  SiLU(x) ≈ 0
    //   -4 ≤ x < 0: SiLU(x) ≈ x * (0.5 + x/8 + ...)
    //   x ≥ 0:   SiLU(x) ≈ x (approaches identity)
    
    function signed [ACT_WIDTH-1:0] silu_approx;
        input signed [ACT_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] x_ext;
        reg signed [ACC_WIDTH-1:0] result;
        reg signed [ACC_WIDTH-1:0] sigmoid;
        begin
            x_ext = x;
            
            // Approximate sigmoid
            if (x < -64) begin          // x < -4 (in Q4.4)
                sigmoid = 0;
            end else if (x < 0) begin
                // sigmoid ≈ 0.5 + x/8 for x in [-4, 0]
                sigmoid = (1 << (FRAC_BITS-1)) + (x_ext >>> 3);
            end else if (x < 64) begin  // 0 ≤ x < 4
                // sigmoid ≈ 0.5 + x/8
                sigmoid = (1 << (FRAC_BITS-1)) + (x_ext >>> 3);
            end else begin
                sigmoid = (1 << FRAC_BITS);  // 1.0
            end
            
            // SiLU = x * sigmoid
            result = (x_ext * sigmoid) >>> FRAC_BITS;
            
            silu_approx = saturate(result);
        end
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer proc_i;
    reg signed [ACC_WIDTH-1:0] gate_sum, up_sum, down_sum;
    reg signed [ACC_WIDTH-1:0] mult_result;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            hidden_idx <= 0;
            out_idx <= 0;
            valid_out <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        state <= STATE_GATE_UP;
                        hidden_idx <= 0;
                    end
                end
                
                STATE_GATE_UP: begin
                    // Compute gate and up projections for current hidden dim
                    gate_sum = 0;
                    up_sum = 0;
                    for (proc_i = 0; proc_i < DIM; proc_i = proc_i + 1) begin
                        gate_sum = gate_sum + ternary_mac_single(
                            x_buf[proc_i],
                            w_gate[(hidden_idx * DIM + proc_i) * 2 +: 2]
                        );
                        up_sum = up_sum + ternary_mac_single(
                            x_buf[proc_i],
                            w_up[(hidden_idx * DIM + proc_i) * 2 +: 2]
                        );
                    end
                    gate_buf[hidden_idx] <= saturate(gate_sum);
                    up_buf[hidden_idx] <= saturate(up_sum);
                    
                    if (hidden_idx >= HIDDEN_DIM - 1) begin
                        hidden_idx <= 0;
                        state <= STATE_SILU;
                    end else begin
                        hidden_idx <= hidden_idx + 1;
                    end
                end
                
                STATE_SILU: begin
                    // Apply SiLU to gate values
                    gate_buf[hidden_idx] <= silu_approx(gate_buf[hidden_idx]);
                    
                    if (hidden_idx >= HIDDEN_DIM - 1) begin
                        hidden_idx <= 0;
                        state <= STATE_MULT;
                    end else begin
                        hidden_idx <= hidden_idx + 1;
                    end
                end
                
                STATE_MULT: begin
                    // Element-wise multiply: gate * up
                    mult_result = ($signed(gate_buf[hidden_idx]) * 
                                  $signed(up_buf[hidden_idx])) >>> FRAC_BITS;
                    hidden_buf[hidden_idx] <= saturate(mult_result);
                    
                    if (hidden_idx >= HIDDEN_DIM - 1) begin
                        hidden_idx <= 0;
                        out_idx <= 0;
                        state <= STATE_DOWN;
                    end else begin
                        hidden_idx <= hidden_idx + 1;
                    end
                end
                
                STATE_DOWN: begin
                    // Down projection
                    down_sum = 0;
                    for (proc_i = 0; proc_i < HIDDEN_DIM; proc_i = proc_i + 1) begin
                        down_sum = down_sum + ternary_mac_single(
                            hidden_buf[proc_i],
                            w_down[(out_idx * HIDDEN_DIM + proc_i) * 2 +: 2]
                        );
                    end
                    y_out[out_idx*ACT_WIDTH +: ACT_WIDTH] <= saturate(down_sum);
                    
                    if (out_idx >= DIM - 1) begin
                        out_idx <= 0;
                        state <= STATE_OUTPUT;
                    end else begin
                        out_idx <= out_idx + 1;
                    end
                end
                
                STATE_OUTPUT: begin
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

module llm_mlp_tb;
    parameter DIM = 32;
    parameter HIDDEN_DIM = 64;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 4;
    
    reg clk, rst_n;
    reg [DIM*ACT_WIDTH-1:0] x_in;
    reg valid_in;
    wire ready_in;
    reg [DIM*HIDDEN_DIM*2-1:0] w_gate, w_up;
    reg [HIDDEN_DIM*DIM*2-1:0] w_down;
    wire [DIM*ACT_WIDTH-1:0] y_out;
    wire valid_out;
    reg ready_out;
    
    llm_mlp #(
        .DIM(DIM),
        .HIDDEN_DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i;
    
    initial begin
        $display("LLM MLP Testbench");
        $display("=================");
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        valid_in = 0;
        ready_out = 1;
        
        // Initialize weights
        w_gate = {(DIM*HIDDEN_DIM){2'b01}};
        w_up = {(DIM*HIDDEN_DIM){2'b01}};
        w_down = {(HIDDEN_DIM*DIM){2'b01}};
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Create input
        for (i = 0; i < DIM; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = (i % 16);
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for output
        repeat(100000) begin
            @(posedge clk);
            if (valid_out) begin
                $display("Output received!");
                break;
            end
        end
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
