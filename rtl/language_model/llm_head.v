// =============================================================================
// SiLens - Language Model Output Head
// =============================================================================
// Implements the output head for the language model.
//
// Architecture:
//   1. Final RMSNorm
//   2. Linear projection to vocabulary (576 -> 49152)
//   3. Output logits (or argmax for token prediction)
//
// The vocabulary projection is the largest single weight matrix.
// For 49152 vocab x 576 dim = 28.3M weights
// With ternary encoding: 56.6 Mbits = 7.08 MB
//
// License: Apache 2.0
// =============================================================================

module llm_head #(
    parameter DIM         = 576,                    // Input dimension
    parameter VOCAB_SIZE  = 49152,                  // Vocabulary size
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter FRAC_BITS   = 4,                      // Fractional bits
    parameter PARALLEL    = 16,                     // Parallel MAC operations
    parameter TOP_K       = 1                       // Return top-K predictions
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface
    input  wire [DIM*ACT_WIDTH-1:0]     x_in,               // Final hidden state
    input  wire                         valid_in,
    output wire                         ready_in,
    
    // RMSNorm weights
    input  wire [DIM*ACT_WIDTH-1:0]     rms_gamma,
    
    // Vocabulary projection weights (hardwired ternary)
    // vocab_weights: VOCAB_SIZE x DIM = 49152 x 576 x 2 bits
    input  wire [VOCAB_SIZE*DIM*2-1:0]  vocab_weights,
    
    // Output interface
    output reg  [$clog2(VOCAB_SIZE)-1:0] token_out,         // Predicted token ID
    output reg  signed [ACC_WIDTH-1:0]  logit_out,          // Logit value (optional)
    output reg                          valid_out,
    input  wire                         ready_out,
    
    // Mode control
    input  wire                         output_logits       // 1: output all logits, 0: argmax only
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
    localparam STATE_RMS      = 3'd1;   // Apply RMSNorm
    localparam STATE_PROJECT  = 3'd2;   // Vocabulary projection
    localparam STATE_ARGMAX   = 3'd3;   // Find maximum
    localparam STATE_OUTPUT   = 3'd4;
    
    reg [2:0] state;
    
    // =========================================================================
    // Buffers
    // =========================================================================
    
    reg signed [ACT_WIDTH-1:0] x_buf [0:DIM-1];           // Input buffer
    reg signed [ACT_WIDTH-1:0] rms_buf [0:DIM-1];         // After RMSNorm
    
    // Logit computation
    reg signed [ACC_WIDTH-1:0] current_logit;
    reg signed [ACC_WIDTH-1:0] max_logit;
    reg [$clog2(VOCAB_SIZE)-1:0] max_idx;
    reg [$clog2(VOCAB_SIZE)-1:0] vocab_idx;
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign ready_in = (state == STATE_IDLE);
    
    // =========================================================================
    // RMSNorm computation
    // =========================================================================
    
    function signed [ACC_WIDTH-1:0] inv_sqrt_approx;
        input signed [ACC_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] y;
        reg signed [ACC_WIDTH-1:0] three;
        integer iter;
        begin
            three = 3 << FRAC_BITS;
            y = 1 << FRAC_BITS;
            
            for (iter = 0; iter < 3; iter = iter + 1) begin
                y = (y * (three - ((x * ((y * y) >>> FRAC_BITS)) >>> FRAC_BITS))) >>> (FRAC_BITS + 1);
            end
            
            inv_sqrt_approx = y;
        end
    endfunction
    
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
    // Main FSM
    // =========================================================================
    
    integer load_i, rms_i, proj_i;
    reg signed [ACC_WIDTH-1:0] mean_sq;
    reg signed [ACC_WIDTH-1:0] inv_rms;
    reg signed [ACC_WIDTH-1:0] proj_sum;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            vocab_idx <= 0;
            max_logit <= {1'b1, {(ACC_WIDTH-1){1'b0}}};  // Min value
            max_idx <= 0;
            valid_out <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    valid_out <= 1'b0;
                    
                    if (valid_in) begin
                        // Load input
                        for (load_i = 0; load_i < DIM; load_i = load_i + 1) begin
                            x_buf[load_i] <= $signed(x_in[load_i*ACT_WIDTH +: ACT_WIDTH]);
                        end
                        state <= STATE_RMS;
                    end
                end
                
                STATE_RMS: begin
                    // Compute RMSNorm
                    mean_sq = 0;
                    for (rms_i = 0; rms_i < DIM; rms_i = rms_i + 1) begin
                        mean_sq = mean_sq + ($signed(x_buf[rms_i]) * $signed(x_buf[rms_i]));
                    end
                    mean_sq = mean_sq / DIM;
                    inv_rms = inv_sqrt_approx(mean_sq + 1);
                    
                    // Apply normalization with gamma
                    for (rms_i = 0; rms_i < DIM; rms_i = rms_i + 1) begin
                        rms_buf[rms_i] <= saturate(
                            ($signed(x_buf[rms_i]) * inv_rms * 
                             $signed(rms_gamma[rms_i*ACT_WIDTH +: ACT_WIDTH])) >>> (2*FRAC_BITS)
                        );
                    end
                    
                    state <= STATE_PROJECT;
                    vocab_idx <= 0;
                    max_logit <= {1'b1, {(ACC_WIDTH-1){1'b0}}};
                    max_idx <= 0;
                end
                
                STATE_PROJECT: begin
                    // Compute logit for current vocabulary position
                    proj_sum = 0;
                    for (proj_i = 0; proj_i < DIM; proj_i = proj_i + 1) begin
                        proj_sum = proj_sum + ternary_mac_single(
                            rms_buf[proj_i],
                            vocab_weights[(vocab_idx * DIM + proj_i) * 2 +: 2]
                        );
                    end
                    current_logit <= proj_sum;
                    
                    // Update max tracking
                    if (proj_sum > max_logit) begin
                        max_logit <= proj_sum;
                        max_idx <= vocab_idx;
                    end
                    
                    // Continue or finish
                    if (vocab_idx >= VOCAB_SIZE - 1) begin
                        state <= STATE_ARGMAX;
                    end else begin
                        vocab_idx <= vocab_idx + 1;
                    end
                end
                
                STATE_ARGMAX: begin
                    // Finalize argmax result
                    token_out <= max_idx;
                    logit_out <= max_logit;
                    state <= STATE_OUTPUT;
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

module llm_head_tb;
    parameter DIM = 32;
    parameter VOCAB_SIZE = 64;  // Small for testing
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    
    reg clk, rst_n;
    reg [DIM*ACT_WIDTH-1:0] x_in;
    reg valid_in;
    wire ready_in;
    reg [DIM*ACT_WIDTH-1:0] rms_gamma;
    reg [VOCAB_SIZE*DIM*2-1:0] vocab_weights;
    wire [$clog2(VOCAB_SIZE)-1:0] token_out;
    wire signed [ACC_WIDTH-1:0] logit_out;
    wire valid_out;
    reg ready_out;
    reg output_logits;
    
    llm_head #(
        .DIM(DIM),
        .VOCAB_SIZE(VOCAB_SIZE),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i;
    
    initial begin
        $display("LLM Head Testbench");
        $display("==================");
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        valid_in = 0;
        ready_out = 1;
        output_logits = 0;
        
        // Initialize
        rms_gamma = {DIM{8'd16}};
        
        // Set weights: make vocab_idx=5 have highest response
        vocab_weights = {(VOCAB_SIZE*DIM){2'b00}};  // All zero
        for (i = 0; i < DIM; i = i + 1) begin
            vocab_weights[(5 * DIM + i) * 2 +: 2] = 2'b01;  // +1 for token 5
        end
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Create input
        for (i = 0; i < DIM; i = i + 1) begin
            x_in[i*ACT_WIDTH +: ACT_WIDTH] = 8'd16;  // 1.0 in Q4.4
        end
        
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for output
        repeat(100000) begin
            @(posedge clk);
            if (valid_out) begin
                $display("Predicted token: %0d, logit: %0d", token_out, logit_out);
                if (token_out == 5)
                    $display("PASS: Correctly predicted token 5");
                else
                    $display("FAIL: Expected token 5, got %0d", token_out);
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
