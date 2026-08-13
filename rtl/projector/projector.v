// =============================================================================
// SiLens - Multimodal Projector
// =============================================================================
// Linear projection layer mapping vision encoder output to language model space.
//
// Architecture:
//   - Input: 576 tokens x 768 dimensions (from vision encoder)
//   - Linear projection: 768 -> 576 dimensions
//   - Output: 576 tokens x 576 dimensions (for language model)
//
// Parameters:
//   - ~18M parameters (768 x 576 x 49 weight bits) -- simplified to ternary
//   - Using ternary weights for hardware efficiency
//
// Operation:
//   For each input token x[768]:
//     y[576] = W[576x768] * x[768]
//
// License: Apache 2.0
// =============================================================================

module projector #(
    parameter IN_DIM      = 768,                    // Input dimension (vision)
    parameter OUT_DIM     = 576,                    // Output dimension (LLM)
    parameter SEQ_LEN     = 576,                    // Sequence length
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input interface (streaming tokens)
    input  wire [IN_DIM*ACT_WIDTH-1:0]  x_in,               // Input token
    input  wire [$clog2(SEQ_LEN)-1:0]   token_idx_in,       // Token index
    input  wire                         token_valid_in,
    output wire                         token_ready_in,
    
    // Sequence control
    input  wire                         seq_start,          // Start of sequence
    input  wire                         seq_done_in,        // All input tokens received
    
    // Hardwired ternary weights
    // W: OUT_DIM x IN_DIM = 576 x 768 = 442,368 weights x 2 bits = 884,736 bits
    input  wire [OUT_DIM*IN_DIM*2-1:0]  weights,
    
    // Optional bias
    input  wire [OUT_DIM*ACT_WIDTH-1:0] bias,
    
    // Output interface (streaming tokens)
    output reg  [OUT_DIM*ACT_WIDTH-1:0] y_out,              // Output token
    output reg  [$clog2(SEQ_LEN)-1:0]   token_idx_out,
    output reg                          token_valid_out,
    input  wire                         token_ready_out,
    
    // Status
    output wire                         busy
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
    
    localparam STATE_IDLE    = 3'd0;
    localparam STATE_LOAD    = 3'd1;   // Load input token
    localparam STATE_PROJECT = 3'd2;   // Compute projection
    localparam STATE_OUTPUT  = 3'd3;   // Output projected token
    
    reg [2:0] state;
    
    // =========================================================================
    // Processing counters
    // =========================================================================
    
    reg [$clog2(OUT_DIM)-1:0] out_idx;              // Current output dimension
    reg [$clog2(IN_DIM/PARALLEL+1)-1:0] mac_iter;   // MAC iteration within dimension
    reg [$clog2(SEQ_LEN)-1:0] current_token;        // Current token being processed
    reg [$clog2(SEQ_LEN)-1:0] tokens_received;
    reg all_tokens_done;
    
    localparam NUM_MAC_ITERS = (IN_DIM + PARALLEL - 1) / PARALLEL;
    
    // =========================================================================
    // Input buffer
    // =========================================================================
    
    reg signed [ACT_WIDTH-1:0] x_buf [0:IN_DIM-1];
    
    // =========================================================================
    // Accumulator
    // =========================================================================
    
    reg signed [ACC_WIDTH-1:0] mac_accum;
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign token_ready_in = (state == STATE_LOAD);
    assign busy = (state != STATE_IDLE);
    
    // =========================================================================
    // Input loading
    // =========================================================================
    
    integer load_i;
    
    always @(posedge clk) begin
        if (state == STATE_LOAD && token_valid_in) begin
            for (load_i = 0; load_i < IN_DIM; load_i = load_i + 1) begin
                x_buf[load_i] <= $signed(x_in[load_i*ACT_WIDTH +: ACT_WIDTH]);
            end
        end
    end
    
    // =========================================================================
    // Ternary MAC computation
    // =========================================================================
    
    // Parallel MAC for projection
    wire signed [ACC_WIDTH-1:0] mac_partial;
    reg signed [ACC_WIDTH-1:0] partial_sum [0:PARALLEL-1];
    
    integer mac_i;
    always @(*) begin
        for (mac_i = 0; mac_i < PARALLEL; mac_i = mac_i + 1) begin
            partial_sum[mac_i] = 0;
            if (mac_iter * PARALLEL + mac_i < IN_DIM) begin
                // Weight index: out_idx * IN_DIM + (mac_iter * PARALLEL + mac_i)
                case (weights[(out_idx * IN_DIM + mac_iter * PARALLEL + mac_i) * 2 +: 2])
                    W_POS:   partial_sum[mac_i] = $signed({1'b0, x_buf[mac_iter * PARALLEL + mac_i]});
                    W_NEG:   partial_sum[mac_i] = -$signed({1'b0, x_buf[mac_iter * PARALLEL + mac_i]});
                    default: partial_sum[mac_i] = 0;
                endcase
            end
        end
    end
    
    // Sum partial results
    reg signed [ACC_WIDTH-1:0] mac_tree;
    integer tree_i;
    always @(*) begin
        mac_tree = 0;
        for (tree_i = 0; tree_i < PARALLEL; tree_i = tree_i + 1) begin
            mac_tree = mac_tree + partial_sum[tree_i];
        end
    end
    assign mac_partial = mac_tree;
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            out_idx <= 0;
            mac_iter <= 0;
            current_token <= 0;
            tokens_received <= 0;
            all_tokens_done <= 1'b0;
            mac_accum <= 0;
            token_valid_out <= 1'b0;
            token_idx_out <= 0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    token_valid_out <= 1'b0;
                    if (seq_start) begin
                        state <= STATE_LOAD;
                        current_token <= 0;
                        tokens_received <= 0;
                        all_tokens_done <= 1'b0;
                    end
                end
                
                STATE_LOAD: begin
                    // Wait for input token
                    if (token_valid_in && token_ready_in) begin
                        current_token <= token_idx_in;
                        tokens_received <= tokens_received + 1;
                        state <= STATE_PROJECT;
                        out_idx <= 0;
                        mac_iter <= 0;
                        mac_accum <= 0;
                    end else if (seq_done_in && tokens_received > 0) begin
                        all_tokens_done <= 1'b1;
                    end
                end
                
                STATE_PROJECT: begin
                    // Accumulate partial sum
                    mac_accum <= mac_accum + mac_partial;
                    
                    if (mac_iter >= NUM_MAC_ITERS - 1) begin
                        // Finished one output dimension
                        // Add bias and saturate
                        y_out[out_idx*ACT_WIDTH +: ACT_WIDTH] <= saturate(
                            mac_accum + mac_partial + 
                            $signed(bias[out_idx*ACT_WIDTH +: ACT_WIDTH])
                        );
                        
                        mac_iter <= 0;
                        mac_accum <= 0;
                        
                        if (out_idx >= OUT_DIM - 1) begin
                            // Finished all output dimensions
                            out_idx <= 0;
                            token_idx_out <= current_token;
                            state <= STATE_OUTPUT;
                        end else begin
                            out_idx <= out_idx + 1;
                        end
                    end else begin
                        mac_iter <= mac_iter + 1;
                    end
                end
                
                STATE_OUTPUT: begin
                    token_valid_out <= 1'b1;
                    
                    if (token_ready_out) begin
                        token_valid_out <= 1'b0;
                        
                        // Check if more tokens to process
                        if (all_tokens_done) begin
                            state <= STATE_IDLE;
                        end else begin
                            state <= STATE_LOAD;
                        end
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

module projector_tb;
    parameter IN_DIM = 32;
    parameter OUT_DIM = 24;
    parameter SEQ_LEN = 8;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter PARALLEL = 8;
    
    reg clk, rst_n;
    reg [IN_DIM*ACT_WIDTH-1:0] x_in;
    reg [$clog2(SEQ_LEN)-1:0] token_idx_in;
    reg token_valid_in;
    wire token_ready_in;
    reg seq_start, seq_done_in;
    reg [OUT_DIM*IN_DIM*2-1:0] weights;
    reg [OUT_DIM*ACT_WIDTH-1:0] bias;
    wire [OUT_DIM*ACT_WIDTH-1:0] y_out;
    wire [$clog2(SEQ_LEN)-1:0] token_idx_out;
    wire token_valid_out;
    reg token_ready_out;
    wire busy;
    
    projector #(
        .IN_DIM(IN_DIM),
        .OUT_DIM(OUT_DIM),
        .SEQ_LEN(SEQ_LEN),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i, j, tokens_out;
    
    initial begin
        $display("Projector Testbench");
        $display("===================");
        $display("IN_DIM=%0d, OUT_DIM=%0d, SEQ_LEN=%0d", IN_DIM, OUT_DIM, SEQ_LEN);
        
        clk = 0;
        rst_n = 0;
        x_in = 0;
        token_idx_in = 0;
        token_valid_in = 0;
        seq_start = 0;
        seq_done_in = 0;
        token_ready_out = 1;
        
        // Initialize weights (all +1 for simple test)
        weights = {(OUT_DIM*IN_DIM){2'b01}};
        bias = 0;
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Start sequence
        seq_start = 1;
        @(posedge clk);
        seq_start = 0;
        
        // Send tokens
        tokens_out = 0;
        
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            while (!token_ready_in) begin
                @(posedge clk);
                if (token_valid_out) begin
                    tokens_out = tokens_out + 1;
                    $display("Output token %0d received", token_idx_out);
                end
            end
            
            // Create input vector
            for (j = 0; j < IN_DIM; j = j + 1) begin
                x_in[j*ACT_WIDTH +: ACT_WIDTH] = (i + j) % 32;
            end
            token_idx_in = i;
            token_valid_in = 1;
            
            @(posedge clk);
            token_valid_in = 0;
        end
        
        // Signal sequence done
        seq_done_in = 1;
        @(posedge clk);
        seq_done_in = 0;
        
        // Wait for remaining outputs
        repeat(10000) begin
            @(posedge clk);
            if (token_valid_out) begin
                tokens_out = tokens_out + 1;
                $display("Output token %0d received", token_idx_out);
            end
            if (!busy && tokens_out >= SEQ_LEN) break;
        end
        
        $display("Received %0d output tokens (expected %0d)", tokens_out, SEQ_LEN);
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #1000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
