// =============================================================================
// Layer Normalization Edge Block (192-dim) - Streaming Interface
// =============================================================================
// Edge-optimized Layer Norm with AXI-Stream style serialized interface.
// Implements: y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// STREAMING INTERFACE (reduces IO pins from 6150 to ~150):
// - 64-bit data bus for input/output
// - Loads x, gamma, beta sequentially (3 × 24 cycles = 72 cycles load)
// - Outputs normalized result over 24 cycles
//
// Protocol:
// 1. Set cfg_load_weights=1, stream gamma (24 × 64-bit words)
// 2. Set cfg_load_weights=1, stream beta (24 × 64-bit words)  
// 3. Set cfg_load_weights=0, stream x_in (24 × 64-bit words)
// 4. Wait for valid_out, read y_out (24 × 64-bit words)
//
// Target: ~0.8mm² on SKY130 at 50MHz with streaming interface
// Latency: ~200 cycles (72 load + 100 compute + 24 output)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module layer_norm_edge #(
    parameter DIM       = 192,      // Hidden dimension (NanoViT Edge)
    parameter ACT_WIDTH = 8,        // Activation bits
    parameter ACC_WIDTH = 32,       // Accumulator bits
    parameter BUS_WIDTH = 64        // Streaming bus width
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Configuration
    input  wire                     cfg_load_gamma,  // 1=loading gamma weights
    input  wire                     cfg_load_beta,   // 1=loading beta weights
    
    // Streaming input interface (AXI-Stream style)
    input  wire [BUS_WIDTH-1:0]     s_data,
    input  wire                     s_valid,
    output wire                     s_ready,
    input  wire                     s_last,          // Last word of current transfer
    
    // Streaming output interface
    output wire [BUS_WIDTH-1:0]     m_data,
    output wire                     m_valid,
    input  wire                     m_ready,
    output wire                     m_last
);

    // =========================================================================
    // Parameters
    // =========================================================================
    
    localparam FRAC_BITS = 8;       // Fixed-point fractional bits
    localparam EPS = 1;             // Small constant for numerical stability
    localparam DIM_BITS = $clog2(DIM);  // 8 bits for 192
    
    // Number of bus transfers needed: 192 elements × 8 bits / 64 bits = 24 words
    localparam WORDS_PER_VEC = (DIM * ACT_WIDTH + BUS_WIDTH - 1) / BUS_WIDTH;  // 24
    localparam WORD_CNT_BITS = $clog2(WORDS_PER_VEC + 1);
    
    // Elements per bus word: 64 bits / 8 bits = 8 elements
    localparam ELEM_PER_WORD = BUS_WIDTH / ACT_WIDTH;  // 8
    
    // =========================================================================
    // FSM States
    // =========================================================================
    
    localparam [3:0] ST_IDLE       = 4'd0;
    localparam [3:0] ST_LOAD_X     = 4'd1;
    localparam [3:0] ST_LOAD_GAMMA = 4'd2;
    localparam [3:0] ST_LOAD_BETA  = 4'd3;
    localparam [3:0] ST_CALC_MEAN  = 4'd4;
    localparam [3:0] ST_CALC_VAR   = 4'd5;
    localparam [3:0] ST_INV_SQRT   = 4'd6;
    localparam [3:0] ST_NORMALIZE  = 4'd7;
    localparam [3:0] ST_OUTPUT     = 4'd8;
    
    reg [3:0] state;
    
    // =========================================================================
    // Registers
    // =========================================================================
    
    // Input/weight buffers (stored as packed arrays for synthesis)
    reg [DIM*ACT_WIDTH-1:0] x_buf;
    reg [DIM*ACT_WIDTH-1:0] gamma_buf;
    reg [DIM*ACT_WIDTH-1:0] beta_buf;
    reg [DIM*ACT_WIDTH-1:0] y_buf;
    
    // Word counter for streaming
    reg [WORD_CNT_BITS-1:0] word_cnt;
    
    // Computation registers
    reg signed [ACC_WIDTH-1:0] sum_x;
    reg signed [ACC_WIDTH-1:0] sum_var;
    reg signed [ACC_WIDTH-1:0] mean;
    reg [ACC_WIDTH-1:0] variance;
    reg [ACC_WIDTH-1:0] inv_std;
    reg [DIM_BITS:0] elem_idx;
    reg [2:0] nr_iter;
    
    // Newton-Raphson working register
    reg [ACC_WIDTH-1:0] y_nr;
    
    // Output handshake
    reg out_valid_reg;
    reg out_last_reg;
    
    // =========================================================================
    // Streaming interface signals
    // =========================================================================
    
    wire loading_weights = cfg_load_gamma || cfg_load_beta;
    wire input_ready = (state == ST_IDLE && !loading_weights) ||
                       (state == ST_LOAD_X) ||
                       (state == ST_IDLE && cfg_load_gamma) ||
                       (state == ST_LOAD_GAMMA) ||
                       (state == ST_IDLE && cfg_load_beta) ||
                       (state == ST_LOAD_BETA);
    
    assign s_ready = input_ready;
    assign m_valid = out_valid_reg;
    assign m_last = out_last_reg;
    
    // =========================================================================
    // Input buffer loading - sequential word writes
    // =========================================================================
    
    wire [DIM_BITS:0] base_elem = word_cnt * ELEM_PER_WORD;
    
    // Unpack current streaming word into elements
    wire signed [ACT_WIDTH-1:0] stream_elem [0:ELEM_PER_WORD-1];
    genvar gi;
    generate
        for (gi = 0; gi < ELEM_PER_WORD; gi = gi + 1) begin : unpack_stream
            assign stream_elem[gi] = $signed(s_data[gi*ACT_WIDTH +: ACT_WIDTH]);
        end
    endgenerate
    
    // =========================================================================
    // Buffer write logic using generate blocks
    // =========================================================================
    
    generate
        for (gi = 0; gi < DIM; gi = gi + 1) begin : gen_x_buf
            always @(posedge clk) begin
                if (s_valid && s_ready && (state == ST_IDLE || state == ST_LOAD_X) && !loading_weights) begin
                    if (gi >= base_elem && gi < base_elem + ELEM_PER_WORD && gi < DIM) begin
                        x_buf[gi*ACT_WIDTH +: ACT_WIDTH] <= s_data[(gi - base_elem)*ACT_WIDTH +: ACT_WIDTH];
                    end
                end
            end
        end
    endgenerate
    
    generate
        for (gi = 0; gi < DIM; gi = gi + 1) begin : gen_gamma_buf
            always @(posedge clk) begin
                if (s_valid && s_ready && (state == ST_IDLE || state == ST_LOAD_GAMMA) && cfg_load_gamma) begin
                    if (gi >= base_elem && gi < base_elem + ELEM_PER_WORD && gi < DIM) begin
                        gamma_buf[gi*ACT_WIDTH +: ACT_WIDTH] <= s_data[(gi - base_elem)*ACT_WIDTH +: ACT_WIDTH];
                    end
                end
            end
        end
    endgenerate
    
    generate
        for (gi = 0; gi < DIM; gi = gi + 1) begin : gen_beta_buf
            always @(posedge clk) begin
                if (s_valid && s_ready && (state == ST_IDLE || state == ST_LOAD_BETA) && cfg_load_beta) begin
                    if (gi >= base_elem && gi < base_elem + ELEM_PER_WORD && gi < DIM) begin
                        beta_buf[gi*ACT_WIDTH +: ACT_WIDTH] <= s_data[(gi - base_elem)*ACT_WIDTH +: ACT_WIDTH];
                    end
                end
            end
        end
    endgenerate
    
    // =========================================================================
    // Current element access for computation
    // =========================================================================
    
    wire signed [ACT_WIDTH-1:0] x_curr = $signed(x_buf[elem_idx[DIM_BITS-1:0]*ACT_WIDTH +: ACT_WIDTH]);
    wire signed [ACT_WIDTH-1:0] gamma_curr = $signed(gamma_buf[elem_idx[DIM_BITS-1:0]*ACT_WIDTH +: ACT_WIDTH]);
    wire signed [ACT_WIDTH-1:0] beta_curr = $signed(beta_buf[elem_idx[DIM_BITS-1:0]*ACT_WIDTH +: ACT_WIDTH]);
    
    // =========================================================================
    // Mean calculation: sum(x) / N
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] x_curr_ext = {{(ACC_WIDTH-ACT_WIDTH){x_curr[ACT_WIDTH-1]}}, x_curr};
    
    // =========================================================================
    // Variance calculation: sum((x - mean)^2) / N
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] x_centered = x_curr_ext - mean;
    wire signed [2*ACC_WIDTH-1:0] x_centered_sq = x_centered * x_centered;
    
    // =========================================================================
    // Newton-Raphson inverse square root
    // y_{n+1} = y_n * (3 - x * y_n^2) / 2
    // =========================================================================
    
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
    
    // Saturation
    wire signed [ACT_WIDTH-1:0] y_saturated;
    assign y_saturated = (y_biased > $signed(32'sd127)) ? 8'sd127 :
                         (y_biased < $signed(-32'sd128)) ? -8'sd128 :
                         y_biased[ACT_WIDTH-1:0];
    
    // =========================================================================
    // Output buffer write during normalization
    // =========================================================================
    
    generate
        for (gi = 0; gi < DIM; gi = gi + 1) begin : gen_y_buf
            always @(posedge clk) begin
                if (state == ST_NORMALIZE && elem_idx[DIM_BITS-1:0] == gi) begin
                    y_buf[gi*ACT_WIDTH +: ACT_WIDTH] <= y_saturated;
                end
            end
        end
    endgenerate
    
    // =========================================================================
    // Output data mux
    // =========================================================================
    
    wire [BUS_WIDTH-1:0] out_word;
    assign out_word = y_buf[word_cnt*BUS_WIDTH +: BUS_WIDTH];
    assign m_data = out_word;
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            word_cnt <= 0;
            elem_idx <= 0;
            sum_x <= 0;
            sum_var <= 0;
            mean <= 0;
            variance <= 0;
            inv_std <= 0;
            y_nr <= 0;
            nr_iter <= 0;
            out_valid_reg <= 0;
            out_last_reg <= 0;
        end else begin
            case (state)
                // ---------------------------------------------------------
                // IDLE: Wait for data or weight load
                // ---------------------------------------------------------
                ST_IDLE: begin
                    out_valid_reg <= 0;
                    out_last_reg <= 0;
                    word_cnt <= 0;
                    
                    if (s_valid && s_ready) begin
                        if (cfg_load_gamma) begin
                            state <= ST_LOAD_GAMMA;
                            word_cnt <= 1;
                        end else if (cfg_load_beta) begin
                            state <= ST_LOAD_BETA;
                            word_cnt <= 1;
                        end else begin
                            state <= ST_LOAD_X;
                            word_cnt <= 1;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                // LOAD_GAMMA: Stream in gamma weights
                // ---------------------------------------------------------
                ST_LOAD_GAMMA: begin
                    if (s_valid && s_ready) begin
                        if (word_cnt == WORDS_PER_VEC - 1 || s_last) begin
                            state <= ST_IDLE;
                            word_cnt <= 0;
                        end else begin
                            word_cnt <= word_cnt + 1;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                // LOAD_BETA: Stream in beta weights
                // ---------------------------------------------------------
                ST_LOAD_BETA: begin
                    if (s_valid && s_ready) begin
                        if (word_cnt == WORDS_PER_VEC - 1 || s_last) begin
                            state <= ST_IDLE;
                            word_cnt <= 0;
                        end else begin
                            word_cnt <= word_cnt + 1;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                // LOAD_X: Stream in input vector, then compute
                // ---------------------------------------------------------
                ST_LOAD_X: begin
                    if (s_valid && s_ready) begin
                        if (word_cnt == WORDS_PER_VEC - 1 || s_last) begin
                            state <= ST_CALC_MEAN;
                            word_cnt <= 0;
                            elem_idx <= 0;
                            sum_x <= 0;
                        end else begin
                            word_cnt <= word_cnt + 1;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                // CALC_MEAN: Accumulate sum(x) iteratively
                // ---------------------------------------------------------
                ST_CALC_MEAN: begin
                    sum_x <= sum_x + x_curr_ext;
                    
                    if (elem_idx[DIM_BITS-1:0] == DIM - 1) begin
                        state <= ST_CALC_VAR;
                        mean <= (sum_x + x_curr_ext) >>> DIM_BITS;
                        elem_idx <= 0;
                        sum_var <= 0;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end
                
                // ---------------------------------------------------------
                // CALC_VAR: Accumulate sum((x - mean)^2) iteratively
                // ---------------------------------------------------------
                ST_CALC_VAR: begin
                    sum_var <= sum_var + x_centered_sq[ACC_WIDTH-1:0];
                    
                    if (elem_idx[DIM_BITS-1:0] == DIM - 1) begin
                        state <= ST_INV_SQRT;
                        variance <= ((sum_var + x_centered_sq[ACC_WIDTH-1:0]) >> DIM_BITS) + EPS;
                        nr_iter <= 0;
                        y_nr <= 1 << FRAC_BITS;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end
                
                // ---------------------------------------------------------
                // INV_SQRT: Newton-Raphson iterations for 1/sqrt(var)
                // ---------------------------------------------------------
                ST_INV_SQRT: begin
                    y_nr <= y_next;
                    
                    if (nr_iter == 4) begin
                        inv_std <= y_nr;
                        state <= ST_NORMALIZE;
                        elem_idx <= 0;
                    end else begin
                        nr_iter <= nr_iter + 1;
                    end
                end
                
                // ---------------------------------------------------------
                // NORMALIZE: Apply normalization element by element
                // ---------------------------------------------------------
                ST_NORMALIZE: begin
                    if (elem_idx[DIM_BITS-1:0] == DIM - 1) begin
                        state <= ST_OUTPUT;
                        word_cnt <= 0;
                        out_valid_reg <= 1;
                        out_last_reg <= 0;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end
                
                // ---------------------------------------------------------
                // OUTPUT: Stream out result
                // ---------------------------------------------------------
                ST_OUTPUT: begin
                    if (m_ready) begin
                        if (word_cnt == WORDS_PER_VEC - 1) begin
                            out_valid_reg <= 0;
                            out_last_reg <= 0;
                            state <= ST_IDLE;
                            word_cnt <= 0;
                        end else begin
                            word_cnt <= word_cnt + 1;
                            if (word_cnt == WORDS_PER_VEC - 2) begin
                                out_last_reg <= 1;
                            end
                        end
                    end
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
