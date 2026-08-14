// =============================================================================
// MLP Block (Gated MLP / SwiGLU style) - Level 1 Synthesis Block
// =============================================================================
// Implements Gated MLP used in modern transformers:
//   out = down_proj(silu(gate_proj(x)) * up_proj(x))
//
// This is the SwiGLU variant where:
//   - gate_proj and up_proj both project input to hidden dimension
//   - SiLU activation applied to gate path
//   - Element-wise multiply gates the up_proj path
//   - down_proj reduces back to input dimension
//
// Target: ~3mm² (1700µm × 1700µm) on SKY130
// Reuse: 42× (30 in LLM @ 576→1536, 12 in Vision @ 768→3072)
//
// Uses ternary weights (2-bit encoded: 00=0, 01=+1, 10=-1)
// Streaming token interface for transformer integration
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module mlp_block #(
    parameter IN_DIM      = 576,      // Input dimension (576 LLM, 768 Vision)
    parameter HIDDEN_DIM  = 1536,     // Hidden dimension (1536 LLM, 3072 Vision)
    parameter ACT_WIDTH   = 8,        // Activation bit width
    parameter ACC_WIDTH   = 24,       // Accumulator width for MAC operations
    parameter SILU_LUT_DEPTH = 256    // SiLU lookup table entries
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Token streaming interface - Input
    input  wire                         token_valid_in,
    output wire                         token_ready_in,
    input  wire [IN_DIM*ACT_WIDTH-1:0]  token_data_in,
    input  wire                         token_last_in,      // Last token in sequence
    
    // Token streaming interface - Output  
    output wire                         token_valid_out,
    input  wire                         token_ready_out,
    output wire [IN_DIM*ACT_WIDTH-1:0]  token_data_out,
    output wire                         token_last_out,
    
    // Weight memory interface (external SRAM or weight buffer)
    // Gate projection weights: IN_DIM × HIDDEN_DIM ternary
    output wire                         gate_weight_rd_en,
    output wire [$clog2(IN_DIM)-1:0]    gate_weight_addr,
    input  wire [HIDDEN_DIM*2-1:0]      gate_weight_data,
    
    // Up projection weights: IN_DIM × HIDDEN_DIM ternary
    output wire                         up_weight_rd_en,
    output wire [$clog2(IN_DIM)-1:0]    up_weight_addr,
    input  wire [HIDDEN_DIM*2-1:0]      up_weight_data,
    
    // Down projection weights: HIDDEN_DIM × IN_DIM ternary
    output wire                         down_weight_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0] down_weight_addr,
    input  wire [IN_DIM*2-1:0]          down_weight_data,
    
    // Status
    output wire                         busy
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam IN_ADDR_WIDTH   = $clog2(IN_DIM);
    localparam HID_ADDR_WIDTH  = $clog2(HIDDEN_DIM);
    
    // Pipeline stages
    localparam STAGE_IDLE       = 3'd0;
    localparam STAGE_GATE_UP    = 3'd1;  // Compute gate_proj and up_proj
    localparam STAGE_SILU_MUL   = 3'd2;  // SiLU(gate) * up
    localparam STAGE_DOWN_PROJ  = 3'd3;  // down_proj
    localparam STAGE_OUTPUT     = 3'd4;  // Output token
    
    // =========================================================================
    // State Machine
    // =========================================================================
    reg [2:0]                   state;
    reg [2:0]                   next_state;
    reg [IN_ADDR_WIDTH-1:0]     input_cnt;
    reg [HID_ADDR_WIDTH-1:0]    hidden_cnt;
    reg                         token_last_reg;
    
    // =========================================================================
    // Input Token Buffer
    // =========================================================================
    reg [IN_DIM*ACT_WIDTH-1:0]  input_buffer;
    reg                         input_valid;
    
    // =========================================================================
    // Intermediate Buffers
    // =========================================================================
    // Gate projection result (before SiLU)
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] gate_buffer;
    // Up projection result
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] up_buffer;
    // After SiLU and multiply
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] gated_buffer;
    // Output buffer
    reg [IN_DIM*ACT_WIDTH-1:0]    output_buffer;
    reg                           output_valid;
    
    // =========================================================================
    // SiLU Approximation LUT
    // =========================================================================
    // SiLU(x) = x * sigmoid(x)
    // Pre-computed lookup table for 8-bit signed input
    reg signed [ACT_WIDTH-1:0] silu_lut [0:SILU_LUT_DEPTH-1];
    
    // Initialize SiLU LUT (piecewise linear approximation)
    integer lut_idx;
    initial begin
        for (lut_idx = 0; lut_idx < SILU_LUT_DEPTH; lut_idx = lut_idx + 1) begin
            // Map LUT index to signed input range [-128, 127] -> scaled SiLU output
            // SiLU approximation: for negative x, output is small; for positive x, ~x
            if (lut_idx < 128) begin
                // Negative region: SiLU approaches 0 for very negative, linear near 0
                silu_lut[lut_idx] = (lut_idx < 64) ? 8'sd0 : 
                                    $signed(lut_idx - 128) >>> 2;
            end else begin
                // Positive region: approximately linear with slight curve
                silu_lut[lut_idx] = $signed(lut_idx - 128);
            end
        end
    end
    
    // =========================================================================
    // Ternary MAC Unit (reused for all projections)
    // =========================================================================
    // Process one row at a time for area efficiency
    
    wire signed [ACT_WIDTH-1:0] mac_activations [0:63];
    wire [1:0]                  mac_weights [0:63];
    wire signed [ACC_WIDTH-1:0] mac_result;
    
    // 64-wide parallel MAC (matches ternary_mac_array_64)
    reg [63:0]                  mac_chunk_sel;
    reg [5:0]                   mac_chunk_idx;
    
    // Extract 64 activations for current MAC operation
    genvar gi;
    generate
        for (gi = 0; gi < 64; gi = gi + 1) begin : mac_act_extract
            wire [IN_ADDR_WIDTH-1:0] act_idx;
            assign act_idx = mac_chunk_idx * 64 + gi;
            
            // Safely extract activation (zero if out of bounds)
            wire in_bounds = (act_idx < IN_DIM);
            assign mac_activations[gi] = in_bounds ? 
                $signed(input_buffer[act_idx*ACT_WIDTH +: ACT_WIDTH]) : 
                {ACT_WIDTH{1'b0}};
        end
    endgenerate
    
    // Extract 64 weights based on current operation
    wire [HIDDEN_DIM*2-1:0] current_weights;
    assign current_weights = (state == STAGE_GATE_UP && !mac_chunk_sel[0]) ? gate_weight_data :
                             (state == STAGE_GATE_UP &&  mac_chunk_sel[0]) ? up_weight_data :
                             {{(HIDDEN_DIM*2-IN_DIM*2){1'b0}}, down_weight_data};
    
    generate
        for (gi = 0; gi < 64; gi = gi + 1) begin : mac_weight_extract
            wire [HID_ADDR_WIDTH-1:0] weight_offset;
            assign weight_offset = hidden_cnt + gi;
            
            // Extract weight pair
            assign mac_weights[gi] = current_weights[weight_offset*2 +: 2];
        end
    endgenerate
    
    // Parallel ternary MAC computation
    wire signed [ACT_WIDTH:0] products [0:63];
    generate
        for (gi = 0; gi < 64; gi = gi + 1) begin : ternary_mult
            assign products[gi] = (mac_weights[gi] == 2'b01) ?  {mac_activations[gi][ACT_WIDTH-1], mac_activations[gi]} :
                                  (mac_weights[gi] == 2'b10) ? -{mac_activations[gi][ACT_WIDTH-1], mac_activations[gi]} :
                                                                {(ACT_WIDTH+1){1'b0}};
        end
    endgenerate
    
    // 6-level reduction tree for 64 products
    wire signed [ACT_WIDTH+1:0] sum_l1 [0:31];
    wire signed [ACT_WIDTH+2:0] sum_l2 [0:15];
    wire signed [ACT_WIDTH+3:0] sum_l3 [0:7];
    wire signed [ACT_WIDTH+4:0] sum_l4 [0:3];
    wire signed [ACT_WIDTH+5:0] sum_l5 [0:1];
    wire signed [ACT_WIDTH+6:0] sum_l6;
    
    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : l1_add
            assign sum_l1[gi] = $signed(products[gi*2]) + $signed(products[gi*2+1]);
        end
        for (gi = 0; gi < 16; gi = gi + 1) begin : l2_add
            assign sum_l2[gi] = $signed(sum_l1[gi*2]) + $signed(sum_l1[gi*2+1]);
        end
        for (gi = 0; gi < 8; gi = gi + 1) begin : l3_add
            assign sum_l3[gi] = $signed(sum_l2[gi*2]) + $signed(sum_l2[gi*2+1]);
        end
        for (gi = 0; gi < 4; gi = gi + 1) begin : l4_add
            assign sum_l4[gi] = $signed(sum_l3[gi*2]) + $signed(sum_l3[gi*2+1]);
        end
    endgenerate
    
    assign sum_l5[0] = $signed(sum_l4[0]) + $signed(sum_l4[1]);
    assign sum_l5[1] = $signed(sum_l4[2]) + $signed(sum_l4[3]);
    assign sum_l6 = $signed(sum_l5[0]) + $signed(sum_l5[1]);
    
    // Sign-extend to accumulator width
    assign mac_result = {{(ACC_WIDTH-ACT_WIDTH-7){sum_l6[ACT_WIDTH+6]}}, sum_l6};
    
    // =========================================================================
    // Accumulator for multi-chunk MAC
    // =========================================================================
    reg signed [ACC_WIDTH-1:0] accumulator;
    reg                        acc_clear;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= {ACC_WIDTH{1'b0}};
        end else if (acc_clear) begin
            accumulator <= mac_result;
        end else begin
            accumulator <= accumulator + mac_result;
        end
    end
    
    // Saturate and truncate accumulator to activation width
    wire signed [ACT_WIDTH-1:0] acc_saturated;
    assign acc_saturated = (accumulator > $signed({{(ACC_WIDTH-ACT_WIDTH){1'b0}}, {(ACT_WIDTH-1){1'b1}}})) ? 
                           $signed({1'b0, {(ACT_WIDTH-1){1'b1}}}) :  // Max positive
                           (accumulator < $signed({{(ACC_WIDTH-ACT_WIDTH+1){1'b1}}, {(ACT_WIDTH-1){1'b0}}})) ?
                           $signed({1'b1, {(ACT_WIDTH-1){1'b0}}}) :  // Max negative  
                           accumulator[ACT_WIDTH-1:0];
    
    // =========================================================================
    // SiLU Application
    // =========================================================================
    wire [7:0] silu_addr;
    wire signed [ACT_WIDTH-1:0] silu_out;
    
    assign silu_addr = acc_saturated + 8'd128;  // Map signed to unsigned index
    assign silu_out = silu_lut[silu_addr];
    
    // =========================================================================
    // Gating Multiply
    // =========================================================================
    reg signed [ACT_WIDTH-1:0] gate_val;
    reg signed [ACT_WIDTH-1:0] up_val;
    wire signed [2*ACT_WIDTH-1:0] gated_product;
    wire signed [ACT_WIDTH-1:0] gated_val;
    
    assign gated_product = silu_out * up_val;
    // Truncate and saturate
    assign gated_val = gated_product[2*ACT_WIDTH-2:ACT_WIDTH-1];
    
    // =========================================================================
    // State Machine Logic
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STAGE_IDLE;
            input_cnt <= {IN_ADDR_WIDTH{1'b0}};
            hidden_cnt <= {HID_ADDR_WIDTH{1'b0}};
            input_valid <= 1'b0;
            output_valid <= 1'b0;
            token_last_reg <= 1'b0;
            mac_chunk_idx <= 6'd0;
            mac_chunk_sel <= 64'd0;
            acc_clear <= 1'b1;
            gate_val <= {ACT_WIDTH{1'b0}};
            up_val <= {ACT_WIDTH{1'b0}};
        end else begin
            case (state)
                STAGE_IDLE: begin
                    output_valid <= 1'b0;
                    if (token_valid_in && token_ready_in) begin
                        input_buffer <= token_data_in;
                        token_last_reg <= token_last_in;
                        input_valid <= 1'b1;
                        state <= STAGE_GATE_UP;
                        input_cnt <= {IN_ADDR_WIDTH{1'b0}};
                        hidden_cnt <= {HID_ADDR_WIDTH{1'b0}};
                        mac_chunk_idx <= 6'd0;
                        mac_chunk_sel <= 64'd0;  // 0 = gate, 1 = up
                        acc_clear <= 1'b1;
                    end
                end
                
                STAGE_GATE_UP: begin
                    // Process all input dimensions in chunks of 64
                    acc_clear <= 1'b0;
                    
                    if (mac_chunk_idx == (IN_DIM + 63) / 64 - 1) begin
                        // Finished one hidden dimension
                        mac_chunk_idx <= 6'd0;
                        acc_clear <= 1'b1;
                        
                        if (!mac_chunk_sel[0]) begin
                            // Just finished gate_proj for this hidden dim
                            gate_buffer[hidden_cnt*ACT_WIDTH +: ACT_WIDTH] <= acc_saturated;
                            mac_chunk_sel <= 64'd1;  // Switch to up_proj
                        end else begin
                            // Just finished up_proj for this hidden dim
                            up_buffer[hidden_cnt*ACT_WIDTH +: ACT_WIDTH] <= acc_saturated;
                            mac_chunk_sel <= 64'd0;  // Back to gate
                            
                            if (hidden_cnt == HIDDEN_DIM - 1) begin
                                hidden_cnt <= {HID_ADDR_WIDTH{1'b0}};
                                state <= STAGE_SILU_MUL;
                            end else begin
                                hidden_cnt <= hidden_cnt + 1'b1;
                            end
                        end
                    end else begin
                        mac_chunk_idx <= mac_chunk_idx + 1'b1;
                    end
                end
                
                STAGE_SILU_MUL: begin
                    // Apply SiLU to gate and multiply with up
                    gate_val <= gate_buffer[hidden_cnt*ACT_WIDTH +: ACT_WIDTH];
                    up_val <= up_buffer[hidden_cnt*ACT_WIDTH +: ACT_WIDTH];
                    
                    // Store result from previous cycle (if not first)
                    if (hidden_cnt > 0) begin
                        gated_buffer[(hidden_cnt-1)*ACT_WIDTH +: ACT_WIDTH] <= gated_val;
                    end
                    
                    if (hidden_cnt == HIDDEN_DIM - 1) begin
                        state <= STAGE_DOWN_PROJ;
                        hidden_cnt <= {HID_ADDR_WIDTH{1'b0}};
                        input_cnt <= {IN_ADDR_WIDTH{1'b0}};
                        mac_chunk_idx <= 6'd0;
                        acc_clear <= 1'b1;
                    end else begin
                        hidden_cnt <= hidden_cnt + 1'b1;
                    end
                end
                
                STAGE_DOWN_PROJ: begin
                    // Store last gated value
                    if (hidden_cnt == 0 && mac_chunk_idx == 0 && acc_clear) begin
                        gated_buffer[(HIDDEN_DIM-1)*ACT_WIDTH +: ACT_WIDTH] <= gated_val;
                    end
                    
                    acc_clear <= 1'b0;
                    
                    if (mac_chunk_idx == (HIDDEN_DIM + 63) / 64 - 1) begin
                        // Finished one output dimension
                        mac_chunk_idx <= 6'd0;
                        acc_clear <= 1'b1;
                        output_buffer[input_cnt*ACT_WIDTH +: ACT_WIDTH] <= acc_saturated;
                        
                        if (input_cnt == IN_DIM - 1) begin
                            state <= STAGE_OUTPUT;
                        end else begin
                            input_cnt <= input_cnt + 1'b1;
                        end
                    end else begin
                        mac_chunk_idx <= mac_chunk_idx + 1'b1;
                    end
                end
                
                STAGE_OUTPUT: begin
                    output_valid <= 1'b1;
                    if (token_ready_out) begin
                        output_valid <= 1'b0;
                        input_valid <= 1'b0;
                        state <= STAGE_IDLE;
                    end
                end
                
                default: begin
                    state <= STAGE_IDLE;
                end
            endcase
        end
    end
    
    // =========================================================================
    // Output Assignments
    // =========================================================================
    assign token_ready_in = (state == STAGE_IDLE);
    assign token_valid_out = output_valid;
    assign token_data_out = output_buffer;
    assign token_last_out = token_last_reg;
    assign busy = (state != STAGE_IDLE);
    
    // Weight memory interface
    assign gate_weight_rd_en = (state == STAGE_GATE_UP) && !mac_chunk_sel[0];
    assign gate_weight_addr = input_cnt;
    assign up_weight_rd_en = (state == STAGE_GATE_UP) && mac_chunk_sel[0];
    assign up_weight_addr = input_cnt;
    assign down_weight_rd_en = (state == STAGE_DOWN_PROJ);
    assign down_weight_addr = hidden_cnt;

endmodule

`default_nettype wire
