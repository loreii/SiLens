// =============================================================================
// MLP Block Edge (Gated MLP / SwiGLU style) - Level 1 Synthesis Block
// =============================================================================
// Edge-optimized Gated MLP for NanoViT transformers:
//   out = down_proj(silu(gate_proj(x)) * up_proj(x))
//
// This is the SwiGLU variant where:
//   - gate_proj and up_proj both project input to hidden dimension
//   - SiLU activation applied to gate path (PWL approximation)
//   - Element-wise multiply gates the up_proj path
//   - down_proj reduces back to input dimension
//
// Edge specifications:
// - IN_DIM = 192 (NanoViT hidden dimension)
// - HIDDEN_DIM = 384 (2× expansion for edge, not 2.67× like VLM)
// - Target: 200MHz (5ns clock period)
// - Target area: ~2mm² (1414µm × 1414µm) on SKY130
//
// Uses ternary weights (2-bit encoded: 00=0, 01=+1, 10=-1)
// Streaming token interface for transformer integration
// Uses generate blocks for Verilator compatibility
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module mlp_block_edge #(
    parameter IN_DIM      = 192,      // Input dimension (NanoViT hidden)
    parameter HIDDEN_DIM  = 384,      // Hidden dimension (2× expansion for edge)
    parameter ACT_WIDTH   = 8,        // Activation bit width
    parameter ACC_WIDTH   = 24        // Accumulator width for MAC operations
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
    localparam IN_ADDR_WIDTH   = $clog2(IN_DIM);    // 8 bits for 192
    localparam HID_ADDR_WIDTH  = $clog2(HIDDEN_DIM); // 9 bits for 384
    
    // MAC chunk size (64-wide parallel MAC)
    localparam MAC_WIDTH = 64;
    localparam IN_CHUNKS = (IN_DIM + MAC_WIDTH - 1) / MAC_WIDTH;    // 3 chunks for 192
    localparam HID_CHUNKS = (HIDDEN_DIM + MAC_WIDTH - 1) / MAC_WIDTH; // 6 chunks for 384
    localparam CHUNK_BITS = $clog2(IN_CHUNKS > HID_CHUNKS ? IN_CHUNKS : HID_CHUNKS);
    
    // Pipeline stages
    localparam STAGE_IDLE       = 3'd0;
    localparam STAGE_GATE_UP    = 3'd1;  // Compute gate_proj and up_proj
    localparam STAGE_SILU_MUL   = 3'd2;  // SiLU(gate) * up
    localparam STAGE_DOWN_PROJ  = 3'd3;  // down_proj
    localparam STAGE_OUTPUT     = 3'd4;  // Output token
    
    // =========================================================================
    // State Machine Registers
    // =========================================================================
    reg [2:0]                   state;
    reg [IN_ADDR_WIDTH-1:0]     input_cnt;
    reg [HID_ADDR_WIDTH-1:0]    hidden_cnt;
    reg                         token_last_reg;
    
    // =========================================================================
    // Input Token Buffer
    // =========================================================================
    reg [IN_DIM*ACT_WIDTH-1:0]  input_buffer;
    reg                         input_valid;
    
    // =========================================================================
    // Intermediate Buffers (using generate for Verilator compatibility)
    // =========================================================================
    // Gate projection result (before SiLU)
    reg signed [ACT_WIDTH-1:0] gate_buffer [0:HIDDEN_DIM-1];
    // Up projection result
    reg signed [ACT_WIDTH-1:0] up_buffer [0:HIDDEN_DIM-1];
    // After SiLU and multiply
    reg signed [ACT_WIDTH-1:0] gated_buffer [0:HIDDEN_DIM-1];
    // Output buffer
    reg signed [ACT_WIDTH-1:0] output_buffer [0:IN_DIM-1];
    reg                        output_valid;
    
    // MAC control
    reg [CHUNK_BITS-1:0]       mac_chunk_idx;
    reg                        mac_phase;   // 0 = gate, 1 = up
    reg                        acc_clear;
    
    // =========================================================================
    // SiLU Approximation (PWL function - synthesizable, no LUT)
    // =========================================================================
    // SiLU(x) = x * sigmoid(x)
    // Using piecewise linear approximation
    
    localparam signed [ACT_WIDTH-1:0] ONE_FP = 8'sd16;     // 1.0 in Q4.4
    localparam signed [ACT_WIDTH-1:0] BP_N2  = -8'sd32;    // -2.0
    localparam signed [ACT_WIDTH-1:0] BP_N1  = -8'sd16;    // -1.0
    localparam signed [ACT_WIDTH-1:0] BP_P1  = 8'sd16;     // +1.0
    localparam signed [ACT_WIDTH-1:0] BP_P2  = 8'sd32;     // +2.0
    
    // PWL sigmoid approximation function
    function automatic signed [ACT_WIDTH-1:0] pwl_sigmoid;
        input signed [ACT_WIDTH-1:0] x;
        reg signed [2*ACT_WIDTH-1:0] interp;
        begin
            if (x <= -8'sd64) begin
                pwl_sigmoid = 8'sd0;
            end else if (x < BP_N2) begin
                interp = (x + 8'sd64) >>> 4;
                pwl_sigmoid = interp[ACT_WIDTH-1:0];
            end else if (x < BP_N1) begin
                interp = (x - BP_N2) >>> 3;
                pwl_sigmoid = 8'sd2 + interp[ACT_WIDTH-1:0];
            end else if (x < 0) begin
                interp = (x - BP_N1) >>> 2;
                pwl_sigmoid = 8'sd4 + interp[ACT_WIDTH-1:0];
            end else if (x < BP_P1) begin
                interp = x >>> 2;
                pwl_sigmoid = 8'sd8 + interp[ACT_WIDTH-1:0];
            end else if (x < BP_P2) begin
                interp = (x - BP_P1) >>> 3;
                pwl_sigmoid = 8'sd12 + interp[ACT_WIDTH-1:0];
            end else if (x <= 8'sd63) begin
                interp = (x - BP_P2) >>> 4;
                pwl_sigmoid = 8'sd14 + interp[ACT_WIDTH-1:0];
            end else begin
                pwl_sigmoid = 8'sd16;
            end
        end
    endfunction
    
    // SiLU = x * sigmoid(x)
    function automatic signed [ACT_WIDTH-1:0] silu_func;
        input signed [ACT_WIDTH-1:0] x;
        reg signed [ACT_WIDTH-1:0] sig;
        reg signed [2*ACT_WIDTH-1:0] prod;
        begin
            sig = pwl_sigmoid(x);
            prod = x * sig;
            // Normalize by 16 (Q4.4 format) and saturate
            if (prod > $signed(16'sd2047))
                silu_func = 8'sd127;
            else if (prod < $signed(-16'sd2048))
                silu_func = -8'sd128;
            else
                silu_func = prod[ACT_WIDTH+3:4];
        end
    endfunction
    
    // =========================================================================
    // Ternary MAC Unit - 64-wide parallel MAC
    // =========================================================================
    wire signed [ACT_WIDTH-1:0] mac_activations [0:MAC_WIDTH-1];
    wire [1:0]                  mac_weights_gate [0:MAC_WIDTH-1];
    wire [1:0]                  mac_weights_up [0:MAC_WIDTH-1];
    wire [1:0]                  mac_weights_down [0:MAC_WIDTH-1];
    
    // Extract 64 activations for current MAC operation
    genvar gi;
    generate
        for (gi = 0; gi < MAC_WIDTH; gi = gi + 1) begin : gen_mac_act_extract
            wire [IN_ADDR_WIDTH:0] act_idx;
            assign act_idx = mac_chunk_idx * MAC_WIDTH + gi;
            
            // Safely extract activation (zero if out of bounds)
            wire in_bounds;
            assign in_bounds = (act_idx < IN_DIM);
            assign mac_activations[gi] = in_bounds ? 
                $signed(input_buffer[act_idx*ACT_WIDTH +: ACT_WIDTH]) : 
                {ACT_WIDTH{1'b0}};
        end
    endgenerate
    
    // Extract 64 weights for gate projection
    generate
        for (gi = 0; gi < MAC_WIDTH; gi = gi + 1) begin : gen_mac_weight_gate
            wire [HID_ADDR_WIDTH:0] weight_offset;
            assign weight_offset = hidden_cnt;
            assign mac_weights_gate[gi] = gate_weight_data[weight_offset*2 +: 2];
        end
    endgenerate
    
    // Extract 64 weights for up projection
    generate
        for (gi = 0; gi < MAC_WIDTH; gi = gi + 1) begin : gen_mac_weight_up
            wire [HID_ADDR_WIDTH:0] weight_offset;
            assign weight_offset = hidden_cnt;
            assign mac_weights_up[gi] = up_weight_data[weight_offset*2 +: 2];
        end
    endgenerate
    
    // Extract 64 weights for down projection
    generate
        for (gi = 0; gi < MAC_WIDTH; gi = gi + 1) begin : gen_mac_weight_down
            wire [IN_ADDR_WIDTH:0] weight_offset;
            assign weight_offset = input_cnt;
            assign mac_weights_down[gi] = down_weight_data[weight_offset*2 +: 2];
        end
    endgenerate
    
    // Select current weights based on state
    wire [1:0] mac_weights [0:MAC_WIDTH-1];
    generate
        for (gi = 0; gi < MAC_WIDTH; gi = gi + 1) begin : gen_weight_select
            assign mac_weights[gi] = (state == STAGE_GATE_UP && !mac_phase) ? mac_weights_gate[gi] :
                                     (state == STAGE_GATE_UP &&  mac_phase) ? mac_weights_up[gi] :
                                     mac_weights_down[gi];
        end
    endgenerate
    
    // Parallel ternary multiplication
    wire signed [ACT_WIDTH:0] products [0:MAC_WIDTH-1];
    generate
        for (gi = 0; gi < MAC_WIDTH; gi = gi + 1) begin : gen_ternary_mult
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
        for (gi = 0; gi < 32; gi = gi + 1) begin : gen_l1_add
            assign sum_l1[gi] = $signed(products[gi*2]) + $signed(products[gi*2+1]);
        end
    endgenerate
    
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : gen_l2_add
            assign sum_l2[gi] = $signed(sum_l1[gi*2]) + $signed(sum_l1[gi*2+1]);
        end
    endgenerate
    
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : gen_l3_add
            assign sum_l3[gi] = $signed(sum_l2[gi*2]) + $signed(sum_l2[gi*2+1]);
        end
    endgenerate
    
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : gen_l4_add
            assign sum_l4[gi] = $signed(sum_l3[gi*2]) + $signed(sum_l3[gi*2+1]);
        end
    endgenerate
    
    assign sum_l5[0] = $signed(sum_l4[0]) + $signed(sum_l4[1]);
    assign sum_l5[1] = $signed(sum_l4[2]) + $signed(sum_l4[3]);
    assign sum_l6 = $signed(sum_l5[0]) + $signed(sum_l5[1]);
    
    // MAC result with proper sign extension
    wire signed [ACC_WIDTH-1:0] mac_result;
    assign mac_result = {{(ACC_WIDTH-ACT_WIDTH-7){sum_l6[ACT_WIDTH+6]}}, sum_l6};
    
    // =========================================================================
    // Accumulator for multi-chunk MAC
    // =========================================================================
    reg signed [ACC_WIDTH-1:0] accumulator;
    
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
    assign acc_saturated = (accumulator > $signed(24'sd127)) ? 8'sd127 :
                           (accumulator < $signed(-24'sd128)) ? -8'sd128 :
                           accumulator[ACT_WIDTH-1:0];
    
    // =========================================================================
    // SiLU Application
    // =========================================================================
    wire signed [ACT_WIDTH-1:0] silu_out;
    assign silu_out = silu_func(acc_saturated);
    
    // =========================================================================
    // Gating Multiply
    // =========================================================================
    reg signed [ACT_WIDTH-1:0] current_gate_val;
    reg signed [ACT_WIDTH-1:0] current_up_val;
    wire signed [2*ACT_WIDTH-1:0] gated_product;
    wire signed [ACT_WIDTH-1:0] gated_val;
    
    assign gated_product = silu_out * current_up_val;
    // Truncate and saturate (Q4.4 * Q4.4 = Q8.8, need Q4.4)
    assign gated_val = (gated_product > $signed(16'sd2047)) ? 8'sd127 :
                       (gated_product < $signed(-16'sd2048)) ? -8'sd128 :
                       gated_product[2*ACT_WIDTH-2:ACT_WIDTH-1];
    
    // =========================================================================
    // Buffer updates using generate blocks (Verilator compatible)
    // =========================================================================
    
    // Gate buffer updates
    generate
        for (gi = 0; gi < HIDDEN_DIM; gi = gi + 1) begin : gen_gate_buffer
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    gate_buffer[gi] <= {ACT_WIDTH{1'b0}};
                end else if (state == STAGE_GATE_UP && !mac_phase && 
                            mac_chunk_idx == IN_CHUNKS - 1 && hidden_cnt == gi) begin
                    gate_buffer[gi] <= acc_saturated;
                end
            end
        end
    endgenerate
    
    // Up buffer updates
    generate
        for (gi = 0; gi < HIDDEN_DIM; gi = gi + 1) begin : gen_up_buffer
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    up_buffer[gi] <= {ACT_WIDTH{1'b0}};
                end else if (state == STAGE_GATE_UP && mac_phase && 
                            mac_chunk_idx == IN_CHUNKS - 1 && hidden_cnt == gi) begin
                    up_buffer[gi] <= acc_saturated;
                end
            end
        end
    endgenerate
    
    // Gated buffer updates (after SiLU * up)
    generate
        for (gi = 0; gi < HIDDEN_DIM; gi = gi + 1) begin : gen_gated_buffer
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    gated_buffer[gi] <= {ACT_WIDTH{1'b0}};
                end else if (state == STAGE_SILU_MUL && hidden_cnt == gi + 1) begin
                    // Store result with one cycle delay
                    gated_buffer[gi] <= gated_val;
                end else if (state == STAGE_DOWN_PROJ && hidden_cnt == 0 && 
                            mac_chunk_idx == 0 && acc_clear && gi == HIDDEN_DIM - 1) begin
                    // Store last gated value
                    gated_buffer[gi] <= gated_val;
                end
            end
        end
    endgenerate
    
    // Output buffer updates
    generate
        for (gi = 0; gi < IN_DIM; gi = gi + 1) begin : gen_output_buffer
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    output_buffer[gi] <= {ACT_WIDTH{1'b0}};
                end else if (state == STAGE_DOWN_PROJ && 
                            mac_chunk_idx == HID_CHUNKS - 1 && input_cnt == gi) begin
                    output_buffer[gi] <= acc_saturated;
                end
            end
        end
    endgenerate
    
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
            mac_chunk_idx <= {CHUNK_BITS{1'b0}};
            mac_phase <= 1'b0;
            acc_clear <= 1'b1;
            current_gate_val <= {ACT_WIDTH{1'b0}};
            current_up_val <= {ACT_WIDTH{1'b0}};
            input_buffer <= {(IN_DIM*ACT_WIDTH){1'b0}};
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
                        mac_chunk_idx <= {CHUNK_BITS{1'b0}};
                        mac_phase <= 1'b0;
                        acc_clear <= 1'b1;
                    end
                end
                
                STAGE_GATE_UP: begin
                    // Process all input dimensions in chunks of 64
                    acc_clear <= 1'b0;
                    
                    if (mac_chunk_idx == IN_CHUNKS - 1) begin
                        // Finished one hidden dimension
                        mac_chunk_idx <= {CHUNK_BITS{1'b0}};
                        acc_clear <= 1'b1;
                        
                        if (!mac_phase) begin
                            // Just finished gate_proj for this hidden dim
                            mac_phase <= 1'b1;  // Switch to up_proj
                        end else begin
                            // Just finished up_proj for this hidden dim
                            mac_phase <= 1'b0;  // Back to gate
                            
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
                    current_gate_val <= gate_buffer[hidden_cnt];
                    current_up_val <= up_buffer[hidden_cnt];
                    
                    if (hidden_cnt == HIDDEN_DIM - 1) begin
                        state <= STAGE_DOWN_PROJ;
                        hidden_cnt <= {HID_ADDR_WIDTH{1'b0}};
                        input_cnt <= {IN_ADDR_WIDTH{1'b0}};
                        mac_chunk_idx <= {CHUNK_BITS{1'b0}};
                        acc_clear <= 1'b1;
                    end else begin
                        hidden_cnt <= hidden_cnt + 1'b1;
                    end
                end
                
                STAGE_DOWN_PROJ: begin
                    acc_clear <= 1'b0;
                    
                    if (mac_chunk_idx == HID_CHUNKS - 1) begin
                        // Finished one output dimension
                        mac_chunk_idx <= {CHUNK_BITS{1'b0}};
                        acc_clear <= 1'b1;
                        
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
    // Output Data Packing (using generate for Verilator compatibility)
    // =========================================================================
    wire [IN_DIM*ACT_WIDTH-1:0] token_data_out_packed;
    generate
        for (gi = 0; gi < IN_DIM; gi = gi + 1) begin : gen_output_pack
            assign token_data_out_packed[gi*ACT_WIDTH +: ACT_WIDTH] = output_buffer[gi];
        end
    endgenerate
    
    // =========================================================================
    // Output Assignments
    // =========================================================================
    assign token_ready_in = (state == STAGE_IDLE);
    assign token_valid_out = output_valid;
    assign token_data_out = token_data_out_packed;
    assign token_last_out = token_last_reg;
    assign busy = (state != STAGE_IDLE);
    
    // Weight memory interface
    assign gate_weight_rd_en = (state == STAGE_GATE_UP) && !mac_phase;
    assign gate_weight_addr = input_cnt;
    assign up_weight_rd_en = (state == STAGE_GATE_UP) && mac_phase;
    assign up_weight_addr = input_cnt;
    assign down_weight_rd_en = (state == STAGE_DOWN_PROJ);
    assign down_weight_addr = hidden_cnt;

endmodule

`default_nettype wire
