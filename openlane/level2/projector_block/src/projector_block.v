// =============================================================================
// Projector Block (Vision-to-LLM Embedding Projection) - Level 2 Synthesis Block
// =============================================================================
// Projects vision encoder output embeddings to LLM input space.
// Part of SiLens hierarchical synthesis for multimodal LLM.
//
// Architecture: 2-layer MLP with GELU activation
//   vision_out (768) → Linear1 → GELU → Linear2 → llm_input (576)
//   Hidden dimension: 1152 (1.5× input dimension)
//
// Target: ~10mm² (3200µm × 3200µm) on SKY130
// Reuse: 1× (single projector between vision encoder and LLM)
//
// Processing: Sequential patch processing (one patch at a time)
//   - 64-wide SIMD using ternary MAC arrays
//   - Multiple cycles per dimension for area efficiency
//   - Streaming pipeline with valid/ready backpressure
//
// Ternary weights: 2-bit encoding (00=0, 01=+1, 10=-1)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module projector_block #(
    parameter VIS_DIM     = 768,      // Vision embedding dimension (input)
    parameter LLM_DIM     = 576,      // LLM embedding dimension (output)
    parameter HIDDEN_DIM  = 1152,     // Intermediate hidden dimension
    parameter ACT_WIDTH   = 8,        // Activation bit width
    parameter ACC_WIDTH   = 24,       // Accumulator width for MAC operations
    parameter SIMD_WIDTH  = 64,       // SIMD width (matches ternary_mac_array_64)
    parameter PATCH_CNT_WIDTH = 10    // Supports up to 1024 patches
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // =========================================================================
    // Patch Streaming Input (from Vision Encoder)
    // =========================================================================
    input  wire                         patch_valid_in,
    output wire                         patch_ready_in,
    input  wire [VIS_DIM*ACT_WIDTH-1:0] patch_data_in,      // 768×8 = 6144 bits
    input  wire                         patch_last_in,      // Last patch in sequence
    
    // =========================================================================
    // Token Streaming Output (to LLM Decoder)
    // =========================================================================
    output wire                         token_valid_out,
    input  wire                         token_ready_out,
    output wire [LLM_DIM*ACT_WIDTH-1:0] token_data_out,     // 576×8 = 4608 bits
    output wire                         token_last_out,
    
    // =========================================================================
    // Weight Memory Interface - First Linear Layer (768 → 1152)
    // =========================================================================
    output wire                         w1_rd_en,
    output wire [$clog2(VIS_DIM)-1:0]   w1_row_addr,        // Row of input being processed
    input  wire [HIDDEN_DIM*2-1:0]      w1_data,            // 1152 ternary weights = 2304 bits
    
    // =========================================================================
    // Weight Memory Interface - Second Linear Layer (1152 → 576)
    // =========================================================================
    output wire                         w2_rd_en,
    output wire [$clog2(HIDDEN_DIM)-1:0] w2_row_addr,       // Row of hidden being processed
    input  wire [LLM_DIM*2-1:0]         w2_data,            // 576 ternary weights = 1152 bits
    
    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire [PATCH_CNT_WIDTH-1:0]   num_patches,        // Total patches to process
    output wire                         busy,
    output wire [PATCH_CNT_WIDTH-1:0]   patches_processed
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam VIS_ADDR_WIDTH = $clog2(VIS_DIM);      // 10 bits for 768
    localparam HID_ADDR_WIDTH = $clog2(HIDDEN_DIM);   // 11 bits for 1152
    localparam LLM_ADDR_WIDTH = $clog2(LLM_DIM);      // 10 bits for 576
    
    // Cycles to process each dimension with 64-wide SIMD
    localparam VIS_CHUNKS = (VIS_DIM + SIMD_WIDTH - 1) / SIMD_WIDTH;   // 12 chunks
    localparam HID_CHUNKS = (HIDDEN_DIM + SIMD_WIDTH - 1) / SIMD_WIDTH; // 18 chunks
    localparam CHUNK_ADDR_WIDTH = 5;  // Max 18 chunks
    
    // State machine states
    localparam [2:0] ST_IDLE        = 3'd0;
    localparam [2:0] ST_LOAD_PATCH  = 3'd1;
    localparam [2:0] ST_LINEAR1     = 3'd2;
    localparam [2:0] ST_GELU        = 3'd3;
    localparam [2:0] ST_LINEAR2     = 3'd4;
    localparam [2:0] ST_OUTPUT      = 3'd5;
    
    // =========================================================================
    // GELU Approximation LUT
    // =========================================================================
    // GELU(x) = x * Φ(x) ≈ x * sigmoid(1.702 * x) (fast approximation)
    // Alternative: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    // We use piecewise linear approximation for hardware efficiency
    //
    // GELU characteristics:
    //   - For x < -3: GELU ≈ 0
    //   - For x > 3:  GELU ≈ x
    //   - Around 0:   smooth transition, GELU(0) = 0
    //   - Minimum around x ≈ -0.75 where GELU ≈ -0.17
    
    localparam GELU_LUT_DEPTH = 256;
    reg signed [ACT_WIDTH-1:0] gelu_lut [0:GELU_LUT_DEPTH-1];
    
    // Initialize GELU LUT with piecewise linear approximation
    integer lut_i;
    initial begin
        for (lut_i = 0; lut_i < GELU_LUT_DEPTH; lut_i = lut_i + 1) begin
            // Map index [0,255] to signed input [-128, 127]
            // Scale: input represents fixed-point with ~4 fractional bits
            // So range is approximately [-8, 8) in real values
            if (lut_i < 64) begin
                // Very negative: GELU ≈ 0
                gelu_lut[lut_i] = 8'sd0;
            end else if (lut_i < 96) begin
                // Negative transition region [-4, -2): small negative dip
                // GELU has a minimum around -0.75 ≈ -12 in fixed-point
                gelu_lut[lut_i] = -8'sd2 + ($signed(lut_i) - 8'sd80) >>> 4;
            end else if (lut_i < 128) begin
                // Near-zero negative [-2, 0): transition from negative to 0
                gelu_lut[lut_i] = ($signed(lut_i) - 8'sd128) >>> 2;
            end else if (lut_i < 160) begin
                // Near-zero positive [0, 2): gradual ramp up
                gelu_lut[lut_i] = ($signed(lut_i) - 8'sd128) >>> 1;
            end else if (lut_i < 192) begin
                // Positive transition [2, 4): steeper ramp
                gelu_lut[lut_i] = ($signed(lut_i) - 8'sd128) - (($signed(lut_i) - 8'sd160) >>> 3);
            end else begin
                // Large positive [4, 8): nearly identity
                gelu_lut[lut_i] = $signed(lut_i) - 8'sd128;
            end
        end
    end
    
    // =========================================================================
    // State Machine Registers
    // =========================================================================
    reg [2:0]                       state;
    reg [VIS_ADDR_WIDTH-1:0]        vis_row_cnt;
    reg [HID_ADDR_WIDTH-1:0]        hid_cnt;
    reg [LLM_ADDR_WIDTH-1:0]        llm_cnt;
    reg [CHUNK_ADDR_WIDTH-1:0]      chunk_cnt;
    reg [PATCH_CNT_WIDTH-1:0]       patch_cnt;
    reg                             is_last_patch;
    
    // =========================================================================
    // Data Buffers
    // =========================================================================
    // Input patch buffer (768 × 8 bits)
    reg [VIS_DIM*ACT_WIDTH-1:0]     input_buffer;
    
    // Hidden layer buffer after Linear1 + GELU (1152 × 8 bits)
    reg [HIDDEN_DIM*ACT_WIDTH-1:0]  hidden_buffer;
    
    // Output buffer (576 × 8 bits)
    reg [LLM_DIM*ACT_WIDTH-1:0]     output_buffer;
    reg                             output_valid_reg;
    
    // =========================================================================
    // MAC Accumulator
    // =========================================================================
    reg signed [ACC_WIDTH-1:0]      accumulator;
    reg                             acc_clear;
    wire signed [ACC_WIDTH-1:0]     mac_partial_sum;
    
    // =========================================================================
    // 64-wide Ternary MAC Unit
    // =========================================================================
    // Extract 64 activations based on current chunk
    wire signed [ACT_WIDTH-1:0] mac_activations [0:SIMD_WIDTH-1];
    wire [1:0]                  mac_weights     [0:SIMD_WIDTH-1];
    
    // Select activation source based on current stage
    wire [VIS_DIM*ACT_WIDTH-1:0]    linear1_input = input_buffer;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] linear2_input = hidden_buffer;
    
    genvar gi;
    generate
        for (gi = 0; gi < SIMD_WIDTH; gi = gi + 1) begin : mac_input_mux
            wire [HID_ADDR_WIDTH-1:0] act_idx_l1 = chunk_cnt * SIMD_WIDTH + gi;
            wire [HID_ADDR_WIDTH-1:0] act_idx_l2 = chunk_cnt * SIMD_WIDTH + gi;
            
            // Activation selection (stage-dependent)
            wire in_bounds_l1 = (act_idx_l1 < VIS_DIM);
            wire in_bounds_l2 = (act_idx_l2 < HIDDEN_DIM);
            
            // For Linear1: use input_buffer (768 elements)
            // For Linear2: use hidden_buffer (1152 elements)
            assign mac_activations[gi] = (state == ST_LINEAR1) ? 
                (in_bounds_l1 ? $signed(linear1_input[act_idx_l1*ACT_WIDTH +: ACT_WIDTH]) : {ACT_WIDTH{1'b0}}) :
                (in_bounds_l2 ? $signed(linear2_input[act_idx_l2*ACT_WIDTH +: ACT_WIDTH]) : {ACT_WIDTH{1'b0}});
            
            // Weight extraction from memory data
            // For Linear1: w1_data has HIDDEN_DIM weights
            // For Linear2: w2_data has LLM_DIM weights
            wire [HID_ADDR_WIDTH-1:0] w_offset = gi;
            assign mac_weights[gi] = (state == ST_LINEAR1) ?
                w1_data[(hid_cnt % SIMD_WIDTH + gi) * 2 +: 2] :
                w2_data[(llm_cnt % SIMD_WIDTH + gi) * 2 +: 2];
        end
    endgenerate
    
    // =========================================================================
    // Ternary Multiply (no actual multiplier - just conditional negate/zero)
    // =========================================================================
    wire signed [ACT_WIDTH:0] products [0:SIMD_WIDTH-1];
    generate
        for (gi = 0; gi < SIMD_WIDTH; gi = gi + 1) begin : ternary_mult
            // 00 = 0, 01 = +1, 10 = -1, 11 = 0 (reserved)
            assign products[gi] = (mac_weights[gi] == 2'b01) ?  {mac_activations[gi][ACT_WIDTH-1], mac_activations[gi]} :
                                  (mac_weights[gi] == 2'b10) ? -{mac_activations[gi][ACT_WIDTH-1], mac_activations[gi]} :
                                                                {(ACT_WIDTH+1){1'b0}};
        end
    endgenerate
    
    // =========================================================================
    // 6-Level Reduction Tree (64 → 1)
    // =========================================================================
    // Level 1: 64 → 32 (9-bit + 9-bit = 10-bit)
    wire signed [ACT_WIDTH+1:0] sum_l1 [0:31];
    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : l1_reduce
            assign sum_l1[gi] = $signed(products[gi*2]) + $signed(products[gi*2+1]);
        end
    endgenerate
    
    // Level 2: 32 → 16 (10-bit + 10-bit = 11-bit)
    wire signed [ACT_WIDTH+2:0] sum_l2 [0:15];
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : l2_reduce
            assign sum_l2[gi] = $signed(sum_l1[gi*2]) + $signed(sum_l1[gi*2+1]);
        end
    endgenerate
    
    // Level 3: 16 → 8 (11-bit + 11-bit = 12-bit)
    wire signed [ACT_WIDTH+3:0] sum_l3 [0:7];
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : l3_reduce
            assign sum_l3[gi] = $signed(sum_l2[gi*2]) + $signed(sum_l2[gi*2+1]);
        end
    endgenerate
    
    // Level 4: 8 → 4 (12-bit + 12-bit = 13-bit)
    wire signed [ACT_WIDTH+4:0] sum_l4 [0:3];
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : l4_reduce
            assign sum_l4[gi] = $signed(sum_l3[gi*2]) + $signed(sum_l3[gi*2+1]);
        end
    endgenerate
    
    // Level 5: 4 → 2 (13-bit + 13-bit = 14-bit)
    wire signed [ACT_WIDTH+5:0] sum_l5 [0:1];
    assign sum_l5[0] = $signed(sum_l4[0]) + $signed(sum_l4[1]);
    assign sum_l5[1] = $signed(sum_l4[2]) + $signed(sum_l4[3]);
    
    // Level 6: 2 → 1 (14-bit + 14-bit = 15-bit)
    wire signed [ACT_WIDTH+6:0] sum_total;
    assign sum_total = $signed(sum_l5[0]) + $signed(sum_l5[1]);
    
    // Sign-extend to accumulator width
    assign mac_partial_sum = {{(ACC_WIDTH-ACT_WIDTH-7){sum_total[ACT_WIDTH+6]}}, sum_total};
    
    // =========================================================================
    // Accumulator Logic
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= {ACC_WIDTH{1'b0}};
        end else if (acc_clear) begin
            accumulator <= mac_partial_sum;
        end else begin
            accumulator <= accumulator + mac_partial_sum;
        end
    end
    
    // =========================================================================
    // Saturation Logic
    // =========================================================================
    wire signed [ACT_WIDTH-1:0] acc_saturated;
    localparam signed [ACC_WIDTH-1:0] MAX_VAL = {{(ACC_WIDTH-ACT_WIDTH){1'b0}}, {(ACT_WIDTH-1){1'b1}}};
    localparam signed [ACC_WIDTH-1:0] MIN_VAL = {{(ACC_WIDTH-ACT_WIDTH+1){1'b1}}, {(ACT_WIDTH-1){1'b0}}};
    
    assign acc_saturated = (accumulator > MAX_VAL) ? $signed({1'b0, {(ACT_WIDTH-1){1'b1}}}) :
                           (accumulator < MIN_VAL) ? $signed({1'b1, {(ACT_WIDTH-1){1'b0}}}) :
                           accumulator[ACT_WIDTH-1:0];
    
    // =========================================================================
    // GELU Application
    // =========================================================================
    wire [7:0] gelu_addr = acc_saturated + 8'd128;  // Map signed to unsigned index
    wire signed [ACT_WIDTH-1:0] gelu_out = gelu_lut[gelu_addr];
    
    // =========================================================================
    // Main State Machine
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            vis_row_cnt <= {VIS_ADDR_WIDTH{1'b0}};
            hid_cnt <= {HID_ADDR_WIDTH{1'b0}};
            llm_cnt <= {LLM_ADDR_WIDTH{1'b0}};
            chunk_cnt <= {CHUNK_ADDR_WIDTH{1'b0}};
            patch_cnt <= {PATCH_CNT_WIDTH{1'b0}};
            is_last_patch <= 1'b0;
            input_buffer <= {(VIS_DIM*ACT_WIDTH){1'b0}};
            hidden_buffer <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            output_buffer <= {(LLM_DIM*ACT_WIDTH){1'b0}};
            output_valid_reg <= 1'b0;
            acc_clear <= 1'b1;
        end else begin
            case (state)
                // ---------------------------------------------------------
                // IDLE: Wait for input patch
                // ---------------------------------------------------------
                ST_IDLE: begin
                    output_valid_reg <= 1'b0;
                    if (patch_valid_in && patch_ready_in) begin
                        input_buffer <= patch_data_in;
                        is_last_patch <= patch_last_in;
                        state <= ST_LINEAR1;
                        vis_row_cnt <= {VIS_ADDR_WIDTH{1'b0}};
                        hid_cnt <= {HID_ADDR_WIDTH{1'b0}};
                        chunk_cnt <= {CHUNK_ADDR_WIDTH{1'b0}};
                        acc_clear <= 1'b1;
                    end
                end
                
                // ---------------------------------------------------------
                // LINEAR1: Project 768 → 1152
                // Process VIS_DIM inputs to generate HIDDEN_DIM outputs
                // For each output element: sum over all 768 inputs
                // ---------------------------------------------------------
                ST_LINEAR1: begin
                    acc_clear <= 1'b0;
                    
                    // Process 64 inputs at a time
                    if (chunk_cnt == VIS_CHUNKS - 1) begin
                        // Finished accumulating all input chunks for one hidden element
                        chunk_cnt <= {CHUNK_ADDR_WIDTH{1'b0}};
                        acc_clear <= 1'b1;
                        
                        // Store result to hidden buffer (before GELU)
                        hidden_buffer[hid_cnt*ACT_WIDTH +: ACT_WIDTH] <= acc_saturated;
                        
                        if (hid_cnt == HIDDEN_DIM - 1) begin
                            // All hidden elements computed
                            hid_cnt <= {HID_ADDR_WIDTH{1'b0}};
                            state <= ST_GELU;
                        end else begin
                            hid_cnt <= hid_cnt + 1'b1;
                            vis_row_cnt <= {VIS_ADDR_WIDTH{1'b0}};
                        end
                    end else begin
                        chunk_cnt <= chunk_cnt + 1'b1;
                        vis_row_cnt <= vis_row_cnt + SIMD_WIDTH;
                    end
                end
                
                // ---------------------------------------------------------
                // GELU: Apply GELU activation to hidden layer
                // Process one element per cycle through LUT
                // ---------------------------------------------------------
                ST_GELU: begin
                    // Apply GELU to current element
                    hidden_buffer[hid_cnt*ACT_WIDTH +: ACT_WIDTH] <= 
                        gelu_lut[hidden_buffer[hid_cnt*ACT_WIDTH +: ACT_WIDTH] + 8'd128];
                    
                    if (hid_cnt == HIDDEN_DIM - 1) begin
                        // All GELU applications done
                        hid_cnt <= {HID_ADDR_WIDTH{1'b0}};
                        llm_cnt <= {LLM_ADDR_WIDTH{1'b0}};
                        chunk_cnt <= {CHUNK_ADDR_WIDTH{1'b0}};
                        acc_clear <= 1'b1;
                        state <= ST_LINEAR2;
                    end else begin
                        hid_cnt <= hid_cnt + 1'b1;
                    end
                end
                
                // ---------------------------------------------------------
                // LINEAR2: Project 1152 → 576
                // Process HIDDEN_DIM inputs to generate LLM_DIM outputs
                // ---------------------------------------------------------
                ST_LINEAR2: begin
                    acc_clear <= 1'b0;
                    
                    // Process 64 hidden elements at a time
                    if (chunk_cnt == HID_CHUNKS - 1) begin
                        // Finished accumulating all hidden chunks for one output element
                        chunk_cnt <= {CHUNK_ADDR_WIDTH{1'b0}};
                        acc_clear <= 1'b1;
                        
                        // Store result to output buffer
                        output_buffer[llm_cnt*ACT_WIDTH +: ACT_WIDTH] <= acc_saturated;
                        
                        if (llm_cnt == LLM_DIM - 1) begin
                            // All output elements computed
                            llm_cnt <= {LLM_ADDR_WIDTH{1'b0}};
                            state <= ST_OUTPUT;
                        end else begin
                            llm_cnt <= llm_cnt + 1'b1;
                            hid_cnt <= {HID_ADDR_WIDTH{1'b0}};
                        end
                    end else begin
                        chunk_cnt <= chunk_cnt + 1'b1;
                        hid_cnt <= hid_cnt + SIMD_WIDTH;
                    end
                end
                
                // ---------------------------------------------------------
                // OUTPUT: Stream token to LLM decoder
                // ---------------------------------------------------------
                ST_OUTPUT: begin
                    output_valid_reg <= 1'b1;
                    
                    if (token_ready_out) begin
                        output_valid_reg <= 1'b0;
                        patch_cnt <= patch_cnt + 1'b1;
                        state <= ST_IDLE;
                    end
                end
                
                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end
    
    // =========================================================================
    // Output Assignments
    // =========================================================================
    
    // Handshake signals
    assign patch_ready_in  = (state == ST_IDLE);
    assign token_valid_out = output_valid_reg;
    assign token_data_out  = output_buffer;
    assign token_last_out  = is_last_patch;
    
    // Weight memory interface
    // Linear1: Read row vis_row_cnt of weight matrix (768 rows × 1152 columns)
    assign w1_rd_en    = (state == ST_LINEAR1);
    assign w1_row_addr = vis_row_cnt;
    
    // Linear2: Read row hid_cnt of weight matrix (1152 rows × 576 columns)
    assign w2_rd_en    = (state == ST_LINEAR2);
    assign w2_row_addr = hid_cnt;
    
    // Status outputs
    assign busy = (state != ST_IDLE);
    assign patches_processed = patch_cnt;

endmodule

`default_nettype wire
