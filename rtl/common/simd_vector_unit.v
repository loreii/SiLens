// =============================================================================
// SiLens SIMD Vector Processing Unit
// =============================================================================
// Parallel vector operations for ternary neural network inference.
//
// Features:
//   - Configurable vector width (8/16/32 elements)
//   - Ternary MAC operations with accumulation
//   - Element-wise activation functions
//   - Vector reduction (sum, max, min)
//   - Register file with multiple read/write ports
//
// Operations:
//   - VMAC:  Vector ternary multiply-accumulate
//   - VADD:  Vector addition
//   - VMUL:  Vector element-wise multiply  
//   - VMAX:  Vector maximum
//   - VREDUCE: Horizontal reduction
//   - VACT:  Activation function (ReLU, GELU approx)
//
// License: Apache 2.0
// =============================================================================

module simd_vector_unit #(
    parameter VECTOR_WIDTH = 16,        // Elements per vector
    parameter ELEMENT_WIDTH = 8,        // Bits per element
    parameter ACC_WIDTH = 32,           // Accumulator width
    parameter NUM_VREGS = 32,           // Vector registers
    parameter FRAC_BITS = 4
)(
    input  wire         clk,
    input  wire         rst_n,
    
    // Instruction interface
    input  wire [31:0]  instruction,
    input  wire         instr_valid,
    output wire         instr_ready,
    
    // Vector operand interface
    input  wire [VECTOR_WIDTH*ELEMENT_WIDTH-1:0] vec_a_in,
    input  wire [VECTOR_WIDTH*2-1:0]             ternary_b_in,  // 2 bits per ternary
    input  wire [VECTOR_WIDTH*ELEMENT_WIDTH-1:0] vec_c_in,
    input  wire         operands_valid,
    
    // Result output
    output reg  [VECTOR_WIDTH*ELEMENT_WIDTH-1:0] vec_result,
    output reg  [ACC_WIDTH-1:0]                  scalar_result,
    output reg          result_valid,
    input  wire         result_ready,
    
    // Status
    output wire         busy,
    output wire [7:0]   status
);

    // =========================================================================
    // Opcode definitions
    // =========================================================================
    
    localparam OP_NOP     = 4'd0;
    localparam OP_VMAC    = 4'd1;   // Ternary MAC
    localparam OP_VADD    = 4'd2;   // Vector add
    localparam OP_VSUB    = 4'd3;   // Vector subtract
    localparam OP_VMUL    = 4'd4;   // Element-wise multiply
    localparam OP_VMAX    = 4'd5;   // Element-wise max
    localparam OP_VMIN    = 4'd6;   // Element-wise min
    localparam OP_VREDUCE = 4'd7;   // Horizontal sum reduction
    localparam OP_VACT    = 4'd8;   // Activation function
    localparam OP_VLOAD   = 4'd9;   // Load to vector register
    localparam OP_VSTORE  = 4'd10;  // Store from vector register
    localparam OP_VMOV    = 4'd11;  // Move between registers
    localparam OP_VSCALE  = 4'd12;  // Scalar multiply
    localparam OP_VCLAMP  = 4'd13;  // Clamp to range
    
    // Activation function types
    localparam ACT_RELU   = 2'd0;
    localparam ACT_GELU   = 2'd1;
    localparam ACT_SILU   = 2'd2;
    localparam ACT_NONE   = 2'd3;
    
    // =========================================================================
    // Instruction decode
    // =========================================================================
    
    wire [3:0]  opcode;
    wire [4:0]  rd;           // Destination register
    wire [4:0]  rs1;          // Source register 1
    wire [4:0]  rs2;          // Source register 2
    wire [1:0]  act_type;     // Activation type
    wire [7:0]  immediate;    // Immediate value
    wire        use_acc;      // Use accumulator
    
    assign opcode    = instruction[3:0];
    assign rd        = instruction[8:4];
    assign rs1       = instruction[13:9];
    assign rs2       = instruction[18:14];
    assign act_type  = instruction[20:19];
    assign immediate = instruction[28:21];
    assign use_acc   = instruction[29];
    
    // =========================================================================
    // Vector register file
    // =========================================================================
    
    reg [VECTOR_WIDTH*ELEMENT_WIDTH-1:0] vreg_file [NUM_VREGS-1:0];
    
    // Accumulator register per lane
    reg [ACC_WIDTH-1:0] acc_reg [VECTOR_WIDTH-1:0];
    
    // Pipeline registers
    reg [3:0]  pipe_opcode;
    reg [4:0]  pipe_rd;
    reg [1:0]  pipe_act_type;
    reg        pipe_use_acc;
    reg        pipe_valid;
    
    // Working registers
    wire [VECTOR_WIDTH*ELEMENT_WIDTH-1:0] operand_a;
    wire [VECTOR_WIDTH*2-1:0]             operand_b_ternary;
    wire [VECTOR_WIDTH*ELEMENT_WIDTH-1:0] operand_c;
    
    assign operand_a = vreg_file[rs1];
    assign operand_b_ternary = ternary_b_in;
    assign operand_c = vreg_file[rs2];
    
    // =========================================================================
    // Ternary MAC computation (parallel lanes)
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] mac_result [VECTOR_WIDTH-1:0];
    wire signed [ELEMENT_WIDTH-1:0] lane_a [VECTOR_WIDTH-1:0];
    wire signed [1:0] lane_b [VECTOR_WIDTH-1:0];
    
    genvar g;
    generate
        for (g = 0; g < VECTOR_WIDTH; g = g + 1) begin : gen_mac_lanes
            // Extract lane operands
            assign lane_a[g] = operand_a[(g+1)*ELEMENT_WIDTH-1 : g*ELEMENT_WIDTH];
            assign lane_b[g] = operand_b_ternary[(g+1)*2-1 : g*2];
            
            // Ternary multiply: 
            //   01 (+1): result = +a
            //   10 (-1): result = -a
            //   00 (0):  result = 0
            wire signed [ELEMENT_WIDTH:0] ternary_product;
            
            assign ternary_product = (lane_b[g] == 2'b01) ? {{1{lane_a[g][ELEMENT_WIDTH-1]}}, lane_a[g]} :
                                     (lane_b[g] == 2'b10) ? -{{1{lane_a[g][ELEMENT_WIDTH-1]}}, lane_a[g]} :
                                     {(ELEMENT_WIDTH+1){1'b0}};
            
            // Accumulate
            assign mac_result[g] = {{(ACC_WIDTH-ELEMENT_WIDTH-1){ternary_product[ELEMENT_WIDTH]}}, ternary_product} + 
                                   (use_acc ? acc_reg[g] : {ACC_WIDTH{1'b0}});
        end
    endgenerate
    
    // =========================================================================
    // Vector addition/subtraction
    // =========================================================================
    
    wire signed [ELEMENT_WIDTH-1:0] add_result [VECTOR_WIDTH-1:0];
    wire signed [ELEMENT_WIDTH-1:0] sub_result [VECTOR_WIDTH-1:0];
    
    generate
        for (g = 0; g < VECTOR_WIDTH; g = g + 1) begin : gen_add_lanes
            wire signed [ELEMENT_WIDTH-1:0] a_signed;
            wire signed [ELEMENT_WIDTH-1:0] c_signed;
            
            assign a_signed = operand_a[(g+1)*ELEMENT_WIDTH-1 : g*ELEMENT_WIDTH];
            assign c_signed = operand_c[(g+1)*ELEMENT_WIDTH-1 : g*ELEMENT_WIDTH];
            
            assign add_result[g] = a_signed + c_signed;
            assign sub_result[g] = a_signed - c_signed;
        end
    endgenerate

    // =========================================================================
    // Vector max/min
    // =========================================================================
    
    wire signed [ELEMENT_WIDTH-1:0] max_result [VECTOR_WIDTH-1:0];
    wire signed [ELEMENT_WIDTH-1:0] min_result [VECTOR_WIDTH-1:0];
    
    generate
        for (g = 0; g < VECTOR_WIDTH; g = g + 1) begin : gen_minmax_lanes
            wire signed [ELEMENT_WIDTH-1:0] a_s;
            wire signed [ELEMENT_WIDTH-1:0] c_s;
            
            assign a_s = operand_a[(g+1)*ELEMENT_WIDTH-1 : g*ELEMENT_WIDTH];
            assign c_s = operand_c[(g+1)*ELEMENT_WIDTH-1 : g*ELEMENT_WIDTH];
            
            assign max_result[g] = (a_s > c_s) ? a_s : c_s;
            assign min_result[g] = (a_s < c_s) ? a_s : c_s;
        end
    endgenerate
    
    // =========================================================================
    // Horizontal reduction (tree reduction)
    // =========================================================================
    
    wire signed [ACC_WIDTH-1:0] reduce_sum;
    
    // Tree adder for VECTOR_WIDTH elements
    // Assuming VECTOR_WIDTH = 16 for this implementation
    wire signed [ACC_WIDTH-1:0] sum_l1 [7:0];
    wire signed [ACC_WIDTH-1:0] sum_l2 [3:0];
    wire signed [ACC_WIDTH-1:0] sum_l3 [1:0];
    
    generate
        // Level 1: 16 -> 8
        for (g = 0; g < 8; g = g + 1) begin : gen_reduce_l1
            wire signed [ELEMENT_WIDTH-1:0] elem0, elem1;
            assign elem0 = operand_a[(2*g+1)*ELEMENT_WIDTH-1 : 2*g*ELEMENT_WIDTH];
            assign elem1 = operand_a[(2*g+2)*ELEMENT_WIDTH-1 : (2*g+1)*ELEMENT_WIDTH];
            assign sum_l1[g] = {{(ACC_WIDTH-ELEMENT_WIDTH){elem0[ELEMENT_WIDTH-1]}}, elem0} +
                               {{(ACC_WIDTH-ELEMENT_WIDTH){elem1[ELEMENT_WIDTH-1]}}, elem1};
        end
        
        // Level 2: 8 -> 4
        for (g = 0; g < 4; g = g + 1) begin : gen_reduce_l2
            assign sum_l2[g] = sum_l1[2*g] + sum_l1[2*g+1];
        end
        
        // Level 3: 4 -> 2
        for (g = 0; g < 2; g = g + 1) begin : gen_reduce_l3
            assign sum_l3[g] = sum_l2[2*g] + sum_l2[2*g+1];
        end
    endgenerate
    
    // Final reduction
    assign reduce_sum = sum_l3[0] + sum_l3[1];
    
    // =========================================================================
    // Activation functions (per-lane)
    // =========================================================================
    
    wire signed [ELEMENT_WIDTH-1:0] act_result [VECTOR_WIDTH-1:0];
    
    generate
        for (g = 0; g < VECTOR_WIDTH; g = g + 1) begin : gen_activation
            wire signed [ELEMENT_WIDTH-1:0] in_val;
            assign in_val = operand_a[(g+1)*ELEMENT_WIDTH-1 : g*ELEMENT_WIDTH];
            
            // ReLU: max(0, x)
            wire signed [ELEMENT_WIDTH-1:0] relu_out;
            assign relu_out = (in_val[ELEMENT_WIDTH-1]) ? {ELEMENT_WIDTH{1'b0}} : in_val;
            
            // GELU approximation: x * sigmoid(1.702 * x)
            // Simplified: x * (x > 0 ? 1 : 0.5 + 0.25*x) for hardware
            wire signed [ELEMENT_WIDTH-1:0] gelu_out;
            wire signed [ELEMENT_WIDTH+1:0] gelu_temp;
            assign gelu_temp = (in_val[ELEMENT_WIDTH-1]) ? 
                               (in_val >>> 1) + (in_val >>> 2) :  // ~0.75x for negative
                               in_val;                             // x for positive
            assign gelu_out = gelu_temp[ELEMENT_WIDTH-1:0];
            
            // Select activation
            assign act_result[g] = (pipe_act_type == ACT_RELU) ? relu_out :
                                   (pipe_act_type == ACT_GELU) ? gelu_out :
                                   in_val;
        end
    endgenerate

    // =========================================================================
    // Control FSM
    // =========================================================================
    
    localparam FSM_IDLE    = 2'd0;
    localparam FSM_EXECUTE = 2'd1;
    localparam FSM_WRITEBACK = 2'd2;
    
    reg [1:0] fsm_state;
    reg [4:0] exec_cycles;
    
    assign instr_ready = (fsm_state == FSM_IDLE);
    assign busy = (fsm_state != FSM_IDLE);
    
    integer i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            fsm_state <= FSM_IDLE;
            pipe_valid <= 1'b0;
            result_valid <= 1'b0;
            exec_cycles <= 0;
            vec_result <= 0;
            scalar_result <= 0;
            
            pipe_opcode <= OP_NOP;
            pipe_rd <= 0;
            pipe_act_type <= ACT_NONE;
            pipe_use_acc <= 0;
            
            // Initialize registers
            for (i = 0; i < NUM_VREGS; i = i + 1) begin
                vreg_file[i] <= 0;
            end
            for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin
                acc_reg[i] <= 0;
            end
            
        end else begin
            case (fsm_state)
                FSM_IDLE: begin
                    result_valid <= 1'b0;
                    
                    if (instr_valid && instr_ready) begin
                        // Latch instruction
                        pipe_opcode <= opcode;
                        pipe_rd <= rd;
                        pipe_act_type <= act_type;
                        pipe_use_acc <= use_acc;
                        pipe_valid <= 1'b1;
                        
                        fsm_state <= FSM_EXECUTE;
                        exec_cycles <= 0;
                    end
                end
                
                FSM_EXECUTE: begin
                    exec_cycles <= exec_cycles + 1;
                    
                    // Most ops complete in 1 cycle
                    // MAC may take multiple cycles for accumulation
                    if (exec_cycles >= 1) begin
                        fsm_state <= FSM_WRITEBACK;
                    end
                end
                
                FSM_WRITEBACK: begin
                    // Write results based on opcode
                    case (pipe_opcode)
                        OP_VMAC: begin
                            // Store MAC results in accumulator and result
                            for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin
                                acc_reg[i] <= mac_result[i];
                                // Saturate to element width for vector result
                                if (mac_result[i] > {{(ACC_WIDTH-ELEMENT_WIDTH){1'b0}}, {ELEMENT_WIDTH{1'b1}}}) begin
                                    vec_result[(i+1)*ELEMENT_WIDTH-1 -: ELEMENT_WIDTH] <= {ELEMENT_WIDTH{1'b1}};
                                end else if (mac_result[i] < {{(ACC_WIDTH-ELEMENT_WIDTH){1'b1}}, {ELEMENT_WIDTH{1'b0}}}) begin
                                    vec_result[(i+1)*ELEMENT_WIDTH-1 -: ELEMENT_WIDTH] <= {1'b1, {(ELEMENT_WIDTH-1){1'b0}}};
                                end else begin
                                    vec_result[(i+1)*ELEMENT_WIDTH-1 -: ELEMENT_WIDTH] <= mac_result[i][ELEMENT_WIDTH-1:0];
                                end
                            end
                            vreg_file[pipe_rd] <= vec_result;
                        end
                        
                        OP_VADD: begin
                            for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin
                                vec_result[(i+1)*ELEMENT_WIDTH-1 -: ELEMENT_WIDTH] <= add_result[i];
                            end
                            vreg_file[pipe_rd] <= vec_result;
                        end
                        
                        OP_VSUB: begin
                            for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin
                                vec_result[(i+1)*ELEMENT_WIDTH-1 -: ELEMENT_WIDTH] <= sub_result[i];
                            end
                            vreg_file[pipe_rd] <= vec_result;
                        end
                        
                        OP_VMAX: begin
                            for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin
                                vec_result[(i+1)*ELEMENT_WIDTH-1 -: ELEMENT_WIDTH] <= max_result[i];
                            end
                            vreg_file[pipe_rd] <= vec_result;
                        end
                        
                        OP_VMIN: begin
                            for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin
                                vec_result[(i+1)*ELEMENT_WIDTH-1 -: ELEMENT_WIDTH] <= min_result[i];
                            end
                            vreg_file[pipe_rd] <= vec_result;
                        end
                        
                        OP_VREDUCE: begin
                            scalar_result <= reduce_sum;
                        end
                        
                        OP_VACT: begin
                            for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin
                                vec_result[(i+1)*ELEMENT_WIDTH-1 -: ELEMENT_WIDTH] <= act_result[i];
                            end
                            vreg_file[pipe_rd] <= vec_result;
                        end
                        
                        OP_VLOAD: begin
                            vreg_file[pipe_rd] <= vec_a_in;
                        end
                        
                        OP_VMOV: begin
                            vreg_file[pipe_rd] <= operand_a;
                        end
                        
                        default: ;
                    endcase
                    
                    result_valid <= 1'b1;
                    pipe_valid <= 1'b0;
                    
                    if (result_ready || !result_valid) begin
                        fsm_state <= FSM_IDLE;
                    end
                end
                
                default: fsm_state <= FSM_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Status output
    // =========================================================================
    
    assign status = {
        3'b0,
        result_valid,
        pipe_valid,
        busy,
        fsm_state
    };

endmodule
