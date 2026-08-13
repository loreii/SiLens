// =============================================================================
// SiLens Top-Level Module
// =============================================================================
// Top-level integration of the SiLens vision-language AI accelerator.
//
// Architecture:
//   - PCIe 3.0 x4 interface
//   - Vision encoder (SigLIP-B/16, 93M params)
//   - Multimodal projector (18M params)
//   - Language model (SmolLM2-135M, 135M params)
//
// License: Apache 2.0
// =============================================================================

module silens_top #(
    // Clock frequencies
    parameter CLK_FREQ_MHZ = 100,
    
    // Model parameters
    parameter VISION_DIM = 768,
    parameter VISION_LAYERS = 12,
    parameter VISION_HEADS = 12,
    parameter LLM_DIM = 576,
    parameter LLM_LAYERS = 30,
    parameter LLM_HEADS = 9,
    parameter VOCAB_SIZE = 49152,
    parameter MAX_SEQ_LEN = 8192,
    
    // Image parameters
    parameter IMG_SIZE = 384,
    parameter PATCH_SIZE = 16,
    parameter NUM_PATCHES = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE),  // 576
    
    // Precision
    parameter ACT_WIDTH = 8,    // Activation bit width
    parameter ACC_WIDTH = 32    // Accumulator bit width
)(
    // Clock and reset
    input  wire         clk,
    input  wire         rst_n,
    
    // PCIe interface (directly connected to PHY)
    // Note: Actual PCIe signals depend on PHY IP used
    input  wire         pcie_clk,
    input  wire         pcie_rst_n,
    input  wire [127:0] pcie_rx_data,
    input  wire         pcie_rx_valid,
    output wire         pcie_rx_ready,
    output wire [127:0] pcie_tx_data,
    output wire         pcie_tx_valid,
    input  wire         pcie_tx_ready,
    
    // Status/debug
    output wire [7:0]   status_leds,
    output wire         heartbeat,
    output wire         error_flag
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    
    // Control FSM states
    localparam STATE_IDLE       = 4'd0;
    localparam STATE_LOAD_IMAGE = 4'd1;
    localparam STATE_VISION     = 4'd2;
    localparam STATE_PROJECT    = 4'd3;
    localparam STATE_LOAD_TEXT  = 4'd4;
    localparam STATE_LLM        = 4'd5;
    localparam STATE_OUTPUT     = 4'd6;
    localparam STATE_ERROR      = 4'd7;
    
    reg [3:0] state, next_state;
    
    // Image buffer
    reg [ACT_WIDTH-1:0] image_buffer [0:IMG_SIZE*IMG_SIZE*3-1];
    reg [$clog2(IMG_SIZE*IMG_SIZE*3)-1:0] img_load_ptr;
    wire img_load_done;
    
    // Vision encoder interface
    wire [VISION_DIM*ACT_WIDTH-1:0] vision_input;
    wire vision_input_valid;
    wire vision_input_ready;
    wire [VISION_DIM*ACT_WIDTH-1:0] vision_output;
    wire vision_output_valid;
    wire vision_output_ready;
    
    // Projector interface
    wire [VISION_DIM*ACT_WIDTH-1:0] proj_input;
    wire proj_input_valid;
    wire proj_input_ready;
    wire [LLM_DIM*ACT_WIDTH-1:0] proj_output;
    wire proj_output_valid;
    wire proj_output_ready;
    
    // Language model interface
    wire [LLM_DIM*ACT_WIDTH-1:0] llm_input;
    wire llm_input_valid;
    wire llm_input_ready;
    wire [$clog2(VOCAB_SIZE)-1:0] llm_token_out;
    wire llm_token_valid;
    wire llm_token_ready;
    
    // =========================================================================
    // Submodule instantiation
    // =========================================================================
    
    // TODO: Instantiate PCIe controller
    // pcie_ctrl #(...) u_pcie_ctrl (...);
    
    // TODO: Instantiate vision encoder
    // vision_encoder #(
    //     .DIM(VISION_DIM),
    //     .LAYERS(VISION_LAYERS),
    //     .HEADS(VISION_HEADS),
    //     .ACT_WIDTH(ACT_WIDTH)
    // ) u_vision_encoder (
    //     .clk(clk),
    //     .rst_n(rst_n),
    //     .input_data(vision_input),
    //     .input_valid(vision_input_valid),
    //     .input_ready(vision_input_ready),
    //     .output_data(vision_output),
    //     .output_valid(vision_output_valid),
    //     .output_ready(vision_output_ready)
    // );
    
    // TODO: Instantiate projector
    // projector #(
    //     .IN_DIM(VISION_DIM),
    //     .OUT_DIM(LLM_DIM),
    //     .ACT_WIDTH(ACT_WIDTH)
    // ) u_projector (...);
    
    // TODO: Instantiate language model
    // language_model #(
    //     .DIM(LLM_DIM),
    //     .LAYERS(LLM_LAYERS),
    //     .HEADS(LLM_HEADS),
    //     .VOCAB_SIZE(VOCAB_SIZE),
    //     .ACT_WIDTH(ACT_WIDTH)
    // ) u_language_model (...);
    
    // =========================================================================
    // Control FSM
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always @(*) begin
        next_state = state;
        
        case (state)
            STATE_IDLE: begin
                // Wait for command from PCIe
                // if (cmd_valid && cmd_type == CMD_INFERENCE)
                //     next_state = STATE_LOAD_IMAGE;
            end
            
            STATE_LOAD_IMAGE: begin
                if (img_load_done)
                    next_state = STATE_VISION;
            end
            
            STATE_VISION: begin
                if (vision_output_valid && vision_output_ready)
                    next_state = STATE_PROJECT;
            end
            
            STATE_PROJECT: begin
                if (proj_output_valid && proj_output_ready)
                    next_state = STATE_LLM;
            end
            
            STATE_LLM: begin
                // Autoregressive generation
                if (llm_token_valid && llm_token_out == 16'd2)  // EOS token
                    next_state = STATE_OUTPUT;
            end
            
            STATE_OUTPUT: begin
                // Send results back via PCIe
                next_state = STATE_IDLE;
            end
            
            STATE_ERROR: begin
                // Stay in error until reset
            end
            
            default: next_state = STATE_IDLE;
        endcase
    end
    
    // =========================================================================
    // Status outputs
    // =========================================================================
    
    assign status_leds = {4'b0, state};
    assign error_flag = (state == STATE_ERROR);
    
    // Heartbeat: toggle every ~1 second
    reg [26:0] heartbeat_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            heartbeat_cnt <= 0;
        end else begin
            heartbeat_cnt <= heartbeat_cnt + 1;
        end
    end
    assign heartbeat = heartbeat_cnt[26];

endmodule
