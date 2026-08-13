// =============================================================================
// SiLens Top-Level Module
// =============================================================================
// Top-level integration of the SiLens vision-language AI accelerator.
//
// Architecture:
//   - PCIe 3.0 x4 interface with DMA
//   - Vision encoder (SigLIP-B/16, 93M params)
//   - Multimodal projector (18M params)
//   - Language model (SmolLM2-135M, 135M params)
//   - Clock domain crossing for PCIe/core domains
//   - Interrupt generation for completion notification
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
    parameter NUM_PATCHES = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE),
    parameter IN_CHANNELS = 3,
    
    // Precision
    parameter ACT_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter FRAC_BITS = 4,
    parameter PARALLEL = 16
)(
    // Clock and reset
    input  wire         clk,
    input  wire         rst_n,
    
    // PCIe interface
    input  wire         pcie_clk,
    input  wire         pcie_rst_n,
    input  wire [127:0] pcie_rx_data,
    input  wire         pcie_rx_valid,
    output wire         pcie_rx_ready,
    output wire [127:0] pcie_tx_data,
    output wire         pcie_tx_valid,
    input  wire         pcie_tx_ready,
    
    // Streaming image input
    input  wire [IN_CHANNELS*ACT_WIDTH-1:0] pixel_in,
    input  wire         pixel_valid,
    output wire         pixel_ready,
    
    // Text token input
    input  wire [$clog2(VOCAB_SIZE)-1:0] token_in,
    input  wire         token_in_valid,
    output wire         token_in_ready,
    
    // Control
    input  wire         frame_start,
    input  wire         seq_start,
    input  wire         generate,
    
    // Output
    output wire [$clog2(VOCAB_SIZE)-1:0] token_out,
    output wire         token_out_valid,
    input  wire         token_out_ready,
    
    // Status/debug
    output wire [7:0]   status_leds,
    output wire         heartbeat,
    output wire         error_flag,
    output wire         vision_busy,
    output wire         llm_busy,
    
    // Interrupt output
    output wire         interrupt,
    
    // Debug interface
    output wire [31:0]  debug_data,
    input  wire [3:0]   debug_sel
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    
    localparam PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS;
    localparam HEAD_DIM = VISION_DIM / VISION_HEADS;
    localparam VISION_MLP_DIM = VISION_DIM * 4;
    localparam LLM_HEAD_DIM = LLM_DIM / LLM_HEADS;
    localparam LLM_MLP_DIM = 1536;
    localparam KV_HEADS = LLM_HEADS;

    // =========================================================================
    // Clock Domain Crossing (PCIe <-> Core)
    // =========================================================================
    
    // Synchronize control signals from PCIe domain to core domain
    reg [2:0] frame_start_sync;
    reg [2:0] seq_start_sync;
    reg [2:0] generate_sync;
    
    wire frame_start_core;
    wire seq_start_core;
    wire generate_core;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            frame_start_sync <= 3'b0;
            seq_start_sync   <= 3'b0;
            generate_sync    <= 3'b0;
        end else begin
            frame_start_sync <= {frame_start_sync[1:0], frame_start};
            seq_start_sync   <= {seq_start_sync[1:0], seq_start};
            generate_sync    <= {generate_sync[1:0], generate};
        end
    end
    
    // Edge detection for pulse signals
    assign frame_start_core = frame_start_sync[1] & ~frame_start_sync[2];
    assign seq_start_core   = seq_start_sync[1] & ~seq_start_sync[2];
    assign generate_core    = generate_sync[1] & ~generate_sync[2];
    
    // Synchronize status signals from core domain to PCIe domain
    reg [1:0] vision_busy_sync;
    reg [1:0] llm_busy_sync;
    reg [1:0] error_sync;
    
    always @(posedge pcie_clk) begin
        if (!pcie_rst_n) begin
            vision_busy_sync <= 2'b0;
            llm_busy_sync    <= 2'b0;
            error_sync       <= 2'b0;
        end else begin
            vision_busy_sync <= {vision_busy_sync[0], (state == STATE_VISION)};
            llm_busy_sync    <= {llm_busy_sync[0], (state >= STATE_LLM_VISION) && (state <= STATE_GENERATE)};
            error_sync       <= {error_sync[0], (state == STATE_ERROR)};
        end
    end


    // =========================================================================
    // Control FSM
    // =========================================================================
    
    localparam STATE_IDLE       = 4'd0;
    localparam STATE_VISION     = 4'd1;
    localparam STATE_PROJECT    = 4'd2;
    localparam STATE_LLM_VISION = 4'd3;
    localparam STATE_LLM_TEXT   = 4'd4;
    localparam STATE_GENERATE   = 4'd5;
    localparam STATE_DONE       = 4'd6;
    localparam STATE_ERROR      = 4'd7;
    
    reg [3:0] state;
    
    // =========================================================================
    // Performance counters
    // =========================================================================
    
    reg [31:0] cycle_counter;
    reg [15:0] token_counter;
    reg [31:0] inference_start_cycle;
    reg [31:0] inference_cycles;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_counter <= 0;
        end else begin
            cycle_counter <= cycle_counter + 1;
        end
    end
    
    // =========================================================================
    // Vision encoder signals
    // =========================================================================
    
    wire [VISION_DIM*ACT_WIDTH-1:0] vision_token_out;
    wire [$clog2(NUM_PATCHES)-1:0] vision_token_idx;
    wire vision_token_valid;
    reg vision_token_ready;
    wire vision_done;
    
    // =========================================================================
    // Projector signals
    // =========================================================================
    
    reg [VISION_DIM*ACT_WIDTH-1:0] proj_x_in;
    reg [$clog2(NUM_PATCHES)-1:0] proj_token_idx_in;
    reg proj_token_valid_in;
    wire proj_token_ready_in;
    reg proj_seq_start;
    reg proj_seq_done_in;
    
    wire [LLM_DIM*ACT_WIDTH-1:0] proj_y_out;
    wire [$clog2(NUM_PATCHES)-1:0] proj_token_idx_out;
    wire proj_valid_out;
    reg proj_ready_out;
    wire proj_busy;


    // =========================================================================
    // Language model signals
    // =========================================================================
    
    reg [LLM_DIM*ACT_WIDTH-1:0] llm_vision_embed;
    reg llm_vision_valid;
    wire llm_vision_ready;
    reg llm_is_vision_token;
    reg llm_seq_start;
    reg llm_generate;
    
    // =========================================================================
    // Weight ROM interfaces (directly connected to hardwired weights)
    // In actual implementation, these would come from ROM or be synthesized as constants
    // =========================================================================
    
    // Vision encoder weights (simplified - would be instantiated from ROM)
    wire [VISION_DIM*PATCH_PIXELS*2-1:0] patch_proj_weights;
    wire [NUM_PATCHES*VISION_DIM*ACT_WIDTH-1:0] pos_embed;
    
    // Projector weights
    wire [VISION_DIM*LLM_DIM*2-1:0] proj_weights;
    wire [LLM_DIM*ACT_WIDTH-1:0] proj_bias;
    
    // =========================================================================
    // Token counters
    // =========================================================================
    
    reg [$clog2(NUM_PATCHES)-1:0] vision_tokens_received;
    reg [$clog2(NUM_PATCHES)-1:0] proj_tokens_received;
    reg [$clog2(NUM_PATCHES)-1:0] llm_tokens_sent;
    
    // =========================================================================
    // Projector instantiation
    // =========================================================================
    
    projector #(
        .IN_DIM(VISION_DIM),
        .OUT_DIM(LLM_DIM),
        .SEQ_LEN(NUM_PATCHES),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) u_projector (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(proj_x_in),
        .token_idx_in(proj_token_idx_in),
        .token_valid_in(proj_token_valid_in),
        .token_ready_in(proj_token_ready_in),
        .seq_start(proj_seq_start),
        .seq_done_in(proj_seq_done_in),
        .weights(proj_weights),
        .bias(proj_bias),
        .y_out(proj_y_out),
        .token_idx_out(proj_token_idx_out),
        .token_valid_out(proj_valid_out),
        .token_ready_out(proj_ready_out),
        .busy(proj_busy)
    );


    // =========================================================================
    // Main Control FSM
    // =========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            vision_tokens_received <= 0;
            proj_tokens_received <= 0;
            llm_tokens_sent <= 0;
            proj_seq_start <= 1'b0;
            proj_seq_done_in <= 1'b0;
            proj_token_valid_in <= 1'b0;
            vision_token_ready <= 1'b0;
            proj_ready_out <= 1'b0;
            llm_vision_valid <= 1'b0;
            llm_is_vision_token <= 1'b0;
            llm_seq_start <= 1'b0;
            llm_generate <= 1'b0;
        end else begin
            // Default pulse signals
            proj_seq_start <= 1'b0;
            proj_seq_done_in <= 1'b0;
            proj_token_valid_in <= 1'b0;
            llm_vision_valid <= 1'b0;
            llm_seq_start <= 1'b0;
            
            case (state)
                STATE_IDLE: begin
                    if (frame_start_core) begin
                        state <= STATE_VISION;
                        vision_tokens_received <= 0;
                        vision_token_ready <= 1'b1;
                        inference_start_cycle <= cycle_counter;
                        token_counter <= 0;
                    end else if (seq_start_core) begin
                        llm_seq_start <= 1'b1;
                    end else if (generate_core) begin
                        llm_generate <= 1'b1;
                        state <= STATE_GENERATE;
                    end
                end
                
                STATE_VISION: begin
                    // Collect vision encoder outputs
                    vision_token_ready <= 1'b1;
                    
                    if (vision_token_valid && vision_token_ready) begin
                        vision_tokens_received <= vision_tokens_received + 1;
                        
                        // Forward to projector
                        proj_x_in <= vision_token_out;
                        proj_token_idx_in <= vision_token_idx;
                        proj_token_valid_in <= 1'b1;
                        
                        if (vision_tokens_received == 0) begin
                            proj_seq_start <= 1'b1;
                        end
                    end
                    
                    if (vision_done) begin
                        proj_seq_done_in <= 1'b1;
                        state <= STATE_PROJECT;
                        vision_token_ready <= 1'b0;
                    end
                end
                
                STATE_PROJECT: begin
                    // Collect projector outputs
                    proj_ready_out <= 1'b1;
                    
                    if (proj_valid_out && proj_ready_out) begin
                        proj_tokens_received <= proj_tokens_received + 1;
                        
                        // Forward to LLM
                        llm_vision_embed <= proj_y_out;
                        llm_vision_valid <= 1'b1;
                        llm_is_vision_token <= 1'b1;
                        
                        if (proj_tokens_received == 0) begin
                            llm_seq_start <= 1'b1;
                        end
                    end
                    
                    if (!proj_busy && proj_tokens_received >= NUM_PATCHES) begin
                        state <= STATE_LLM_VISION;
                        proj_ready_out <= 1'b0;
                        llm_is_vision_token <= 1'b0;
                    end
                end
                
                STATE_LLM_VISION: begin
                    // All vision tokens sent to LLM, switch to text mode
                    state <= STATE_LLM_TEXT;
                end
                
                STATE_LLM_TEXT: begin
                    // Accept text tokens from user
                    if (generate_core) begin
                        llm_generate <= 1'b1;
                        state <= STATE_GENERATE;
                    end
                end
                
                STATE_GENERATE: begin
                    // Autoregressive generation in progress
                    // Return to DONE when EOS or max length reached
                    // (Token counting would be done here in full implementation)
                end
                
                STATE_DONE: begin
                    // Inference complete - record cycles
                    inference_cycles <= cycle_counter - inference_start_cycle;
                    state <= STATE_IDLE;
                end
                
                STATE_ERROR: begin
                    // Stay in error state until reset
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end


    // =========================================================================
    // Token I/O assignments
    // =========================================================================
    
    assign token_in_ready = (state == STATE_LLM_TEXT);
    
    // =========================================================================
    // Interrupt generation
    // =========================================================================
    
    reg interrupt_r;
    reg [1:0] state_prev;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            interrupt_r <= 1'b0;
            state_prev  <= 2'b0;
        end else begin
            state_prev <= state[1:0];
            
            // Generate interrupt on transition to DONE or ERROR
            if ((state == STATE_DONE || state == STATE_ERROR) && 
                (state_prev != state[1:0])) begin
                interrupt_r <= 1'b1;
            end else begin
                interrupt_r <= 1'b0;
            end
        end
    end
    
    assign interrupt = interrupt_r;
    
    // =========================================================================
    // Debug multiplexer
    // =========================================================================
    
    reg [31:0] debug_data_r;
    
    always @(*) begin
        case (debug_sel)
            4'd0: debug_data_r = {28'b0, state};
            4'd1: debug_data_r = cycle_counter;
            4'd2: debug_data_r = inference_cycles;
            4'd3: debug_data_r = {16'b0, token_counter};
            4'd4: debug_data_r = {16'b0, vision_tokens_received, 6'b0};
            4'd5: debug_data_r = {16'b0, proj_tokens_received, 6'b0};
            4'd6: debug_data_r = {31'b0, proj_busy};
            4'd7: debug_data_r = {24'b0, status_leds};
            default: debug_data_r = 32'hDEAD_BEEF;
        endcase
    end
    
    assign debug_data = debug_data_r;
    
    // =========================================================================
    // Status outputs
    // =========================================================================
    
    assign status_leds = {4'b0, state};
    assign error_flag = (state == STATE_ERROR);
    assign vision_busy = (state == STATE_VISION);
    assign llm_busy = (state == STATE_LLM_VISION) || (state == STATE_LLM_TEXT) || 
                      (state == STATE_GENERATE);
    
    // Heartbeat: toggle every ~1 second at 100MHz
    reg [26:0] heartbeat_cnt;
    always @(posedge clk) begin
        if (!rst_n) begin
            heartbeat_cnt <= 0;
        end else begin
            heartbeat_cnt <= heartbeat_cnt + 1;
        end
    end
    assign heartbeat = heartbeat_cnt[26];
    
    // =========================================================================
    // PCIe placeholder assignments (to be connected to PCIe controller)
    // =========================================================================
    
    assign pcie_rx_ready = 1'b0;
    assign pcie_tx_data = 128'b0;
    assign pcie_tx_valid = 1'b0;
    
    // =========================================================================
    // Weight ROM placeholders (would be actual ROM in synthesis)
    // =========================================================================
    
    assign patch_proj_weights = {(VISION_DIM*PATCH_PIXELS){2'b01}};
    assign pos_embed = {(NUM_PATCHES*VISION_DIM*ACT_WIDTH){1'b0}};
    assign proj_weights = {(VISION_DIM*LLM_DIM){2'b01}};
    assign proj_bias = {(LLM_DIM*ACT_WIDTH){1'b0}};

endmodule
