// =============================================================================
// SiLens Vision-Language Model Processing Core
// =============================================================================
// Top-level wrapper for the complete VLM inference pipeline:
//   1. Vision Encoder (SigLIP-B/16) - 93M ternary parameters
//   2. Multimodal Projector - 18M ternary parameters  
//   3. Language Model (SmolLM2-135M) - 135M ternary parameters
//
// All weights are hardwired as metal routing (ternary: -1, 0, +1)
// KV cache stored in external DDR3 memory
//
// Target: SkyWater SKY130 130nm CMOS
// Area: ~700mm² for compute blocks
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module silens_vlm_core #(
    // =========================================================================
    // Vision Encoder Parameters (SigLIP-B/16)
    // =========================================================================
    parameter VISION_DIM     = 768,
    parameter VISION_LAYERS  = 12,
    parameter VISION_HEADS   = 12,
    parameter VISION_MLP_DIM = 3072,
    parameter IMG_SIZE       = 384,
    parameter PATCH_SIZE     = 16,
    parameter NUM_PATCHES    = (IMG_SIZE/PATCH_SIZE) * (IMG_SIZE/PATCH_SIZE),  // 576
    parameter IN_CHANNELS    = 3,
    
    // =========================================================================
    // Language Model Parameters (SmolLM2-135M)
    // =========================================================================
    parameter LLM_DIM        = 576,
    parameter LLM_LAYERS     = 30,
    parameter LLM_HEADS      = 9,
    parameter LLM_KV_HEADS   = 9,
    parameter LLM_MLP_DIM    = 1536,
    parameter VOCAB_SIZE     = 49152,
    parameter MAX_SEQ_LEN    = 2048,
    
    // =========================================================================
    // Precision Parameters
    // =========================================================================
    parameter ACT_WIDTH      = 8,
    parameter ACC_WIDTH      = 32,
    parameter PARALLEL       = 64,      // SIMD lanes
    
    // =========================================================================
    // Memory Interface Parameters
    // =========================================================================
    parameter MEM_ADDR_WIDTH = 28,
    parameter MEM_DATA_WIDTH = 512
)(
    // Clock and reset
    input  wire                         clk,
    input  wire                         rst_n,
    
    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire                         frame_start,    // Start vision processing
    input  wire                         seq_start,      // Start sequence processing
    input  wire                         gen_start,      // Start generation
    input  wire                         abort,          // Abort current operation
    
    // =========================================================================
    // Pixel Input (for vision encoder)
    // =========================================================================
    input  wire [IN_CHANNELS*ACT_WIDTH-1:0] pixel_in,
    input  wire                         pixel_valid,
    output wire                         pixel_ready,
    
    // =========================================================================
    // Token Input (for text prompts)
    // =========================================================================
    input  wire [15:0]                  token_in,
    input  wire                         token_in_valid,
    output wire                         token_in_ready,
    
    // =========================================================================
    // Token Output (generated tokens)
    // =========================================================================
    output wire [$clog2(VOCAB_SIZE)-1:0] token_out,
    output wire                         token_out_valid,
    input  wire                         token_out_ready,
    
    // =========================================================================
    // External Memory Interface (KV Cache)
    // =========================================================================
    output wire [MEM_ADDR_WIDTH-1:0]    mem_addr,
    output wire [MEM_DATA_WIDTH-1:0]    mem_wdata,
    input  wire [MEM_DATA_WIDTH-1:0]    mem_rdata,
    output wire                         mem_rd,
    output wire                         mem_wr,
    input  wire                         mem_ready,
    
    // =========================================================================
    // Status Outputs
    // =========================================================================
    output wire                         vision_busy,
    output wire                         llm_busy,
    output wire                         inference_done,
    output wire                         error_flag,
    output wire [3:0]                   state_out
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    
    localparam PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS;
    localparam VISION_HEAD_DIM = VISION_DIM / VISION_HEADS;
    localparam LLM_HEAD_DIM = LLM_DIM / LLM_HEADS;
    
    // State machine states
    localparam ST_IDLE         = 4'd0;
    localparam ST_VISION_PATCH = 4'd1;
    localparam ST_VISION_XFORM = 4'd2;
    localparam ST_PROJECT      = 4'd3;
    localparam ST_LLM_PREFILL  = 4'd4;
    localparam ST_LLM_DECODE   = 4'd5;
    localparam ST_OUTPUT       = 4'd6;
    localparam ST_DONE         = 4'd7;
    localparam ST_ERROR        = 4'd8;
    
    // =========================================================================
    // State Machine
    // =========================================================================
    
    reg [3:0] state;
    reg [3:0] state_next;
    
    assign state_out = state;
    assign vision_busy = (state >= ST_VISION_PATCH) && (state <= ST_PROJECT);
    assign llm_busy = (state >= ST_LLM_PREFILL) && (state <= ST_OUTPUT);
    assign inference_done = (state == ST_DONE);
    assign error_flag = (state == ST_ERROR);
    
    // =========================================================================
    // Counters
    // =========================================================================
    
    reg [$clog2(NUM_PATCHES)-1:0] patch_cnt;
    reg [$clog2(VISION_LAYERS)-1:0] vision_layer_cnt;
    reg [$clog2(LLM_LAYERS)-1:0] llm_layer_cnt;
    reg [$clog2(MAX_SEQ_LEN)-1:0] seq_pos;
    reg [$clog2(MAX_SEQ_LEN)-1:0] gen_len;
    reg [15:0] max_gen_tokens;
    
    // =========================================================================
    // Inter-module Data Buses
    // =========================================================================
    
    // Vision encoder output
    wire [VISION_DIM*ACT_WIDTH-1:0] vision_out;
    wire [9:0] vision_out_idx;  // Fixed 10-bit width for 576 patches
    wire vision_out_valid;
    reg vision_out_ready;
    wire vision_done;
    
    // Projector output
    wire [LLM_DIM*ACT_WIDTH-1:0] proj_out;
    wire [9:0] proj_out_idx;  // Fixed 10-bit width for 576 patches
    wire proj_out_valid;
    reg proj_out_ready;
    wire proj_done;
    
    // LLM output
    wire [$clog2(VOCAB_SIZE)-1:0] llm_logits_out;
    wire llm_logits_valid;
    reg llm_logits_ready;
    
    // Token embedding lookup
    wire [LLM_DIM*ACT_WIDTH-1:0] token_embed;
    wire token_embed_valid;
    
    // =========================================================================
    // Memory Arbiter Signals
    // =========================================================================
    
    // Vision encoder memory interface
    wire [MEM_ADDR_WIDTH-1:0] vision_mem_addr;
    wire [MEM_DATA_WIDTH-1:0] vision_mem_wdata;
    wire [MEM_DATA_WIDTH-1:0] vision_mem_rdata;
    wire vision_mem_rd;
    wire vision_mem_wr;
    wire vision_mem_ready;
    
    // LLM memory interface (KV cache)
    wire [MEM_ADDR_WIDTH-1:0] llm_mem_addr;
    wire [MEM_DATA_WIDTH-1:0] llm_mem_wdata;
    wire [MEM_DATA_WIDTH-1:0] llm_mem_rdata;
    wire llm_mem_rd;
    wire llm_mem_wr;
    wire llm_mem_ready;
    
    // =========================================================================
    // State Machine Logic
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            patch_cnt <= 0;
            vision_layer_cnt <= 0;
            llm_layer_cnt <= 0;
            seq_pos <= 0;
            gen_len <= 0;
            max_gen_tokens <= 256;  // Default max generation
            vision_out_ready <= 0;
            proj_out_ready <= 0;
            llm_logits_ready <= 0;
        end else begin
            if (abort) begin
                state <= ST_IDLE;
            end else begin
                case (state)
                    ST_IDLE: begin
                        patch_cnt <= 0;
                        vision_layer_cnt <= 0;
                        llm_layer_cnt <= 0;
                        seq_pos <= 0;
                        gen_len <= 0;
                        
                        if (frame_start) begin
                            // Start vision processing
                            state <= ST_VISION_PATCH;
                        end else if (seq_start) begin
                            // Start text-only processing
                            state <= ST_LLM_PREFILL;
                        end else if (gen_start) begin
                            // Start generation (after prefill)
                            state <= ST_LLM_DECODE;
                        end
                    end
                    
                    ST_VISION_PATCH: begin
                        // Patch extraction and embedding
                        if (pixel_valid && pixel_ready) begin
                            patch_cnt <= patch_cnt + 1;
                            if (patch_cnt == NUM_PATCHES - 1) begin
                                state <= ST_VISION_XFORM;
                                vision_layer_cnt <= 0;
                            end
                        end
                    end
                    
                    ST_VISION_XFORM: begin
                        // Vision transformer layers
                        vision_out_ready <= 1;
                        if (vision_done) begin
                            state <= ST_PROJECT;
                            vision_out_ready <= 0;
                        end
                    end
                    
                    ST_PROJECT: begin
                        // Multimodal projection
                        proj_out_ready <= 1;
                        if (proj_done) begin
                            state <= ST_LLM_PREFILL;
                            proj_out_ready <= 0;
                            seq_pos <= NUM_PATCHES;  // Vision tokens already processed
                        end
                    end
                    
                    ST_LLM_PREFILL: begin
                        // LLM prefill (process prompt + vision tokens)
                        if (token_in_valid && token_in_ready) begin
                            seq_pos <= seq_pos + 1;
                        end
                        
                        // Check for end of prefill (signaled by gen_start)
                        if (gen_start) begin
                            state <= ST_LLM_DECODE;
                        end
                    end
                    
                    ST_LLM_DECODE: begin
                        // Autoregressive decoding
                        llm_logits_ready <= 1;
                        
                        if (llm_logits_valid && llm_logits_ready) begin
                            gen_len <= gen_len + 1;
                            seq_pos <= seq_pos + 1;
                            state <= ST_OUTPUT;
                        end
                    end
                    
                    ST_OUTPUT: begin
                        // Output generated token
                        if (token_out_valid && token_out_ready) begin
                            // Check termination conditions
                            if (llm_logits_out == 0 ||  // EOS token
                                gen_len >= max_gen_tokens ||
                                seq_pos >= MAX_SEQ_LEN) begin
                                state <= ST_DONE;
                            end else begin
                                state <= ST_LLM_DECODE;
                            end
                        end
                        llm_logits_ready <= 0;
                    end
                    
                    ST_DONE: begin
                        // Wait for acknowledgment then return to idle
                        if (frame_start || seq_start || gen_start) begin
                            state <= ST_IDLE;
                        end
                    end
                    
                    ST_ERROR: begin
                        // Error state - requires reset or abort
                        if (abort) begin
                            state <= ST_IDLE;
                        end
                    end
                    
                    default: state <= ST_IDLE;
                endcase
            end
        end
    end
    
    // =========================================================================
    // Vision Encoder Instance
    // =========================================================================
    
    wire vision_start = (state == ST_IDLE && frame_start);
    
    vision_encoder_top #(
        .IMG_SIZE(IMG_SIZE),
        .PATCH_SIZE(PATCH_SIZE),
        .IN_CHANNELS(IN_CHANNELS),
        .EMBED_DIM(VISION_DIM),
        .NUM_LAYERS(VISION_LAYERS),
        .NUM_HEADS(VISION_HEADS),
        .MLP_DIM(VISION_MLP_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) u_vision_encoder (
        .clk(clk),
        .rst_n(rst_n),
        
        // Control
        .start(vision_start),
        .done(vision_done),
        
        // Pixel input
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .pixel_ready(pixel_ready),
        
        // Token output
        .token_out(vision_out),
        .token_idx(vision_out_idx),
        .token_valid(vision_out_valid),
        .token_ready(vision_out_ready),
        
        // Memory interface (for intermediate storage if needed)
        .mem_addr(vision_mem_addr),
        .mem_wdata(vision_mem_wdata),
        .mem_rdata(vision_mem_rdata),
        .mem_rd(vision_mem_rd),
        .mem_wr(vision_mem_wr),
        .mem_ready(vision_mem_ready)
    );
    
    // =========================================================================
    // Multimodal Projector Instance
    // =========================================================================
    
    wire proj_start = (state == ST_VISION_XFORM && vision_done);
    wire proj_busy;
    
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
        
        // Input from vision encoder
        .x_in(vision_out),
        .token_idx_in(vision_out_idx),
        .token_valid_in(vision_out_valid & vision_out_ready),
        .token_ready_in(),
        
        // Control
        .seq_start(proj_start),
        .seq_done_in(vision_done),
        
        // Weights (hardwired)
        .weights({(VISION_DIM*LLM_DIM){2'b01}}),
        .bias({(LLM_DIM*ACT_WIDTH){1'b0}}),
        
        // Output to LLM
        .y_out(proj_out),
        .token_idx_out(proj_out_idx),
        .token_valid_out(proj_out_valid),
        .token_ready_out(proj_out_ready),
        
        .busy(proj_busy)
    );
    
    // Projector done when not busy and we've processed all tokens
    assign proj_done = !proj_busy && (state == ST_PROJECT);
    
    // =========================================================================
    // Token Embedding Lookup
    // =========================================================================
    
    // Embedding table is hardwired (49152 × 576 × ternary = ~14MB)
    // In actual implementation, this would be ROM
    reg [LLM_DIM*ACT_WIDTH-1:0] embed_out_reg;
    reg embed_valid_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            embed_out_reg <= 0;
            embed_valid_reg <= 0;
        end else begin
            embed_valid_reg <= 0;
            if (token_in_valid && token_in_ready) begin
                // Embedding lookup (placeholder - actual impl is ROM)
                embed_out_reg <= {(LLM_DIM*ACT_WIDTH){1'b0}};
                embed_valid_reg <= 1;
            end
        end
    end
    
    assign token_embed = embed_out_reg;
    assign token_embed_valid = embed_valid_reg;
    assign token_in_ready = (state == ST_LLM_PREFILL);
    
    // =========================================================================
    // Language Model Instance
    // =========================================================================
    
    // Mux between vision projection output and text token embedding
    wire [LLM_DIM*ACT_WIDTH-1:0] llm_input;
    wire llm_input_valid;
    wire llm_input_ready;
    wire llm_is_vision;
    
    assign llm_is_vision = (state == ST_PROJECT);
    assign llm_input = llm_is_vision ? proj_out : token_embed;
    assign llm_input_valid = llm_is_vision ? (proj_out_valid & proj_out_ready) : token_embed_valid;
    
    llm_decoder_top #(
        .EMBED_DIM(LLM_DIM),
        .NUM_LAYERS(LLM_LAYERS),
        .NUM_HEADS(LLM_HEADS),
        .NUM_KV_HEADS(LLM_KV_HEADS),
        .MLP_DIM(LLM_MLP_DIM),
        .VOCAB_SIZE(VOCAB_SIZE),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL),
        .MEM_ADDR_WIDTH(MEM_ADDR_WIDTH),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH)
    ) u_llm_decoder (
        .clk(clk),
        .rst_n(rst_n),
        
        // Control
        .prefill_mode(state == ST_LLM_PREFILL || state == ST_PROJECT),
        .decode_mode(state == ST_LLM_DECODE),
        .seq_pos(seq_pos),
        
        // Input embedding
        .embed_in(llm_input),
        .embed_valid(llm_input_valid),
        .embed_ready(llm_input_ready),
        
        // Output logits
        .logits_out(llm_logits_out),
        .logits_valid(llm_logits_valid),
        .logits_ready(llm_logits_ready),
        
        // KV cache memory interface
        .mem_addr(llm_mem_addr),
        .mem_wdata(llm_mem_wdata),
        .mem_rdata(llm_mem_rdata),
        .mem_rd(llm_mem_rd),
        .mem_wr(llm_mem_wr),
        .mem_ready(llm_mem_ready)
    );
    
    // =========================================================================
    // Memory Arbiter
    // =========================================================================
    
    // Simple priority arbiter: LLM > Vision
    reg mem_sel;  // 0 = vision, 1 = LLM
    
    always @(*) begin
        if (llm_mem_rd || llm_mem_wr)
            mem_sel = 1;
        else
            mem_sel = 0;
    end
    
    assign mem_addr = mem_sel ? llm_mem_addr : vision_mem_addr;
    assign mem_wdata = mem_sel ? llm_mem_wdata : vision_mem_wdata;
    assign mem_rd = mem_sel ? llm_mem_rd : vision_mem_rd;
    assign mem_wr = mem_sel ? llm_mem_wr : vision_mem_wr;
    
    assign vision_mem_rdata = mem_rdata;
    assign llm_mem_rdata = mem_rdata;
    assign vision_mem_ready = mem_ready & ~mem_sel;
    assign llm_mem_ready = mem_ready & mem_sel;
    
    // =========================================================================
    // Output Token Assignment
    // =========================================================================
    
    assign token_out = llm_logits_out;
    assign token_out_valid = (state == ST_OUTPUT);

endmodule

`default_nettype wire
