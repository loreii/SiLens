// =============================================================================
// LLM Subsystem - Level 3 Synthesis Block
// =============================================================================
// Complete SmolLM2-135M decoder with 30 transformer layers.
// Composes Level 2 macros: 30× transformer_block_llm + 1× embedding_block
//
// Architecture:
//   Input tokens → embedding_block → 30× transformer_block_llm (sequential)
//                                → Final RMS Norm → LM Head → Output logits
//
// Specifications:
//   - Vocabulary: 49152 tokens (16-bit token IDs)
//   - Hidden dimension: 576
//   - Attention heads: 9 per layer (270 total)
//   - MLP hidden: 1536
//   - Max sequence: 256 tokens
//
// Operating Modes:
//   - Prefill: Process entire input sequence, populate KV cache
//   - Decode: Generate one token at a time using KV cache
//
// Target: ~400mm² on SKY130 (20000µm × 20000µm)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module llm_subsystem #(
    parameter VOCAB_SIZE    = 49152,    // Vocabulary size
    parameter HIDDEN_DIM    = 576,      // Model hidden dimension
    parameter NUM_LAYERS    = 30,       // Number of transformer layers
    parameter NUM_HEADS     = 9,        // Attention heads per layer
    parameter HEAD_DIM      = 64,       // Per-head dimension
    parameter MLP_HIDDEN    = 1536,     // MLP intermediate dimension
    parameter MAX_SEQ       = 256,      // Maximum sequence length
    parameter ACT_WIDTH     = 8,        // Activation bit width
    parameter ACC_WIDTH     = 24,       // Accumulator bit width
    parameter TOKEN_BITS    = 16,       // Token ID bit width
    parameter TOP_K         = 8         // Top-K sampling parameter
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire                         start,          // Start processing
    input  wire                         input_mode,     // 0=token IDs, 1=embeddings from projector
    input  wire                         generate_mode,  // 0=prefill, 1=autoregressive decode
    input  wire [$clog2(MAX_SEQ)-1:0]   seq_len,        // Input sequence length (prefill)
    input  wire [$clog2(MAX_SEQ)-1:0]   max_new_tokens, // Max tokens to generate
    input  wire                         use_top_k,      // 0=greedy, 1=top-K sampling
    output wire                         busy,           // Processing in progress
    output wire                         done,           // Generation complete
    output wire [4:0]                   current_layer,  // Current layer being processed
    
    // =========================================================================
    // Token Input Interface (input_mode=0)
    // =========================================================================
    input  wire                         token_in_valid,
    output wire                         token_in_ready,
    input  wire [TOKEN_BITS-1:0]        token_in_id,    // Token ID (0 to 49151)
    input  wire                         token_in_last,  // Last token in sequence
    
    // =========================================================================
    // Embedding Input Interface (input_mode=1, from vision projector)
    // =========================================================================
    input  wire                         emb_in_valid,
    output wire                         emb_in_ready,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0] emb_in_data, // 576 × 8-bit = 4608 bits
    input  wire                         emb_in_last,
    
    // =========================================================================
    // Output Logits Interface (streaming)
    // =========================================================================
    output wire                         logits_valid,
    input  wire                         logits_ready,
    output wire [ACT_WIDTH-1:0]         logits_data,    // One logit at a time
    output wire [$clog2(VOCAB_SIZE)-1:0] logits_idx,    // Current vocabulary index
    output wire                         logits_last,    // Last logit
    
    // =========================================================================
    // Generated Token Output Interface
    // =========================================================================
    output wire                         gen_token_valid,
    input  wire                         gen_token_ready,
    output wire [TOKEN_BITS-1:0]        gen_token_id,   // Generated token ID
    output wire                         gen_token_eos,  // End of sequence token
    
    // =========================================================================
    // Token Embedding Memory Interface (external DRAM for 27MB vocab embeddings)
    // =========================================================================
    output wire                         tok_emb_mem_req,
    output wire [31:0]                  tok_emb_mem_addr,
    output wire [7:0]                   tok_emb_mem_len,
    input  wire                         tok_emb_mem_grant,
    input  wire                         tok_emb_mem_valid,
    input  wire [63:0]                  tok_emb_mem_data,
    output wire                         tok_emb_mem_ready,
    
    // =========================================================================
    // Position Embedding Memory Interface (on-chip SRAM, 1.1MB)
    // =========================================================================
    output wire [11:0]                  pos_emb_mem_addr,
    output wire [9:0]                   pos_emb_mem_elem,
    output wire                         pos_emb_mem_rd,
    input  wire [ACT_WIDTH-1:0]         pos_emb_mem_data,
    input  wire                         pos_emb_mem_valid,
    
    // =========================================================================
    // KV Cache Memory Interface (external DDR3, per-layer arbiter)
    // =========================================================================
    // Unified interface for all 30 layers × 9 heads = 270 head caches
    output wire                         kv_cache_req,
    output wire                         kv_cache_wr,    // 0=read, 1=write
    output wire [4:0]                   kv_cache_layer, // Layer index (0-29)
    output wire [3:0]                   kv_cache_head,  // Head index (0-8)
    output wire                         kv_cache_sel,   // 0=K, 1=V
    output wire [$clog2(MAX_SEQ)-1:0]   kv_cache_pos,   // Position in sequence
    output wire [HEAD_DIM*ACT_WIDTH-1:0] kv_cache_wdata, // 64 × 8 = 512 bits
    input  wire                         kv_cache_grant,
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] kv_cache_rdata,
    input  wire                         kv_cache_rvalid,
    
    // =========================================================================
    // Weight Memory Interface (all layer weights, ternary packed)
    // =========================================================================
    // RMS Norm gamma (per layer × 2 norms)
    output wire                         weight_rms_rd,
    output wire [4:0]                   weight_rms_layer,
    output wire                         weight_rms_sel,  // 0=pre-attn, 1=pre-mlp
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0] weight_rms_data,
    input  wire                         weight_rms_valid,
    
    // Attention Q/K/V/O weights
    output wire                         weight_attn_rd,
    output wire [4:0]                   weight_attn_layer,
    output wire [1:0]                   weight_attn_sel,  // 0=Q, 1=K, 2=V, 3=O
    output wire [$clog2(HIDDEN_DIM)-1:0] weight_attn_addr,
    input  wire [HIDDEN_DIM*2-1:0]      weight_attn_data, // Ternary packed
    input  wire                         weight_attn_valid,
    
    // MLP weights
    output wire                         weight_mlp_rd,
    output wire [4:0]                   weight_mlp_layer,
    output wire [1:0]                   weight_mlp_sel,   // 0=gate, 1=up, 2=down
    output wire [$clog2(MLP_HIDDEN)-1:0] weight_mlp_addr,
    input  wire [MLP_HIDDEN*2-1:0]      weight_mlp_gate_data,
    input  wire [MLP_HIDDEN*2-1:0]      weight_mlp_up_data,
    input  wire [HIDDEN_DIM*2-1:0]      weight_mlp_down_data,
    input  wire                         weight_mlp_valid,
    
    // Final RMS Norm gamma
    output wire                         weight_final_rms_rd,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0] weight_final_rms_data,
    input  wire                         weight_final_rms_valid,
    
    // LM Head weights (576 → 49152, ternary)
    output wire                         weight_lm_head_rd,
    output wire [$clog2(VOCAB_SIZE)-1:0] weight_lm_head_addr,
    input  wire [HIDDEN_DIM*2-1:0]      weight_lm_head_data, // Ternary packed row
    input  wire                         weight_lm_head_valid
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam SEQ_BITS     = $clog2(MAX_SEQ);
    localparam LAYER_BITS   = $clog2(NUM_LAYERS);
    localparam VOCAB_BITS   = $clog2(VOCAB_SIZE);
    localparam DIM_BITS     = $clog2(HIDDEN_DIM);
    localparam MLP_BITS     = $clog2(MLP_HIDDEN);
    
    // Token buffer width
    localparam TOKEN_DATA_WIDTH = HIDDEN_DIM * ACT_WIDTH;  // 4608 bits

    // =========================================================================
    // Main State Machine
    // =========================================================================
    localparam [3:0]
        ST_IDLE         = 4'd0,
        ST_EMBED        = 4'd1,     // Token embedding lookup
        ST_LAYERS       = 4'd2,     // Process through transformer layers
        ST_FINAL_NORM   = 4'd3,     // Final RMS normalization
        ST_LM_HEAD      = 4'd4,     // Project to vocabulary
        ST_SAMPLE       = 4'd5,     // Sample next token
        ST_OUTPUT_LOGITS= 4'd6,     // Stream logits out
        ST_OUTPUT_TOKEN = 4'd7,     // Output generated token
        ST_NEXT_TOKEN   = 4'd8,     // Prepare for next token generation
        ST_DONE         = 4'd9;
    
    reg [3:0] state;
    reg [3:0] next_state;
    
    // =========================================================================
    // Processing Registers
    // =========================================================================
    reg [4:0]              layer_idx;          // Current layer (0-29)
    reg [SEQ_BITS-1:0]     position;           // Current position in sequence
    reg [SEQ_BITS-1:0]     seq_len_reg;        // Registered sequence length
    reg [SEQ_BITS-1:0]     tokens_generated;   // Count of generated tokens
    reg [SEQ_BITS-1:0]     max_new_tokens_reg;
    reg                    is_prefill;         // Prefill vs decode mode
    reg                    use_top_k_reg;
    
    // Token data buffers
    reg [TOKEN_DATA_WIDTH-1:0] hidden_state;   // Current hidden state
    reg [TOKEN_DATA_WIDTH-1:0] layer_input;    // Input to current layer
    reg [TOKEN_DATA_WIDTH-1:0] layer_output;   // Output from current layer
    
    // Generated token register
    reg [TOKEN_BITS-1:0]   sampled_token;
    reg                    sampled_eos;
    
    // =========================================================================
    // Embedding Block Instance
    // =========================================================================
    wire                    emb_busy;
    wire                    emb_done;
    wire                    emb_out_valid;
    wire [ACT_WIDTH-1:0]    emb_out_data;
    wire [DIM_BITS-1:0]     emb_out_elem_idx;
    wire                    emb_out_last_elem;
    wire                    emb_out_last_token;
    reg                     emb_out_ready;
    reg                     emb_start;
    reg [5:0]               emb_batch_size;
    wire                    token_in_ready_emb;

    embedding_block #(
        .VOCAB_SIZE     (VOCAB_SIZE),
        .MAX_POS        (MAX_SEQ),
        .EMBED_DIM      (HIDDEN_DIM),
        .DATA_WIDTH     (ACT_WIDTH),
        .TOKEN_BITS     (TOKEN_BITS),
        .POS_BITS       (12),
        .BATCH_MAX      (32),
        .MEM_BURST_LEN  (8)
    ) u_embedding (
        .clk            (clk),
        .rst_n          (rst_n),
        
        // Control
        .start          (emb_start),
        .batch_size     (emb_batch_size),
        .busy           (emb_busy),
        .done           (emb_done),
        
        // Token input (directly from external input when input_mode=0)
        .token_valid    (token_in_valid && !input_mode && state == ST_EMBED),
        .token_id       (token_in_id),
        .token_pos      ({{4{1'b0}}, position}),  // Zero-extend 8-bit position to 12-bit
        .token_ready    (token_in_ready_emb),
        
        // Token embedding memory
        .tok_mem_req    (tok_emb_mem_req),
        .tok_mem_addr   (tok_emb_mem_addr),
        .tok_mem_len    (tok_emb_mem_len),
        .tok_mem_grant  (tok_emb_mem_grant),
        .tok_mem_valid  (tok_emb_mem_valid),
        .tok_mem_data   (tok_emb_mem_data),
        .tok_mem_ready  (tok_emb_mem_ready),
        
        // Position embedding memory
        .pos_mem_addr   (pos_emb_mem_addr),
        .pos_mem_elem   (pos_emb_mem_elem),
        .pos_mem_rd     (pos_emb_mem_rd),
        .pos_mem_data   (pos_emb_mem_data),
        .pos_mem_valid  (pos_emb_mem_valid),
        
        // Output
        .out_valid      (emb_out_valid),
        .out_data       (emb_out_data),
        .out_elem_idx   (emb_out_elem_idx),
        .out_last_elem  (emb_out_last_elem),
        .out_last_token (emb_out_last_token),
        .out_ready      (emb_out_ready)
    );
    
    // =========================================================================
    // Single Transformer Block Instance (weight-shared for all 30 layers)
    // =========================================================================
    // Sequential processing: same hardware block, different weights per layer
    
    wire                    xfmr_token_ready_in;
    reg                     xfmr_token_valid_in;
    wire                    xfmr_token_valid_out;
    reg                     xfmr_token_ready_out;
    wire [TOKEN_DATA_WIDTH-1:0] xfmr_token_data_out;
    wire                    xfmr_token_last_out;
    wire                    xfmr_busy;
    wire                    xfmr_attn_busy;
    wire                    xfmr_mlp_busy;
    
    // KV cache signals for current layer (muxed to external interface)
    wire [NUM_HEADS-1:0]    xfmr_kv_rd;
    wire [NUM_HEADS-1:0]    xfmr_kv_wr;
    wire [NUM_HEADS-1:0]    xfmr_kv_sel;
    wire [NUM_HEADS*SEQ_BITS-1:0] xfmr_kv_addr;
    wire [NUM_HEADS*HEAD_DIM*ACT_WIDTH-1:0] xfmr_kv_wdata;
    reg  [NUM_HEADS*HEAD_DIM*ACT_WIDTH-1:0] xfmr_kv_rdata;
    
    // Weight interface wires
    wire                    xfmr_rms1_gamma_rd_en;
    wire [DIM_BITS-1:0]     xfmr_rms1_gamma_addr;
    wire                    xfmr_rms2_gamma_rd_en;
    wire [DIM_BITS-1:0]     xfmr_rms2_gamma_addr;
    wire                    xfmr_attn_wq_rd_en;
    wire [DIM_BITS-1:0]     xfmr_attn_wq_addr;
    wire                    xfmr_attn_wk_rd_en;
    wire [DIM_BITS-1:0]     xfmr_attn_wk_addr;
    wire                    xfmr_attn_wv_rd_en;
    wire [DIM_BITS-1:0]     xfmr_attn_wv_addr;
    wire                    xfmr_attn_wo_rd_en;
    wire [DIM_BITS-1:0]     xfmr_attn_wo_addr;
    wire                    xfmr_mlp_gate_rd_en;
    wire [DIM_BITS-1:0]     xfmr_mlp_gate_addr;
    wire                    xfmr_mlp_up_rd_en;
    wire [DIM_BITS-1:0]     xfmr_mlp_up_addr;
    wire                    xfmr_mlp_down_rd_en;
    wire [MLP_BITS-1:0]     xfmr_mlp_down_addr;

    transformer_block_llm #(
        .HIDDEN_DIM     (HIDDEN_DIM),
        .NUM_HEADS      (NUM_HEADS),
        .HEAD_DIM       (HEAD_DIM),
        .MLP_HIDDEN     (MLP_HIDDEN),
        .MAX_SEQ        (MAX_SEQ),
        .ACT_WIDTH      (ACT_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .LAYER_BITS     (5)
    ) u_transformer (
        .clk            (clk),
        .rst_n          (rst_n),
        
        // Token streaming input
        .token_valid_in (xfmr_token_valid_in),
        .token_ready_in (xfmr_token_ready_in),
        .token_data_in  (layer_input),
        .token_last_in  (1'b1),  // Single token processing
        
        // Token streaming output
        .token_valid_out(xfmr_token_valid_out),
        .token_ready_out(xfmr_token_ready_out),
        .token_data_out (xfmr_token_data_out),
        .token_last_out (xfmr_token_last_out),
        
        // Control
        .layer_idx      (layer_idx),
        .seq_len        (seq_len_reg),
        .position       (position),
        .is_prefill     (is_prefill),
        
        // RMS Norm weights
        .rms1_gamma_rd_en(xfmr_rms1_gamma_rd_en),
        .rms1_gamma_addr (xfmr_rms1_gamma_addr),
        .rms1_gamma_data (weight_rms_data),
        .rms2_gamma_rd_en(xfmr_rms2_gamma_rd_en),
        .rms2_gamma_addr (xfmr_rms2_gamma_addr),
        .rms2_gamma_data (weight_rms_data),
        
        // Attention weights
        .attn_wq_rd_en  (xfmr_attn_wq_rd_en),
        .attn_wq_addr   (xfmr_attn_wq_addr),
        .attn_wq_data   (weight_attn_data),
        .attn_wk_rd_en  (xfmr_attn_wk_rd_en),
        .attn_wk_addr   (xfmr_attn_wk_addr),
        .attn_wk_data   (weight_attn_data),
        .attn_wv_rd_en  (xfmr_attn_wv_rd_en),
        .attn_wv_addr   (xfmr_attn_wv_addr),
        .attn_wv_data   (weight_attn_data),
        .attn_wo_rd_en  (xfmr_attn_wo_rd_en),
        .attn_wo_addr   (xfmr_attn_wo_addr),
        .attn_wo_data   (weight_attn_data),

        // MLP weights
        .mlp_gate_weight_rd_en(xfmr_mlp_gate_rd_en),
        .mlp_gate_weight_addr (xfmr_mlp_gate_addr),
        .mlp_gate_weight_data (weight_mlp_gate_data),
        .mlp_up_weight_rd_en  (xfmr_mlp_up_rd_en),
        .mlp_up_weight_addr   (xfmr_mlp_up_addr),
        .mlp_up_weight_data   (weight_mlp_up_data),
        .mlp_down_weight_rd_en(xfmr_mlp_down_rd_en),
        .mlp_down_weight_addr (xfmr_mlp_down_addr),
        .mlp_down_weight_data (weight_mlp_down_data),
        
        // KV cache
        .kv_rd          (xfmr_kv_rd),
        .kv_wr          (xfmr_kv_wr),
        .kv_sel         (xfmr_kv_sel),
        .kv_addr        (xfmr_kv_addr),
        .kv_wdata       (xfmr_kv_wdata),
        .kv_rdata       (xfmr_kv_rdata),
        
        // Status
        .busy           (xfmr_busy),
        .attn_busy      (xfmr_attn_busy),
        .mlp_busy       (xfmr_mlp_busy)
    );
    
    // =========================================================================
    // Final RMS Norm Instance
    // =========================================================================
    reg                     final_rms_valid_in;
    wire                    final_rms_ready_in;
    wire [TOKEN_DATA_WIDTH-1:0] final_rms_out;
    wire                    final_rms_valid_out;
    reg                     final_rms_ready_out;
    
    rms_norm_block #(
        .DIM            (HIDDEN_DIM),
        .ACT_WIDTH      (ACT_WIDTH),
        .ACC_WIDTH      (32),
        .FRAC_BITS      (8)
    ) u_final_rms_norm (
        .clk            (clk),
        .rst_n          (rst_n),
        .x_in           (hidden_state),
        .valid_in       (final_rms_valid_in),
        .ready_in       (final_rms_ready_in),
        .gamma          (weight_final_rms_data),
        .y_out          (final_rms_out),
        .valid_out      (final_rms_valid_out),
        .ready_out      (final_rms_ready_out)
    );

    // =========================================================================
    // LM Head Projection (ternary matmul: 576 → 49152)
    // =========================================================================
    reg [VOCAB_BITS-1:0]    lm_head_row;        // Current vocabulary row
    reg [ACC_WIDTH-1:0]     lm_head_acc;        // Accumulator for dot product
    reg [ACT_WIDTH-1:0]     lm_head_logit;      // Computed logit
    reg                     lm_head_valid;
    
    // Ternary dot product computation
    wire signed [ACC_WIDTH-1:0] ternary_dot;
    
    // Compute ternary dot product for LM head
    // Each weight is 2 bits: 00=0, 01=+1, 11=-1
    reg signed [ACC_WIDTH-1:0] dot_sum;
    integer j;
    
    always @(*) begin
        dot_sum = 0;
        for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
            case (weight_lm_head_data[j*2 +: 2])
                2'b01: dot_sum = dot_sum + $signed({{(ACC_WIDTH-ACT_WIDTH){final_rms_out[j*ACT_WIDTH + ACT_WIDTH-1]}}, 
                                                    final_rms_out[j*ACT_WIDTH +: ACT_WIDTH]});
                2'b11: dot_sum = dot_sum - $signed({{(ACC_WIDTH-ACT_WIDTH){final_rms_out[j*ACT_WIDTH + ACT_WIDTH-1]}}, 
                                                    final_rms_out[j*ACT_WIDTH +: ACT_WIDTH]});
                default: ; // 0 weight, no contribution
            endcase
        end
    end
    
    assign ternary_dot = dot_sum;
    
    // =========================================================================
    // Top-K Sampler
    // =========================================================================
    reg [ACT_WIDTH-1:0]     top_k_values [0:TOP_K-1];
    reg [VOCAB_BITS-1:0]    top_k_indices [0:TOP_K-1];
    reg [3:0]               top_k_count;
    reg                     sampling_done;
    
    // Simple insertion sort for top-K tracking
    reg [ACT_WIDTH-1:0]     current_logit_value;
    reg [VOCAB_BITS-1:0]    current_logit_idx;

    // =========================================================================
    // KV Cache Arbiter
    // =========================================================================
    // Mux transformer block KV requests to external memory interface
    reg [3:0]               active_head;        // Currently serviced head
    reg                     kv_req_pending;
    
    // Find first requesting head (priority encoder)
    wire [NUM_HEADS-1:0]    kv_request = xfmr_kv_rd | xfmr_kv_wr;
    reg [3:0]               first_req_head;
    
    integer h;
    always @(*) begin
        first_req_head = 0;
        for (h = NUM_HEADS-1; h >= 0; h = h - 1) begin
            if (kv_request[h]) first_req_head = h[3:0];
        end
    end
    
    // KV cache interface assignments
    assign kv_cache_req   = |kv_request && (state == ST_LAYERS);
    assign kv_cache_wr    = xfmr_kv_wr[active_head];
    assign kv_cache_layer = layer_idx;
    assign kv_cache_head  = active_head;
    assign kv_cache_sel   = xfmr_kv_sel[active_head];
    assign kv_cache_pos   = xfmr_kv_addr[active_head*SEQ_BITS +: SEQ_BITS];
    assign kv_cache_wdata = xfmr_kv_wdata[active_head*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH];
    
    // Route read data back to appropriate head
    always @(posedge clk) begin
        if (kv_cache_rvalid) begin
            xfmr_kv_rdata[active_head*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH] <= kv_cache_rdata;
        end
    end
    
    // =========================================================================
    // Weight Interface Routing
    // =========================================================================
    assign weight_rms_rd    = xfmr_rms1_gamma_rd_en || xfmr_rms2_gamma_rd_en;
    assign weight_rms_layer = layer_idx;
    assign weight_rms_sel   = xfmr_rms2_gamma_rd_en;  // 0=pre-attn, 1=pre-mlp
    
    assign weight_attn_rd   = xfmr_attn_wq_rd_en || xfmr_attn_wk_rd_en || 
                              xfmr_attn_wv_rd_en || xfmr_attn_wo_rd_en;
    assign weight_attn_layer = layer_idx;
    assign weight_attn_sel  = xfmr_attn_wq_rd_en ? 2'd0 :
                              xfmr_attn_wk_rd_en ? 2'd1 :
                              xfmr_attn_wv_rd_en ? 2'd2 : 2'd3;
    assign weight_attn_addr = xfmr_attn_wq_rd_en ? xfmr_attn_wq_addr :
                              xfmr_attn_wk_rd_en ? xfmr_attn_wk_addr :
                              xfmr_attn_wv_rd_en ? xfmr_attn_wv_addr : xfmr_attn_wo_addr;

    assign weight_mlp_rd    = xfmr_mlp_gate_rd_en || xfmr_mlp_up_rd_en || xfmr_mlp_down_rd_en;
    assign weight_mlp_layer = layer_idx;
    assign weight_mlp_sel   = xfmr_mlp_gate_rd_en ? 2'd0 :
                              xfmr_mlp_up_rd_en   ? 2'd1 : 2'd2;
    assign weight_mlp_addr  = xfmr_mlp_down_rd_en ? {{(MLP_BITS-DIM_BITS){1'b0}}, xfmr_mlp_down_addr} :
                              xfmr_mlp_gate_rd_en ? {{(MLP_BITS-DIM_BITS){1'b0}}, xfmr_mlp_gate_addr} :
                                                    {{(MLP_BITS-DIM_BITS){1'b0}}, xfmr_mlp_up_addr};
    
    assign weight_final_rms_rd = (state == ST_FINAL_NORM);
    assign weight_lm_head_rd   = (state == ST_LM_HEAD);
    assign weight_lm_head_addr = lm_head_row;
    
    // =========================================================================
    // Output Interface Assignments
    // =========================================================================
    assign token_in_ready = (state == ST_EMBED) && !input_mode && token_in_ready_emb;
    assign emb_in_ready   = (state == ST_EMBED) && input_mode;
    
    assign logits_valid   = lm_head_valid && (state == ST_OUTPUT_LOGITS);
    assign logits_data    = lm_head_logit;
    assign logits_idx     = lm_head_row;
    assign logits_last    = (lm_head_row == VOCAB_SIZE - 1);
    
    assign gen_token_valid = (state == ST_OUTPUT_TOKEN);
    assign gen_token_id    = sampled_token;
    assign gen_token_eos   = sampled_eos;
    
    assign busy           = (state != ST_IDLE);
    assign done           = (state == ST_DONE);
    assign current_layer  = layer_idx;
    
    // =========================================================================
    // Embedding Accumulation Buffer
    // =========================================================================
    // Accumulate streaming embedding output into hidden_state
    reg [DIM_BITS-1:0] emb_elem_cnt;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            emb_elem_cnt <= 0;
        end else if (state == ST_EMBED && emb_out_valid && emb_out_ready) begin
            hidden_state[emb_out_elem_idx*ACT_WIDTH +: ACT_WIDTH] <= emb_out_data;
            if (emb_out_last_elem) begin
                emb_elem_cnt <= 0;
            end else begin
                emb_elem_cnt <= emb_elem_cnt + 1'b1;
            end
        end
    end

    // =========================================================================
    // Main State Machine
    // =========================================================================
    integer k;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            layer_idx <= 5'd0;
            position <= {SEQ_BITS{1'b0}};
            seq_len_reg <= {SEQ_BITS{1'b0}};
            tokens_generated <= {SEQ_BITS{1'b0}};
            max_new_tokens_reg <= {SEQ_BITS{1'b0}};
            is_prefill <= 1'b0;
            use_top_k_reg <= 1'b0;
            
            hidden_state <= {TOKEN_DATA_WIDTH{1'b0}};
            layer_input <= {TOKEN_DATA_WIDTH{1'b0}};
            layer_output <= {TOKEN_DATA_WIDTH{1'b0}};
            
            sampled_token <= {TOKEN_BITS{1'b0}};
            sampled_eos <= 1'b0;
            
            emb_start <= 1'b0;
            emb_batch_size <= 6'd0;
            emb_out_ready <= 1'b0;
            
            xfmr_token_valid_in <= 1'b0;
            xfmr_token_ready_out <= 1'b0;
            
            final_rms_valid_in <= 1'b0;
            final_rms_ready_out <= 1'b0;
            
            lm_head_row <= {VOCAB_BITS{1'b0}};
            lm_head_acc <= {ACC_WIDTH{1'b0}};
            lm_head_logit <= {ACT_WIDTH{1'b0}};
            lm_head_valid <= 1'b0;
            
            top_k_count <= 4'd0;
            sampling_done <= 1'b0;
            
            active_head <= 4'd0;
            kv_req_pending <= 1'b0;
            
            for (k = 0; k < TOP_K; k = k + 1) begin
                top_k_values[k] <= {ACT_WIDTH{1'b0}};
                top_k_indices[k] <= {VOCAB_BITS{1'b0}};
            end
            
        end else begin
            // Default deassertions
            emb_start <= 1'b0;

            case (state)
                // ---------------------------------------------------------
                ST_IDLE: begin
                    if (start) begin
                        seq_len_reg <= seq_len;
                        max_new_tokens_reg <= max_new_tokens;
                        use_top_k_reg <= use_top_k;
                        is_prefill <= !generate_mode;
                        position <= {SEQ_BITS{1'b0}};
                        layer_idx <= 5'd0;
                        tokens_generated <= {SEQ_BITS{1'b0}};
                        
                        if (input_mode) begin
                            // Embeddings from projector, skip embedding lookup
                            state <= ST_EMBED;
                            emb_out_ready <= 1'b1;
                        end else begin
                            // Token IDs, need embedding lookup
                            state <= ST_EMBED;
                            emb_start <= 1'b1;
                            emb_batch_size <= 6'd1;  // One token at a time
                            emb_out_ready <= 1'b1;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                ST_EMBED: begin
                    if (input_mode) begin
                        // Direct embedding input from vision projector
                        if (emb_in_valid) begin
                            hidden_state <= emb_in_data;
                            if (emb_in_last || !is_prefill) begin
                                state <= ST_LAYERS;
                                layer_idx <= 5'd0;
                                layer_input <= emb_in_data;
                                xfmr_token_valid_in <= 1'b1;
                            end else begin
                                position <= position + 1'b1;
                            end
                        end
                    end else begin
                        // Embedding from lookup
                        if (emb_done) begin
                            state <= ST_LAYERS;
                            layer_idx <= 5'd0;
                            layer_input <= hidden_state;
                            xfmr_token_valid_in <= 1'b1;
                            emb_out_ready <= 1'b0;
                        end
                    end
                end

                // ---------------------------------------------------------
                ST_LAYERS: begin
                    // Handle KV cache arbitration
                    if (|kv_request) begin
                        active_head <= first_req_head;
                        kv_req_pending <= 1'b1;
                    end else if (kv_cache_grant) begin
                        kv_req_pending <= 1'b0;
                    end
                    
                    // Wait for transformer to accept input
                    if (xfmr_token_ready_in) begin
                        xfmr_token_valid_in <= 1'b0;
                    end
                    
                    // Wait for layer output
                    xfmr_token_ready_out <= 1'b1;
                    
                    if (xfmr_token_valid_out) begin
                        layer_output <= xfmr_token_data_out;
                        xfmr_token_ready_out <= 1'b0;
                        
                        if (layer_idx == NUM_LAYERS - 1) begin
                            // All layers done
                            hidden_state <= xfmr_token_data_out;
                            state <= ST_FINAL_NORM;
                            final_rms_valid_in <= 1'b1;
                        end else begin
                            // Move to next layer
                            layer_idx <= layer_idx + 1'b1;
                            layer_input <= xfmr_token_data_out;
                            xfmr_token_valid_in <= 1'b1;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                ST_FINAL_NORM: begin
                    if (final_rms_ready_in) begin
                        final_rms_valid_in <= 1'b0;
                    end
                    
                    final_rms_ready_out <= 1'b1;
                    
                    if (final_rms_valid_out) begin
                        final_rms_ready_out <= 1'b0;
                        hidden_state <= final_rms_out;
                        state <= ST_LM_HEAD;
                        lm_head_row <= {VOCAB_BITS{1'b0}};
                        
                        // Reset top-K tracker
                        for (k = 0; k < TOP_K; k = k + 1) begin
                            top_k_values[k] <= {1'b1, {(ACT_WIDTH-1){1'b0}}}; // Min value
                            top_k_indices[k] <= {VOCAB_BITS{1'b0}};
                        end
                        top_k_count <= 4'd0;
                    end
                end

                // ---------------------------------------------------------
                ST_LM_HEAD: begin
                    // Compute logit for current vocabulary row
                    if (weight_lm_head_valid) begin
                        // Saturate accumulator to 8-bit logit
                        if (ternary_dot > $signed({{(ACC_WIDTH-ACT_WIDTH+1){1'b0}}, {(ACT_WIDTH-1){1'b1}}}))
                            lm_head_logit <= {1'b0, {(ACT_WIDTH-1){1'b1}}};
                        else if (ternary_dot < $signed({{(ACC_WIDTH-ACT_WIDTH+1){1'b1}}, {(ACT_WIDTH-1){1'b0}}}))
                            lm_head_logit <= {1'b1, {(ACT_WIDTH-1){1'b0}}};
                        else
                            lm_head_logit <= ternary_dot[ACT_WIDTH-1:0];
                        
                        lm_head_valid <= 1'b1;
                        
                        // Update top-K tracking (simplified insertion)
                        if ($signed(ternary_dot[ACT_WIDTH-1:0]) > $signed(top_k_values[TOP_K-1])) begin
                            top_k_values[TOP_K-1] <= ternary_dot[ACT_WIDTH-1:0];
                            top_k_indices[TOP_K-1] <= lm_head_row;
                            // Simple bubble sort would go here for proper top-K
                        end
                        
                        if (lm_head_row == VOCAB_SIZE - 1) begin
                            state <= ST_SAMPLE;
                            lm_head_valid <= 1'b0;
                        end else begin
                            lm_head_row <= lm_head_row + 1'b1;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                ST_SAMPLE: begin
                    // Select token based on sampling mode
                    if (use_top_k_reg) begin
                        // Top-K sampling: select from top_k_indices[0] (best)
                        // In real implementation, would add randomness
                        sampled_token <= top_k_indices[0];
                    end else begin
                        // Greedy: select highest logit
                        sampled_token <= top_k_indices[0];
                    end
                    
                    // Check for EOS token (typically token 2 or similar)
                    sampled_eos <= (top_k_indices[0] == 16'd2);  // EOS token ID
                    
                    sampling_done <= 1'b1;
                    state <= ST_OUTPUT_TOKEN;
                end

                // ---------------------------------------------------------
                ST_OUTPUT_LOGITS: begin
                    // Optional state for streaming logits out
                    // Currently bypassed, goes directly to sample
                    if (logits_ready && logits_last) begin
                        state <= ST_SAMPLE;
                    end
                end
                
                // ---------------------------------------------------------
                ST_OUTPUT_TOKEN: begin
                    if (gen_token_ready) begin
                        tokens_generated <= tokens_generated + 1'b1;
                        sampling_done <= 1'b0;
                        
                        // Check termination conditions
                        if (sampled_eos || tokens_generated >= max_new_tokens_reg - 1) begin
                            state <= ST_DONE;
                        end else begin
                            state <= ST_NEXT_TOKEN;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                ST_NEXT_TOKEN: begin
                    // Prepare for next token generation
                    position <= position + 1'b1;
                    is_prefill <= 1'b0;  // Now in decode mode
                    layer_idx <= 5'd0;
                    
                    // Feed sampled token back through embedding
                    state <= ST_EMBED;
                    emb_start <= 1'b1;
                    emb_batch_size <= 6'd1;
                    emb_out_ready <= 1'b1;
                end
                
                // ---------------------------------------------------------
                ST_DONE: begin
                    // Wait for acknowledgment, then return to idle
                    state <= ST_IDLE;
                end
                
                // ---------------------------------------------------------
                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
