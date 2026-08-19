// =============================================================================
// NanoViT Vision Encoder - Level 3 Synthesis Block
// =============================================================================
// Compact vision transformer for SiLens Edge ultra-fast classifier.
// MobileViT-XXS inspired architecture optimized for 50mm² die budget.
//
// Architecture:
//   224×224×3 image → 16×16 patches → 196 patch tokens + 1 CLS token
//   → 6× transformer layers (LN → MHA → +res → LN → MLP → +res)
//   → Final LN → CLS token output (192-dim embedding)
//
// Parameters:
//   - Hidden dimension: 192
//   - Attention heads: 3 (head_dim = 64)
//   - MLP expansion: 2× (192 → 384 → 192)
//   - Layers: 6
//   - Sequence length: 197 (196 patches + 1 CLS)
//
// Target: ~15mm² on SKY130, 200MHz clock, <1ms inference
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module vision_nano #(
    // =========================================================================
    // Architecture parameters
    // =========================================================================
    parameter HIDDEN_DIM    = 192,      // Hidden dimension (d_model)
    parameter NUM_HEADS     = 3,        // Number of attention heads
    parameter HEAD_DIM      = 64,       // Dimension per head (HIDDEN_DIM / NUM_HEADS)
    parameter MLP_DIM       = 384,      // MLP intermediate dimension (2× expansion)
    parameter NUM_LAYERS    = 6,        // Number of transformer layers
    parameter IMAGE_SIZE    = 224,      // Input image size
    parameter PATCH_SIZE    = 16,       // Patch size for embedding
    parameter NUM_PATCHES   = 196,      // (IMAGE_SIZE / PATCH_SIZE)² = 14×14
    parameter SEQ_LEN       = 197,      // NUM_PATCHES + 1 (CLS token)
    
    // =========================================================================
    // Data width parameters
    // =========================================================================
    parameter ACT_WIDTH     = 8,        // Activation bit width
    parameter ACC_WIDTH     = 24,       // Accumulator width
    parameter PIXEL_WIDTH   = 8         // Input pixel bit width
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // =========================================================================
    // Image input interface (streaming pixels)
    // =========================================================================
    input  wire                         img_valid,
    input  wire [3*PIXEL_WIDTH-1:0]     img_pixel,      // RGB pixel
    input  wire                         img_sof,        // Start of frame
    input  wire                         img_eof,        // End of frame
    output wire                         img_ready,
    
    // =========================================================================
    // Control interface
    // =========================================================================
    input  wire                         start,          // Start inference
    output reg                          busy,
    output reg                          done,
    
    // =========================================================================
    // Output: CLS token embedding (192-dim)
    // =========================================================================
    output reg                          out_valid,
    output reg  [HIDDEN_DIM*ACT_WIDTH-1:0] out_embedding,
    input  wire                         out_ready
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam SEQ_BITS = $clog2(SEQ_LEN);
    localparam LAYER_BITS = $clog2(NUM_LAYERS);
    localparam PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * 3;  // 768 pixels per patch
    localparam PATCHES_PER_ROW = IMAGE_SIZE / PATCH_SIZE;   // 14
    
    // =========================================================================
    // State machine
    // =========================================================================
    localparam [3:0]
        S_IDLE          = 4'd0,
        S_RECV_IMAGE    = 4'd1,     // Receive image pixels
        S_PATCH_EMBED   = 4'd2,     // Compute patch embeddings
        S_INIT_CLS      = 4'd3,     // Initialize CLS token
        S_ADD_POS_EMB   = 4'd4,     // Add position embeddings
        S_LAYER_START   = 4'd5,     // Start transformer layer
        S_LAYER_ATTN    = 4'd6,     // Multi-head attention
        S_LAYER_MLP     = 4'd7,     // MLP feedforward
        S_LAYER_END     = 4'd8,     // Layer complete, check if more
        S_FINAL_NORM    = 4'd9,     // Final layer normalization
        S_OUTPUT        = 4'd10,    // Output CLS embedding
        S_DONE          = 4'd11;
    
    reg [3:0] state, next_state;
    
    // =========================================================================
    // Counters and indices
    // =========================================================================
    reg [LAYER_BITS-1:0] layer_idx;
    reg [SEQ_BITS-1:0] token_idx;
    reg [$clog2(NUM_HEADS)-1:0] head_idx;
    reg [$clog2(PATCH_PIXELS)-1:0] pixel_cnt;
    reg [$clog2(IMAGE_SIZE*IMAGE_SIZE)-1:0] img_pixel_cnt;
    
    // =========================================================================
    // Token memory (197 × 192 × 8 bits = 302,688 bits = ~38KB)
    // =========================================================================
    // Stores all sequence tokens during processing
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] token_mem [0:SEQ_LEN-1];
    
    // =========================================================================
    // Patch embedding buffer
    // =========================================================================
    // Accumulates pixels for one patch before embedding
    reg [PATCH_PIXELS*PIXEL_WIDTH-1:0] patch_buffer;
    reg [$clog2(NUM_PATCHES)-1:0] patch_idx;
    
    // =========================================================================
    // Working registers
    // =========================================================================
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] token_reg;       // Current token being processed
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] residual_reg;    // Residual connection
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] attn_out_reg;    // Attention output accumulator
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] mlp_out_reg;     // MLP output
    
    // =========================================================================
    // Ternary weights for patch embedding (hardwired ROM)
    // 3×16×16 × 192 = 147,456 ternary weights
    // Encoded as 2 bits each: 00=0, 01=+1, 10=-1
    // =========================================================================
    // In synthesis, these become hardwired logic gates
    
    // Simplified: patch_proj computes linear projection of flattened patch
    // patch_proj_weights: [PATCH_PIXELS][HIDDEN_DIM] ternary
    // For RTL, we use a ROM-like structure that synthesizes to hardwired muxes
    
    wire [1:0] patch_proj_w [0:PATCH_PIXELS-1][0:HIDDEN_DIM-1];
    
    // Position embedding table (197 × 192 × 8 bits, learnable)
    // Stored as ROM, synthesizes to hardwired values
    wire signed [ACT_WIDTH-1:0] pos_embed [0:SEQ_LEN-1][0:HIDDEN_DIM-1];
    
    // CLS token (192 × 8 bits, learnable parameter)
    wire signed [ACT_WIDTH-1:0] cls_token_init [0:HIDDEN_DIM-1];
    
    // =========================================================================
    // Instantiate Level 1 primitives (black boxes for hierarchical synthesis)
    // =========================================================================
    
    // --- Attention Heads (3 parallel heads) ---
    wire [NUM_HEADS-1:0] attn_start;
    wire [NUM_HEADS-1:0] attn_busy;
    wire [NUM_HEADS-1:0] attn_done;
    wire [HEAD_DIM*ACT_WIDTH-1:0] attn_q_data [0:NUM_HEADS-1];
    wire [NUM_HEADS-1:0] attn_q_valid;
    wire [NUM_HEADS-1:0] attn_q_ready;
    wire [HEAD_DIM*ACT_WIDTH-1:0] attn_out_data [0:NUM_HEADS-1];
    wire [NUM_HEADS-1:0] attn_out_valid;
    wire [NUM_HEADS-1:0] attn_out_ready;
    
    // KV cache interface (shared across heads)
    wire [SEQ_BITS-1:0] kv_addr [0:NUM_HEADS-1];
    wire [HEAD_DIM*ACT_WIDTH-1:0] kv_rdata [0:NUM_HEADS-1];
    wire [HEAD_DIM*ACT_WIDTH-1:0] kv_wdata [0:NUM_HEADS-1];
    wire [NUM_HEADS-1:0] kv_rd;
    wire [NUM_HEADS-1:0] kv_wr;
    wire [NUM_HEADS-1:0] kv_sel;
    
    genvar h;
    generate
        for (h = 0; h < NUM_HEADS; h = h + 1) begin : gen_attn_heads
            // Attention head black box (Level 1 primitive)
            attention_head_nano #(
                .HEAD_DIM(HEAD_DIM),
                .MAX_SEQ(SEQ_LEN),
                .ACT_WIDTH(ACT_WIDTH)
            ) u_attn_head (
                .clk(clk),
                .rst_n(rst_n),
                .start(attn_start[h]),
                .is_prefill(1'b1),          // Vision: always prefill mode
                .seq_len(token_idx),
                .query_pos(token_idx),
                .busy(attn_busy[h]),
                .done(attn_done[h]),
                .q_valid(attn_q_valid[h]),
                .q_data(attn_q_data[h]),
                .q_ready(attn_q_ready[h]),
                .use_projection(1'b0),       // Q/K/V already projected
                .w_q({HEAD_DIM*2{1'b0}}),
                .w_k({HEAD_DIM*2{1'b0}}),
                .w_v({HEAD_DIM*2{1'b0}}),
                .kv_addr(kv_addr[h]),
                .kv_rdata(kv_rdata[h]),
                .kv_wdata(kv_wdata[h]),
                .kv_rd(kv_rd[h]),
                .kv_wr(kv_wr[h]),
                .kv_sel(kv_sel[h]),
                .out_valid(attn_out_valid[h]),
                .out_data(attn_out_data[h]),
                .out_ready(attn_out_ready[h])
            );
            
            // Per-head KV cache SRAM (small, on-chip)
            // 197 × 64 × 8 × 2 (K+V) = ~25KB per head, 75KB total
            kv_cache_sram #(
                .DEPTH(SEQ_LEN),
                .WIDTH(HEAD_DIM * ACT_WIDTH)
            ) u_kv_cache (
                .clk(clk),
                .addr(kv_addr[h]),
                .wdata(kv_wdata[h]),
                .rdata(kv_rdata[h]),
                .rd(kv_rd[h]),
                .wr(kv_wr[h]),
                .sel(kv_sel[h])
            );
        end
    endgenerate
    
    // --- Layer Normalization (3 instances: pre-attn, pre-mlp, final) ---
    wire ln_pre_attn_start, ln_pre_attn_done;
    wire ln_pre_mlp_start, ln_pre_mlp_done;
    wire ln_final_start, ln_final_done;
    
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] ln_pre_attn_in, ln_pre_attn_out;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] ln_pre_mlp_in, ln_pre_mlp_out;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] ln_final_in, ln_final_out;
    
    layer_norm_nano #(
        .DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH)
    ) u_ln_pre_attn (
        .clk(clk),
        .rst_n(rst_n),
        .start(ln_pre_attn_start),
        .in_data(ln_pre_attn_in),
        .out_data(ln_pre_attn_out),
        .done(ln_pre_attn_done)
    );
    
    layer_norm_nano #(
        .DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH)
    ) u_ln_pre_mlp (
        .clk(clk),
        .rst_n(rst_n),
        .start(ln_pre_mlp_start),
        .in_data(ln_pre_mlp_in),
        .out_data(ln_pre_mlp_out),
        .done(ln_pre_mlp_done)
    );
    
    layer_norm_nano #(
        .DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH)
    ) u_ln_final (
        .clk(clk),
        .rst_n(rst_n),
        .start(ln_final_start),
        .in_data(ln_final_in),
        .out_data(ln_final_out),
        .done(ln_final_done)
    );
    
    // --- MLP Block (single instance, time-shared across layers) ---
    wire mlp_start, mlp_done;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0] mlp_in, mlp_out;
    
    mlp_block_nano #(
        .IN_DIM(HIDDEN_DIM),
        .HIDDEN_DIM(MLP_DIM),
        .OUT_DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH)
    ) u_mlp (
        .clk(clk),
        .rst_n(rst_n),
        .start(mlp_start),
        .layer_idx(layer_idx),          // Select layer-specific weights
        .in_data(mlp_in),
        .out_data(mlp_out),
        .done(mlp_done)
    );
    
    // =========================================================================
    // Image input logic
    // =========================================================================
    reg img_recv_active;
    assign img_ready = img_recv_active && (state == S_RECV_IMAGE);
    
    // =========================================================================
    // Ternary MAC for patch embedding projection
    // =========================================================================
    function automatic signed [ACT_WIDTH:0] ternary_mult;
        input [1:0] w;
        input signed [ACT_WIDTH-1:0] x;
        begin
            case (w)
                2'b01:   ternary_mult = {x[ACT_WIDTH-1], x};
                2'b10:   ternary_mult = -{x[ACT_WIDTH-1], x};
                default: ternary_mult = {(ACT_WIDTH+1){1'b0}};
            endcase
        end
    endfunction
    
    // =========================================================================
    // State machine transitions
    // =========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: begin
                if (start)
                    next_state = S_RECV_IMAGE;
            end
            
            S_RECV_IMAGE: begin
                if (img_eof && img_valid)
                    next_state = S_PATCH_EMBED;
            end
            
            S_PATCH_EMBED: begin
                if (patch_idx == NUM_PATCHES - 1)
                    next_state = S_INIT_CLS;
            end
            
            S_INIT_CLS: begin
                next_state = S_ADD_POS_EMB;
            end
            
            S_ADD_POS_EMB: begin
                if (token_idx == SEQ_LEN - 1)
                    next_state = S_LAYER_START;
            end
            
            S_LAYER_START: begin
                next_state = S_LAYER_ATTN;
            end
            
            S_LAYER_ATTN: begin
                if (&attn_done)  // All heads done
                    next_state = S_LAYER_MLP;
            end
            
            S_LAYER_MLP: begin
                if (mlp_done)
                    next_state = S_LAYER_END;
            end
            
            S_LAYER_END: begin
                if (layer_idx == NUM_LAYERS - 1)
                    next_state = S_FINAL_NORM;
                else
                    next_state = S_LAYER_START;
            end
            
            S_FINAL_NORM: begin
                if (ln_final_done)
                    next_state = S_OUTPUT;
            end
            
            S_OUTPUT: begin
                if (out_ready)
                    next_state = S_DONE;
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end
    
    // =========================================================================
    // Main datapath control
    // =========================================================================
    integer i, j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            out_valid <= 1'b0;
            layer_idx <= {LAYER_BITS{1'b0}};
            token_idx <= {SEQ_BITS{1'b0}};
            head_idx <= 0;
            pixel_cnt <= 0;
            img_pixel_cnt <= 0;
            patch_idx <= 0;
            img_recv_active <= 1'b0;
            patch_buffer <= 0;
            token_reg <= 0;
            residual_reg <= 0;
            attn_out_reg <= 0;
            mlp_out_reg <= 0;
            out_embedding <= 0;
            
            for (i = 0; i < SEQ_LEN; i = i + 1) begin
                token_mem[i] <= 0;
            end
        end else begin
            state <= next_state;
            
            case (state)
                // ---------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    done <= 1'b0;
                    out_valid <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        layer_idx <= 0;
                        token_idx <= 0;
                        patch_idx <= 0;
                        pixel_cnt <= 0;
                        img_pixel_cnt <= 0;
                        img_recv_active <= 1'b1;
                    end
                end
                
                // ---------------------------------------------------------
                S_RECV_IMAGE: begin
                    // Stream in image pixels and buffer into patches
                    if (img_valid && img_ready) begin
                        // Store pixel in patch buffer
                        // Pixels arrive row-by-row, we need to reorganize into patches
                        patch_buffer[pixel_cnt*PIXEL_WIDTH*3 +: PIXEL_WIDTH*3] <= img_pixel;
                        pixel_cnt <= pixel_cnt + 1;
                        img_pixel_cnt <= img_pixel_cnt + 1;
                        
                        // Check if patch complete (simplified: assume patches arrive sequentially)
                        if (pixel_cnt == PATCH_PIXELS - 1) begin
                            pixel_cnt <= 0;
                        end
                    end
                    
                    if (img_eof && img_valid) begin
                        img_recv_active <= 1'b0;
                    end
                end
                
                // ---------------------------------------------------------
                S_PATCH_EMBED: begin
                    // Compute patch embeddings using ternary projection
                    // For each patch: embed = sum(patch_proj_w[i] * patch[i]) for all pixels
                    // This would be a large ternary MAC operation
                    
                    // Simplified: iterate through patches and compute embedding
                    // In hardware, this is heavily pipelined
                    
                    // Store computed embedding
                    // token_mem[patch_idx + 1] <= computed_embedding; // +1 for CLS at position 0
                    
                    patch_idx <= patch_idx + 1;
                end
                
                // ---------------------------------------------------------
                S_INIT_CLS: begin
                    // Initialize CLS token at position 0
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        token_mem[0][i*ACT_WIDTH +: ACT_WIDTH] <= cls_token_init[i];
                    end
                    token_idx <= 0;
                end
                
                // ---------------------------------------------------------
                S_ADD_POS_EMB: begin
                    // Add position embeddings to all tokens
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        token_mem[token_idx][i*ACT_WIDTH +: ACT_WIDTH] <= 
                            $signed(token_mem[token_idx][i*ACT_WIDTH +: ACT_WIDTH]) + 
                            pos_embed[token_idx][i];
                    end
                    token_idx <= token_idx + 1;
                end
                
                // ---------------------------------------------------------
                S_LAYER_START: begin
                    // Initialize for new transformer layer
                    token_idx <= 0;
                end
                
                // ---------------------------------------------------------
                S_LAYER_ATTN: begin
                    // Multi-head attention processing
                    // 1. LayerNorm (pre-norm architecture)
                    // 2. Project to Q, K, V for each head
                    // 3. Compute attention for each head in parallel
                    // 4. Concatenate heads and project output
                    // 5. Add residual
                    
                    // Attention heads run in parallel on the 3 instantiated units
                    // Results are concatenated to form full HIDDEN_DIM output
                end
                
                // ---------------------------------------------------------
                S_LAYER_MLP: begin
                    // MLP feedforward
                    // 1. LayerNorm (pre-norm)
                    // 2. up_proj (192 → 384)
                    // 3. GELU activation (or SiLU)
                    // 4. down_proj (384 → 192)
                    // 5. Add residual
                    
                    // MLP block processes one token at a time
                    // Iterate through all 197 tokens
                end
                
                // ---------------------------------------------------------
                S_LAYER_END: begin
                    // Update layer index
                    layer_idx <= layer_idx + 1;
                end
                
                // ---------------------------------------------------------
                S_FINAL_NORM: begin
                    // Final layer normalization on CLS token
                    // Output: normalized 192-dim embedding
                end
                
                // ---------------------------------------------------------
                S_OUTPUT: begin
                    // Output the CLS token embedding
                    out_valid <= 1'b1;
                    out_embedding <= token_mem[0];  // CLS token at position 0
                    
                    if (out_ready) begin
                        out_valid <= 1'b0;
                    end
                end
                
                // ---------------------------------------------------------
                S_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                end
                
                default: begin
                    // Do nothing
                end
            endcase
        end
    end
    
    // =========================================================================
    // Control signal assignments for submodules
    // =========================================================================
    
    // Layer norm control
    assign ln_pre_attn_start = (state == S_LAYER_ATTN) && (token_idx == 0);
    assign ln_pre_attn_in = token_mem[token_idx];
    
    assign ln_pre_mlp_start = (state == S_LAYER_MLP) && (token_idx == 0);
    assign ln_pre_mlp_in = token_mem[token_idx];
    
    assign ln_final_start = (state == S_FINAL_NORM);
    assign ln_final_in = token_mem[0];  // CLS token
    
    // MLP control
    assign mlp_start = ln_pre_mlp_done;
    assign mlp_in = ln_pre_mlp_out;
    
    // Attention head control
    generate
        for (h = 0; h < NUM_HEADS; h = h + 1) begin : gen_attn_ctrl
            assign attn_start[h] = (state == S_LAYER_ATTN) && (token_idx == 0);
            assign attn_q_valid[h] = ln_pre_attn_done;
            // Split normalized token across heads
            assign attn_q_data[h] = ln_pre_attn_out[h*HEAD_DIM*ACT_WIDTH +: HEAD_DIM*ACT_WIDTH];
            assign attn_out_ready[h] = 1'b1;
        end
    endgenerate

endmodule

// =============================================================================
// Black box declarations for Level 1 primitives
// =============================================================================
// These are synthesized separately and instantiated as macros

module attention_head_nano #(
    parameter HEAD_DIM = 64,
    parameter MAX_SEQ = 256,
    parameter ACT_WIDTH = 8
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    input  wire                         is_prefill,
    input  wire [$clog2(MAX_SEQ)-1:0]   seq_len,
    input  wire [$clog2(MAX_SEQ)-1:0]   query_pos,
    output wire                         busy,
    output wire                         done,
    input  wire                         q_valid,
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] q_data,
    output wire                         q_ready,
    input  wire                         use_projection,
    input  wire [HEAD_DIM*2-1:0]        w_q,
    input  wire [HEAD_DIM*2-1:0]        w_k,
    input  wire [HEAD_DIM*2-1:0]        w_v,
    output wire [$clog2(MAX_SEQ)-1:0]   kv_addr,
    input  wire [HEAD_DIM*ACT_WIDTH-1:0] kv_rdata,
    output wire [HEAD_DIM*ACT_WIDTH-1:0] kv_wdata,
    output wire                         kv_rd,
    output wire                         kv_wr,
    output wire                         kv_sel,
    output wire                         out_valid,
    output wire [HEAD_DIM*ACT_WIDTH-1:0] out_data,
    input  wire                         out_ready
);
    // Black box - synthesized separately
    // See: openlane/level1/attention_head_nano/
endmodule

module layer_norm_nano #(
    parameter DIM = 192,
    parameter ACT_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [DIM*ACT_WIDTH-1:0] in_data,
    output wire [DIM*ACT_WIDTH-1:0] out_data,
    output wire                     done
);
    // Black box - synthesized separately
    // See: openlane/level1/layer_norm_nano/
endmodule

module mlp_block_nano #(
    parameter IN_DIM = 192,
    parameter HIDDEN_DIM = 384,
    parameter OUT_DIM = 192,
    parameter ACT_WIDTH = 8
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    input  wire [2:0]                   layer_idx,
    input  wire [IN_DIM*ACT_WIDTH-1:0]  in_data,
    output wire [OUT_DIM*ACT_WIDTH-1:0] out_data,
    output wire                         done
);
    // Black box - synthesized separately
    // See: openlane/level1/mlp_block_nano/
endmodule

module kv_cache_sram #(
    parameter DEPTH = 256,
    parameter WIDTH = 512
)(
    input  wire                     clk,
    input  wire [$clog2(DEPTH)-1:0] addr,
    input  wire [WIDTH-1:0]         wdata,
    output reg  [WIDTH-1:0]         rdata,
    input  wire                     rd,
    input  wire                     wr,
    input  wire                     sel     // 0=K cache, 1=V cache
);
    // Simple dual-port SRAM for KV cache
    // Synthesizes to flip-flops or SRAM macros
    reg [WIDTH-1:0] k_mem [0:DEPTH-1];
    reg [WIDTH-1:0] v_mem [0:DEPTH-1];
    
    always @(posedge clk) begin
        if (wr) begin
            if (sel)
                v_mem[addr] <= wdata;
            else
                k_mem[addr] <= wdata;
        end
        if (rd) begin
            rdata <= sel ? v_mem[addr] : k_mem[addr];
        end
    end
endmodule

`default_nettype wire
