// =============================================================================
// Vision Subsystem - Level 3 Hierarchical Synthesis Block
// =============================================================================
// Complete SigLIP-B/16 Vision Encoder for SiLens multimodal accelerator.
//
// Architecture:
//   Input Image (384×384×3)
//     → Patch Embedding (16×16 patches = 576 + 1 CLS = 577 patches)
//     → 12× transformer_block_vision (sequential processing)
//     → Final Layer Norm
//     → CLS token output (768-dim) → to projector
//
// Specifications:
//   - Input: 384×384 RGB image (streaming)
//   - Output: 768-dimensional CLS token embedding
//   - Patch size: 16×16 pixels
//   - Number of patches: 576 (24×24 grid) + 1 CLS = 577
//   - Hidden dimension: 768
//   - Transformer layers: 12
//   - Target area: ~250mm² (15800µm × 15800µm)
//
// Level 2 macro dependencies:
//   - 12× transformer_block_vision (~20mm² each)
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module vision_subsystem #(
    parameter IMAGE_SIZE     = 384,     // Input image dimension (384×384)
    parameter PATCH_SIZE     = 16,      // Patch size (16×16 pixels)
    parameter NUM_PATCHES    = 576,     // Number of patches (24×24)
    parameter SEQ_LEN        = 577,     // Total sequence length (patches + CLS)
    parameter HIDDEN_DIM     = 768,     // SigLIP-B/16 hidden dimension
    parameter NUM_LAYERS     = 12,      // Number of transformer layers
    parameter NUM_HEADS      = 12,      // Attention heads per layer
    parameter MLP_HIDDEN     = 3072,    // MLP intermediate dimension
    parameter ACT_WIDTH      = 8,       // Activation bit width
    parameter ACC_WIDTH      = 24,      // Accumulator bit width
    parameter PIXEL_WIDTH    = 8        // Bits per color channel
)(
    input  wire                             clk,
    input  wire                             rst_n,
    
    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire                             start,          // Start image processing
    input  wire                             image_valid,    // Image data is available
    output wire                             busy,           // Subsystem is processing
    output wire                             done,           // Processing complete
    
    // =========================================================================
    // Image Input Interface (RGB streaming, 24-bit per pixel)
    // =========================================================================
    input  wire                             pixel_valid,
    output wire                             pixel_ready,
    input  wire [3*PIXEL_WIDTH-1:0]         pixel_data,     // RGB 24-bit
    input  wire                             pixel_last,     // Last pixel in image
    
    // =========================================================================
    // Vision Embedding Output (768×8-bit CLS token)
    // =========================================================================
    output wire                             cls_valid,
    input  wire                             cls_ready,
    output wire [HIDDEN_DIM*ACT_WIDTH-1:0]  cls_embedding,
    
    // =========================================================================
    // Weight Memory Interface - Patch Embedding
    // =========================================================================
    // Conv2D 16×16 kernel weights (3 channels → 768 output channels)
    // Weight shape: [768, 3, 16, 16] = 768 × 768 ternary weights per output channel
    output wire                             patch_embed_rd_en,
    output wire [9:0]                       patch_embed_addr,       // 768 output channels
    input  wire [PATCH_SIZE*PATCH_SIZE*3*2-1:0] patch_embed_data,   // 16×16×3×2bits = 1536 bits
    
    // Position embeddings: 577 × 768 (8-bit each)
    output wire                             pos_embed_rd_en,
    output wire [9:0]                       pos_embed_addr,         // 577 positions
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  pos_embed_data,         // 768×8 = 6144 bits
    
    // CLS token embedding (learnable, 768×8-bit)
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  cls_token_embed,
    
    // =========================================================================
    // Weight Memory Interface - Transformer Layers (Shared Bus)
    // =========================================================================
    // Layer selection
    output wire [3:0]                       active_layer,           // 0-11
    
    // Attention Q/K/V/O projections
    output wire                             wq_rd_en,
    output wire [9:0]                       wq_addr,
    input  wire [HIDDEN_DIM*2-1:0]          wq_data,
    
    output wire                             wk_rd_en,
    output wire [9:0]                       wk_addr,
    input  wire [HIDDEN_DIM*2-1:0]          wk_data,
    
    output wire                             wv_rd_en,
    output wire [9:0]                       wv_addr,
    input  wire [HIDDEN_DIM*2-1:0]          wv_data,
    
    output wire                             wo_rd_en,
    output wire [9:0]                       wo_addr,
    input  wire [HIDDEN_DIM*2-1:0]          wo_data,
    
    // MLP weights
    output wire                             mlp_gate_rd_en,
    output wire [9:0]                       mlp_gate_addr,
    input  wire [MLP_HIDDEN*2-1:0]          mlp_gate_data,
    
    output wire                             mlp_up_rd_en,
    output wire [9:0]                       mlp_up_addr,
    input  wire [MLP_HIDDEN*2-1:0]          mlp_up_data,
    
    output wire                             mlp_down_rd_en,
    output wire [11:0]                      mlp_down_addr,
    input  wire [HIDDEN_DIM*2-1:0]          mlp_down_data,
    
    // LayerNorm parameters (per layer)
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  ln1_gamma,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  ln1_beta,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  ln2_gamma,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  ln2_beta,
    
    // Final LayerNorm parameters
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  final_ln_gamma,
    input  wire [HIDDEN_DIM*ACT_WIDTH-1:0]  final_ln_beta
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam GRID_SIZE = IMAGE_SIZE / PATCH_SIZE;  // 24×24 patch grid
    localparam SEQ_BITS  = $clog2(SEQ_LEN + 1);      // 10 bits for 577
    localparam LAYER_BITS = $clog2(NUM_LAYERS);      // 4 bits for 12 layers
    
    // Patch embedding: each 16×16×3 patch → 768-dim vector
    localparam PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * 3;  // 768 pixels per patch
    
    // =========================================================================
    // FSM States
    // =========================================================================
    localparam [3:0] ST_IDLE           = 4'd0;
    localparam [3:0] ST_LOAD_PIXEL     = 4'd1;
    localparam [3:0] ST_PATCH_EMBED    = 4'd2;
    localparam [3:0] ST_ADD_POS_EMBED  = 4'd3;
    localparam [3:0] ST_PREPEND_CLS    = 4'd4;
    localparam [3:0] ST_TRANSFORMER    = 4'd5;
    localparam [3:0] ST_NEXT_LAYER     = 4'd6;
    localparam [3:0] ST_FINAL_LN       = 4'd7;
    localparam [3:0] ST_EXTRACT_CLS    = 4'd8;
    localparam [3:0] ST_OUTPUT         = 4'd9;
    localparam [3:0] ST_DONE           = 4'd10;
    
    reg [3:0] state, next_state;
    
    // =========================================================================
    // Internal Registers
    // =========================================================================
    
    // Pixel accumulation for patch embedding
    reg [PATCH_PIXELS*PIXEL_WIDTH-1:0] patch_pixel_buffer;
    reg [9:0]                          pixel_count;        // Within patch (0-767)
    reg [SEQ_BITS-1:0]                 patch_idx;          // Current patch (0-575)
    reg [4:0]                          patch_row;          // Row within image (0-23)
    reg [4:0]                          patch_col;          // Column within image (0-23)
    
    // Patch embedding computation
    reg [HIDDEN_DIM*ACT_WIDTH-1:0]     embedded_patch;
    reg [9:0]                          embed_channel_idx;  // Output channel (0-767)
    
    // Sequence buffer (stores all 577 embedded patches)
    // In practice this would be external SRAM; modeled as interface here
    reg [SEQ_BITS-1:0]                 seq_write_ptr;
    reg [SEQ_BITS-1:0]                 seq_read_ptr;
    
    // Layer control
    reg [LAYER_BITS-1:0]               current_layer;
    reg                                layer_done;
    
    // Transformer block interface
    reg                                transformer_start;
    wire                               transformer_busy;
    wire                               transformer_done;
    
    // Inter-layer pipeline registers
    reg [HIDDEN_DIM*ACT_WIDTH-1:0]     layer_input_reg;
    reg [HIDDEN_DIM*ACT_WIDTH-1:0]     layer_output_reg;
    reg                                layer_input_valid;
    reg                                layer_input_last;
    
    // Final layer norm
    reg                                final_ln_start;
    wire                               final_ln_done;
    reg [HIDDEN_DIM*ACT_WIDTH-1:0]     final_ln_output;
    
    // CLS token output
    reg [HIDDEN_DIM*ACT_WIDTH-1:0]     cls_output_reg;
    reg                                cls_output_valid;
    
    // Control
    reg                                subsystem_busy;
    reg                                subsystem_done;

    // =========================================================================
    // Sequence Memory Interface (External SRAM for 577 × 768 × 8-bit patches)
    // =========================================================================
    // In a real implementation, this would connect to external SRAM
    // For hierarchical synthesis, we model the interface
    reg                                seq_mem_wr_en;
    reg  [SEQ_BITS-1:0]                seq_mem_wr_addr;
    reg  [HIDDEN_DIM*ACT_WIDTH-1:0]    seq_mem_wr_data;
    reg                                seq_mem_rd_en;
    reg  [SEQ_BITS-1:0]                seq_mem_rd_addr;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0]    seq_mem_rd_data;
    
    // =========================================================================
    // State Machine
    // =========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (start && image_valid)
                    next_state = ST_PREPEND_CLS;
            end
            
            ST_PREPEND_CLS: begin
                // Write CLS token as first sequence element
                next_state = ST_LOAD_PIXEL;
            end
            
            ST_LOAD_PIXEL: begin
                if (pixel_valid && pixel_ready) begin
                    if (pixel_count == PATCH_PIXELS - 1)
                        next_state = ST_PATCH_EMBED;
                end
            end
            
            ST_PATCH_EMBED: begin
                // Embed the patch (Conv2D projection)
                if (embed_channel_idx == HIDDEN_DIM - 1)
                    next_state = ST_ADD_POS_EMBED;
            end
            
            ST_ADD_POS_EMBED: begin
                // Add position embedding and store
                if (patch_idx == NUM_PATCHES - 1)
                    next_state = ST_TRANSFORMER;
                else
                    next_state = ST_LOAD_PIXEL;
            end
            
            ST_TRANSFORMER: begin
                // Process all patches through current transformer layer
                if (transformer_done)
                    next_state = ST_NEXT_LAYER;
            end
            
            ST_NEXT_LAYER: begin
                if (current_layer == NUM_LAYERS - 1)
                    next_state = ST_FINAL_LN;
                else
                    next_state = ST_TRANSFORMER;
            end
            
            ST_FINAL_LN: begin
                if (final_ln_done)
                    next_state = ST_EXTRACT_CLS;
            end
            
            ST_EXTRACT_CLS: begin
                // Extract CLS token (position 0)
                next_state = ST_OUTPUT;
            end
            
            ST_OUTPUT: begin
                if (cls_valid && cls_ready)
                    next_state = ST_DONE;
            end
            
            ST_DONE: begin
                next_state = ST_IDLE;
            end
            
            default: next_state = ST_IDLE;
        endcase
    end
    
    // =========================================================================
    // Main Control Logic
    // =========================================================================
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            subsystem_busy <= 1'b0;
            subsystem_done <= 1'b0;
            
            patch_pixel_buffer <= {(PATCH_PIXELS*PIXEL_WIDTH){1'b0}};
            pixel_count <= 10'd0;
            patch_idx <= {SEQ_BITS{1'b0}};
            patch_row <= 5'd0;
            patch_col <= 5'd0;
            
            embedded_patch <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            embed_channel_idx <= 10'd0;
            
            seq_write_ptr <= {SEQ_BITS{1'b0}};
            seq_read_ptr <= {SEQ_BITS{1'b0}};
            
            current_layer <= {LAYER_BITS{1'b0}};
            layer_done <= 1'b0;
            transformer_start <= 1'b0;
            
            layer_input_reg <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            layer_output_reg <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            layer_input_valid <= 1'b0;
            layer_input_last <= 1'b0;
            
            final_ln_start <= 1'b0;
            final_ln_output <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            
            cls_output_reg <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            cls_output_valid <= 1'b0;
            
            seq_mem_wr_en <= 1'b0;
            seq_mem_wr_addr <= {SEQ_BITS{1'b0}};
            seq_mem_wr_data <= {(HIDDEN_DIM*ACT_WIDTH){1'b0}};
            seq_mem_rd_en <= 1'b0;
            seq_mem_rd_addr <= {SEQ_BITS{1'b0}};
        end else begin
            state <= next_state;
            
            // Default signal clearing
            transformer_start <= 1'b0;
            final_ln_start <= 1'b0;
            seq_mem_wr_en <= 1'b0;
            seq_mem_rd_en <= 1'b0;
            
            case (state)
                ST_IDLE: begin
                    subsystem_busy <= 1'b0;
                    subsystem_done <= 1'b0;
                    cls_output_valid <= 1'b0;
                    
                    if (start && image_valid) begin
                        subsystem_busy <= 1'b1;
                        pixel_count <= 10'd0;
                        patch_idx <= {SEQ_BITS{1'b0}};
                        patch_row <= 5'd0;
                        patch_col <= 5'd0;
                        embed_channel_idx <= 10'd0;
                        seq_write_ptr <= {SEQ_BITS{1'b0}};
                        current_layer <= {LAYER_BITS{1'b0}};
                    end
                end
                
                ST_PREPEND_CLS: begin
                    // Write CLS token embedding as position 0
                    // Add position embedding for position 0
                    seq_mem_wr_en <= 1'b1;
                    seq_mem_wr_addr <= {SEQ_BITS{1'b0}};  // Position 0
                    // CLS token + position embedding[0]
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        seq_mem_wr_data[i*ACT_WIDTH +: ACT_WIDTH] <= 
                            saturating_add_8(
                                cls_token_embed[i*ACT_WIDTH +: ACT_WIDTH],
                                pos_embed_data[i*ACT_WIDTH +: ACT_WIDTH]
                            );
                    end
                    seq_write_ptr <= {{(SEQ_BITS-1){1'b0}}, 1'b1};  // Next write at 1
                    patch_idx <= {SEQ_BITS{1'b0}};
                end
                
                ST_LOAD_PIXEL: begin
                    if (pixel_valid) begin
                        // Accumulate pixels into patch buffer
                        patch_pixel_buffer[pixel_count*3*PIXEL_WIDTH +: 3*PIXEL_WIDTH] <= pixel_data;
                        pixel_count <= pixel_count + 1'b1;
                    end
                end
                
                ST_PATCH_EMBED: begin
                    // Compute patch embedding using Conv2D weights
                    // Each output channel is dot product of patch with kernel
                    embed_channel_idx <= embed_channel_idx + 1'b1;
                    
                    // Note: actual MAC computation would be done by dedicated hardware
                    // This models the control flow
                    if (embed_channel_idx == HIDDEN_DIM - 1) begin
                        embed_channel_idx <= 10'd0;
                    end
                end
                
                ST_ADD_POS_EMBED: begin
                    // Add position embedding and store in sequence memory
                    seq_mem_wr_en <= 1'b1;
                    seq_mem_wr_addr <= seq_write_ptr;
                    // embedded_patch + position_embedding[patch_idx + 1]
                    for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
                        seq_mem_wr_data[i*ACT_WIDTH +: ACT_WIDTH] <= 
                            saturating_add_8(
                                embedded_patch[i*ACT_WIDTH +: ACT_WIDTH],
                                pos_embed_data[i*ACT_WIDTH +: ACT_WIDTH]
                            );
                    end
                    
                    seq_write_ptr <= seq_write_ptr + 1'b1;
                    patch_idx <= patch_idx + 1'b1;
                    pixel_count <= 10'd0;
                    
                    // Update patch position
                    if (patch_col == GRID_SIZE - 1) begin
                        patch_col <= 5'd0;
                        patch_row <= patch_row + 1'b1;
                    end else begin
                        patch_col <= patch_col + 1'b1;
                    end
                end
                
                ST_TRANSFORMER: begin
                    transformer_start <= 1'b1;
                end
                
                ST_NEXT_LAYER: begin
                    current_layer <= current_layer + 1'b1;
                end
                
                ST_FINAL_LN: begin
                    final_ln_start <= 1'b1;
                end
                
                ST_EXTRACT_CLS: begin
                    // Read CLS token (position 0) after final layer norm
                    seq_mem_rd_en <= 1'b1;
                    seq_mem_rd_addr <= {SEQ_BITS{1'b0}};
                    cls_output_reg <= final_ln_output;
                end
                
                ST_OUTPUT: begin
                    cls_output_valid <= 1'b1;
                    if (cls_ready) begin
                        cls_output_valid <= 1'b0;
                    end
                end
                
                ST_DONE: begin
                    subsystem_done <= 1'b1;
                    subsystem_busy <= 1'b0;
                end
                
                default: begin
                    // Do nothing
                end
            endcase
        end
    end

    // =========================================================================
    // Saturating Add Function (8-bit signed)
    // =========================================================================
    function [ACT_WIDTH-1:0] saturating_add_8;
        input [ACT_WIDTH-1:0] a;
        input [ACT_WIDTH-1:0] b;
        reg signed [ACT_WIDTH:0] sum;
        begin
            sum = $signed({a[ACT_WIDTH-1], a}) + $signed({b[ACT_WIDTH-1], b});
            if (sum > 127)
                saturating_add_8 = 8'd127;
            else if (sum < -128)
                saturating_add_8 = 8'hFF & (-128);
            else
                saturating_add_8 = sum[ACT_WIDTH-1:0];
        end
    endfunction
    
    // =========================================================================
    // Patch Embedding Unit
    // =========================================================================
    // Conv2D: 16×16×3 input → 768 output channels
    // Using ternary weights for compute efficiency
    
    wire patch_embed_start = (state == ST_PATCH_EMBED);
    wire patch_embed_done;
    
    // Weight address generation
    assign patch_embed_rd_en = (state == ST_PATCH_EMBED);
    assign patch_embed_addr = embed_channel_idx;
    
    // Position embedding address: patch_idx + 1 (0 is for CLS)
    assign pos_embed_rd_en = (state == ST_ADD_POS_EMBED) || (state == ST_PREPEND_CLS);
    assign pos_embed_addr = (state == ST_PREPEND_CLS) ? 10'd0 : (patch_idx + 1'b1);
    
    // =========================================================================
    // Transformer Layer Pipeline Signals
    // =========================================================================
    
    // Patch streaming to transformer block
    wire                             tf_patch_valid_in;
    wire                             tf_patch_ready_in;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0]  tf_patch_data_in;
    wire                             tf_patch_last_in;
    
    // Patch streaming from transformer block
    wire                             tf_patch_valid_out;
    wire                             tf_patch_ready_out;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0]  tf_patch_data_out;
    wire                             tf_patch_last_out;
    
    assign tf_patch_valid_in = layer_input_valid;
    assign tf_patch_data_in = layer_input_reg;
    assign tf_patch_last_in = layer_input_last;
    assign tf_patch_ready_out = 1'b1;  // Always ready to receive output

    // =========================================================================
    // Weight Bus Multiplexing for 12 Transformer Layers
    // =========================================================================
    // Weights are loaded sequentially per layer from shared memory bus
    
    // Layer-specific weight addresses are offset by layer index
    assign active_layer = {{(4-LAYER_BITS){1'b0}}, current_layer};
    
    // =========================================================================
    // 12× Transformer Block Instances (3×4 Grid Layout)
    // =========================================================================
    // Blocks are connected in series for sequential layer processing
    // Each block processes all 577 patches before passing to next layer
    
    // Inter-block connection wires
    wire [NUM_LAYERS-1:0]            tf_start;
    wire [NUM_LAYERS-1:0]            tf_busy;
    wire [NUM_LAYERS-1:0]            tf_done;
    
    wire [NUM_LAYERS-1:0]            tf_in_valid;
    wire [NUM_LAYERS-1:0]            tf_in_ready;
    wire [NUM_LAYERS-1:0]            tf_in_last;
    
    wire [NUM_LAYERS-1:0]            tf_out_valid;
    wire [NUM_LAYERS-1:0]            tf_out_ready;
    wire [NUM_LAYERS-1:0]            tf_out_last;
    
    // Data buses between layers (768×8-bit each)
    wire [HIDDEN_DIM*ACT_WIDTH-1:0]  tf_data_out [0:NUM_LAYERS-1];
    
    // Weight interface wires per layer
    wire [NUM_LAYERS-1:0]            tf_wq_rd_en;
    wire [9:0]                       tf_wq_addr [0:NUM_LAYERS-1];
    wire [NUM_LAYERS-1:0]            tf_wk_rd_en;
    wire [9:0]                       tf_wk_addr [0:NUM_LAYERS-1];
    wire [NUM_LAYERS-1:0]            tf_wv_rd_en;
    wire [9:0]                       tf_wv_addr [0:NUM_LAYERS-1];
    wire [NUM_LAYERS-1:0]            tf_wo_rd_en;
    wire [9:0]                       tf_wo_addr [0:NUM_LAYERS-1];
    wire [NUM_LAYERS-1:0]            tf_mlp_gate_rd_en;
    wire [9:0]                       tf_mlp_gate_addr [0:NUM_LAYERS-1];
    wire [NUM_LAYERS-1:0]            tf_mlp_up_rd_en;
    wire [9:0]                       tf_mlp_up_addr [0:NUM_LAYERS-1];
    wire [NUM_LAYERS-1:0]            tf_mlp_down_rd_en;
    wire [11:0]                      tf_mlp_down_addr [0:NUM_LAYERS-1];
    
    // Generate 12 transformer blocks
    genvar layer;
    generate
        for (layer = 0; layer < NUM_LAYERS; layer = layer + 1) begin : gen_transformer_layers
            
            // Input selection: layer 0 gets from sequence memory, others from previous layer
            wire [HIDDEN_DIM*ACT_WIDTH-1:0] layer_data_in;
            wire                            layer_valid_in;
            wire                            layer_last_in;
            
            if (layer == 0) begin : first_layer
                assign layer_data_in = tf_patch_data_in;
                assign layer_valid_in = tf_patch_valid_in && (current_layer == layer);
                assign layer_last_in = tf_patch_last_in;
            end else begin : subsequent_layers
                assign layer_data_in = tf_data_out[layer-1];
                assign layer_valid_in = tf_out_valid[layer-1] && (current_layer == layer);
                assign layer_last_in = tf_out_last[layer-1];
            end
            
            // Start signal: only active layer receives start
            assign tf_start[layer] = transformer_start && (current_layer == layer);
            
            transformer_block_vision #(
                .HIDDEN_DIM(HIDDEN_DIM),
                .NUM_HEADS(NUM_HEADS),
                .HEAD_DIM(HIDDEN_DIM / NUM_HEADS),
                .MLP_HIDDEN(MLP_HIDDEN),
                .MAX_SEQ(SEQ_LEN),
                .ACT_WIDTH(ACT_WIDTH),
                .ACC_WIDTH(ACC_WIDTH)
            ) u_transformer_layer (
                .clk(clk),
                .rst_n(rst_n),
                
                // Control
                .layer_idx(layer[7:0]),
                .num_patches(SEQ_LEN[SEQ_BITS-1:0]),
                .start(tf_start[layer]),
                .busy(tf_busy[layer]),
                .done(tf_done[layer]),
                
                // Patch input
                .patch_valid_in(layer_valid_in),
                .patch_ready_in(tf_in_ready[layer]),
                .patch_data_in(layer_data_in),
                .patch_last_in(layer_last_in),
                
                // Patch output
                .patch_valid_out(tf_out_valid[layer]),
                .patch_ready_out(tf_out_ready[layer]),
                .patch_data_out(tf_data_out[layer]),
                .patch_last_out(tf_out_last[layer]),
                
                // Weight memory - Attention
                .wq_rd_en(tf_wq_rd_en[layer]),
                .wq_addr(tf_wq_addr[layer]),
                .wq_data(wq_data),
                
                .wk_rd_en(tf_wk_rd_en[layer]),
                .wk_addr(tf_wk_addr[layer]),
                .wk_data(wk_data),
                
                .wv_rd_en(tf_wv_rd_en[layer]),
                .wv_addr(tf_wv_addr[layer]),
                .wv_data(wv_data),
                
                .wo_rd_en(tf_wo_rd_en[layer]),
                .wo_addr(tf_wo_addr[layer]),
                .wo_data(wo_data),
                
                // Weight memory - MLP
                .mlp_gate_rd_en(tf_mlp_gate_rd_en[layer]),
                .mlp_gate_addr(tf_mlp_gate_addr[layer]),
                .mlp_gate_data(mlp_gate_data),
                
                .mlp_up_rd_en(tf_mlp_up_rd_en[layer]),
                .mlp_up_addr(tf_mlp_up_addr[layer]),
                .mlp_up_data(mlp_up_data),
                
                .mlp_down_rd_en(tf_mlp_down_rd_en[layer]),
                .mlp_down_addr(tf_mlp_down_addr[layer]),
                .mlp_down_data(mlp_down_data),
                
                // LayerNorm parameters
                .ln1_gamma(ln1_gamma),
                .ln1_beta(ln1_beta),
                .ln2_gamma(ln2_gamma),
                .ln2_beta(ln2_beta)
            );
            
            // Ready propagation
            if (layer == NUM_LAYERS - 1) begin : last_layer
                assign tf_out_ready[layer] = 1'b1;  // Final output always ready
            end else begin : middle_layers
                assign tf_out_ready[layer] = tf_in_ready[layer + 1];
            end
        end
    endgenerate

    // =========================================================================
    // Weight Bus Multiplexing
    // =========================================================================
    // Route weight requests from active layer to shared memory bus
    
    assign wq_rd_en = tf_wq_rd_en[current_layer];
    assign wq_addr = tf_wq_addr[current_layer];
    assign wk_rd_en = tf_wk_rd_en[current_layer];
    assign wk_addr = tf_wk_addr[current_layer];
    assign wv_rd_en = tf_wv_rd_en[current_layer];
    assign wv_addr = tf_wv_addr[current_layer];
    assign wo_rd_en = tf_wo_rd_en[current_layer];
    assign wo_addr = tf_wo_addr[current_layer];
    assign mlp_gate_rd_en = tf_mlp_gate_rd_en[current_layer];
    assign mlp_gate_addr = tf_mlp_gate_addr[current_layer];
    assign mlp_up_rd_en = tf_mlp_up_rd_en[current_layer];
    assign mlp_up_addr = tf_mlp_up_addr[current_layer];
    assign mlp_down_rd_en = tf_mlp_down_rd_en[current_layer];
    assign mlp_down_addr = tf_mlp_down_addr[current_layer];
    
    // Aggregate busy/done from current layer
    assign transformer_busy = tf_busy[current_layer];
    assign transformer_done = tf_done[current_layer];
    
    // =========================================================================
    // Final Layer Normalization
    // =========================================================================
    // Applied after all 12 transformer layers, before CLS extraction
    
    wire                             final_ln_valid_in;
    wire                             final_ln_ready_in;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0]  final_ln_data_in;
    wire                             final_ln_valid_out;
    wire                             final_ln_ready_out;
    wire [HIDDEN_DIM*ACT_WIDTH-1:0]  final_ln_data_out;
    
    assign final_ln_valid_in = (state == ST_FINAL_LN);
    assign final_ln_data_in = tf_data_out[NUM_LAYERS-1];  // Output of last transformer
    assign final_ln_ready_out = 1'b1;
    assign final_ln_done = final_ln_valid_out;
    
    layer_norm_block #(
        .DIM(HIDDEN_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(32)
    ) u_final_layer_norm (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(final_ln_data_in),
        .valid_in(final_ln_valid_in),
        .ready_in(final_ln_ready_in),
        .gamma(final_ln_gamma),
        .beta(final_ln_beta),
        .y_out(final_ln_data_out),
        .valid_out(final_ln_valid_out),
        .ready_out(final_ln_ready_out)
    );
    
    // Capture final layer norm output
    always @(posedge clk) begin
        if (final_ln_valid_out)
            final_ln_output <= final_ln_data_out;
    end

    // =========================================================================
    // Sequence Memory (External SRAM Model)
    // =========================================================================
    // 577 × 768 × 8-bit = ~3.5 Mbit sequence buffer
    // In actual implementation, this would be external SRAM macro
    // For synthesis, we model the interface and use behavioral memory
    
    (* ram_style = "block" *)
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] seq_memory [0:SEQ_LEN-1];
    
    reg [HIDDEN_DIM*ACT_WIDTH-1:0] seq_mem_rd_data_reg;
    
    always @(posedge clk) begin
        if (seq_mem_wr_en) begin
            seq_memory[seq_mem_wr_addr] <= seq_mem_wr_data;
        end
        if (seq_mem_rd_en) begin
            seq_mem_rd_data_reg <= seq_memory[seq_mem_rd_addr];
        end
    end
    
    assign seq_mem_rd_data = seq_mem_rd_data_reg;
    
    // =========================================================================
    // Output Assignments
    // =========================================================================
    assign busy = subsystem_busy;
    assign done = subsystem_done;
    
    assign pixel_ready = (state == ST_LOAD_PIXEL);
    
    assign cls_valid = cls_output_valid;
    assign cls_embedding = cls_output_reg;

endmodule

`default_nettype wire
