// =============================================================================
// SiLens Vision Encoder Top (SigLIP-B/16 Architecture)
// =============================================================================
// Complete vision encoder pipeline:
//   1. Patch extraction (16×16 patches from 384×384 image)
//   2. Patch embedding (linear projection + position embedding)
//   3. Transformer blocks (12 layers, 768 dim, 12 heads)
//   4. Final layer norm
//
// Total parameters: 93M (ternary hardwired)
// Input: 384×384 RGB image (streaming pixels)
// Output: 576 tokens × 768 dimensions
//
// Target: SkyWater SKY130, ~250mm² area
//
// License: Apache 2.0
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module vision_encoder_top #(
    parameter IMG_SIZE     = 384,
    parameter PATCH_SIZE   = 16,
    parameter IN_CHANNELS  = 3,
    parameter EMBED_DIM    = 768,
    parameter NUM_LAYERS   = 12,
    parameter NUM_HEADS    = 12,
    parameter MLP_DIM      = 3072,
    parameter ACT_WIDTH    = 8,
    parameter ACC_WIDTH    = 32,
    parameter PARALLEL     = 64,
    parameter MEM_ADDR_WIDTH = 28,
    parameter MEM_DATA_WIDTH = 512
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Control
    input  wire                         start,
    output reg                          done,
    
    // Pixel input (streaming RGB)
    input  wire [IN_CHANNELS*ACT_WIDTH-1:0] pixel_in,
    input  wire                         pixel_valid,
    output wire                         pixel_ready,
    
    // Token output
    output wire [EMBED_DIM*ACT_WIDTH-1:0] token_out,
    output wire [9:0]                   token_idx,      // 10 bits for 576 patches
    output wire                         token_valid,
    input  wire                         token_ready,
    
    // Memory interface
    output wire [MEM_ADDR_WIDTH-1:0]    mem_addr,
    output wire [MEM_DATA_WIDTH-1:0]    mem_wdata,
    input  wire [MEM_DATA_WIDTH-1:0]    mem_rdata,
    output wire                         mem_rd,
    output wire                         mem_wr,
    input  wire                         mem_ready
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    
    localparam NUM_PATCHES   = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE);  // 576
    localparam PATCH_PIXELS  = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS;              // 768
    localparam HEAD_DIM      = EMBED_DIM / NUM_HEADS;                              // 64
    localparam PATCH_IDX_WIDTH = $clog2(NUM_PATCHES);                              // 10 bits
    
    // =========================================================================
    // State Machine
    // =========================================================================
    
    localparam ST_IDLE        = 3'd0;
    localparam ST_PATCH_EMBED = 3'd1;
    localparam ST_TRANSFORMER = 3'd2;
    localparam ST_OUTPUT      = 3'd3;
    localparam ST_DONE        = 3'd4;
    
    reg [2:0] state;
    reg [PATCH_IDX_WIDTH-1:0] layer_cnt;
    reg [PATCH_IDX_WIDTH-1:0] patch_cnt;
    reg [PATCH_IDX_WIDTH-1:0] out_cnt;
    
    // =========================================================================
    // Patch Embedding
    // =========================================================================
    
    // Patch buffer - collect PATCH_SIZE×PATCH_SIZE pixels
    reg [PATCH_PIXELS*ACT_WIDTH-1:0] patch_buffer;
    reg [$clog2(PATCH_PIXELS)-1:0] pixel_cnt;
    reg patch_buffer_valid;
    
    // Pixel ready when collecting patches
    assign pixel_ready = (state == ST_PATCH_EMBED) && (pixel_cnt < PATCH_PIXELS);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            patch_buffer <= 0;
            pixel_cnt <= 0;
            patch_buffer_valid <= 0;
        end else begin
            patch_buffer_valid <= 0;
            
            if (state == ST_PATCH_EMBED) begin
                if (pixel_valid && pixel_ready) begin
                    // Shift in new pixel
                    patch_buffer <= {patch_buffer[PATCH_PIXELS*ACT_WIDTH-IN_CHANNELS*ACT_WIDTH-1:0], pixel_in};
                    pixel_cnt <= pixel_cnt + 1;
                    
                    if (pixel_cnt == PATCH_PIXELS - 1) begin
                        patch_buffer_valid <= 1;
                        pixel_cnt <= 0;
                    end
                end
            end else begin
                pixel_cnt <= 0;
            end
        end
    end
    
    // =========================================================================
    // Patch Projection (Linear: 768 -> 768)
    // =========================================================================
    
    // Hardwired weights for patch projection
    // In actual implementation: ROM or metal routing
    wire [PATCH_PIXELS*EMBED_DIM*2-1:0] patch_proj_weights;
    assign patch_proj_weights = {(PATCH_PIXELS*EMBED_DIM){2'b01}};  // Placeholder
    
    wire [EMBED_DIM*ACT_WIDTH-1:0] embed_out;
    wire embed_valid;
    reg embed_ready;
    
    // Instantiate ternary matrix multiply for patch embedding
    ternary_matmul #(
        .M(1),              // Batch size (1 patch at a time)
        .K(PATCH_PIXELS),   // Input dimension
        .N(EMBED_DIM),      // Output dimension
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) u_patch_embed (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(patch_buffer),
        .x_valid(patch_buffer_valid),
        .x_ready(),
        .weights(patch_proj_weights),
        .y_out(embed_out),
        .y_valid(embed_valid),
        .y_ready(embed_ready)
    );
    
    // =========================================================================
    // Position Embedding Addition
    // =========================================================================
    
    // Position embeddings (576 × 768 × 8-bit = 3.5MB hardwired)
    wire [EMBED_DIM*ACT_WIDTH-1:0] pos_embed_data;
    
    // Position embedding ROM (placeholder)
    position_embedding_rom #(
        .NUM_POS(NUM_PATCHES),
        .EMBED_DIM(EMBED_DIM),
        .ACT_WIDTH(ACT_WIDTH)
    ) u_pos_embed (
        .clk(clk),
        .pos_idx(patch_cnt),
        .embed_out(pos_embed_data)
    );
    
    // Add position embedding to patch embedding
    wire [EMBED_DIM*ACT_WIDTH-1:0] embed_with_pos;
    
    generate
        genvar i;
        for (i = 0; i < EMBED_DIM; i = i + 1) begin : pos_add
            wire signed [ACT_WIDTH:0] sum;
            assign sum = $signed(embed_out[i*ACT_WIDTH +: ACT_WIDTH]) + 
                        $signed(pos_embed_data[i*ACT_WIDTH +: ACT_WIDTH]);
            // Saturation
            assign embed_with_pos[i*ACT_WIDTH +: ACT_WIDTH] = 
                (sum > $signed({1'b0, {(ACT_WIDTH-1){1'b1}}})) ? {1'b0, {(ACT_WIDTH-1){1'b1}}} :
                (sum < $signed({1'b1, {(ACT_WIDTH-1){1'b0}}})) ? {1'b1, {(ACT_WIDTH-1){1'b0}}} :
                sum[ACT_WIDTH-1:0];
        end
    endgenerate
    
    // =========================================================================
    // Embedding Storage (SRAM)
    // =========================================================================
    
    // Buffer to store all embedded patches before transformer
    // 576 × 768 × 8-bit = 3.5MB
    reg [EMBED_DIM*ACT_WIDTH-1:0] embed_buffer [0:NUM_PATCHES-1];
    
    always @(posedge clk) begin
        if (embed_valid && embed_ready) begin
            embed_buffer[patch_cnt] <= embed_with_pos;
        end
    end
    
    // =========================================================================
    // Transformer Blocks (12 layers)
    // =========================================================================
    
    // Token buffer for transformer processing
    reg [EMBED_DIM*ACT_WIDTH-1:0] xform_in;
    reg [PATCH_IDX_WIDTH-1:0] xform_in_idx;
    reg xform_in_valid;
    wire xform_in_ready;
    
    wire [EMBED_DIM*ACT_WIDTH-1:0] xform_out;
    wire [PATCH_IDX_WIDTH-1:0] xform_out_idx;
    wire xform_out_valid;
    reg xform_out_ready;
    wire xform_layer_done;
    
    // Instantiate transformer block (processes all tokens through one layer)
    transformer_block #(
        .SEQ_LEN(NUM_PATCHES),
        .EMBED_DIM(EMBED_DIM),
        .NUM_HEADS(NUM_HEADS),
        .MLP_DIM(MLP_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) u_transformer (
        .clk(clk),
        .rst_n(rst_n),
        
        .layer_idx(layer_cnt),
        
        .x_in(xform_in),
        .x_idx(xform_in_idx),
        .x_valid(xform_in_valid),
        .x_ready(xform_in_ready),
        
        .y_out(xform_out),
        .y_idx(xform_out_idx),
        .y_valid(xform_out_valid),
        .y_ready(xform_out_ready),
        
        .layer_done(xform_layer_done)
    );
    
    // =========================================================================
    // Final Layer Norm
    // =========================================================================
    
    wire [EMBED_DIM*ACT_WIDTH-1:0] final_norm_out;
    wire final_norm_valid;
    wire final_norm_ready;
    
    // Default gamma = 1.0, beta = 0.0
    wire [EMBED_DIM*ACT_WIDTH-1:0] ln_gamma;
    wire [EMBED_DIM*ACT_WIDTH-1:0] ln_beta;
    assign ln_gamma = {EMBED_DIM{8'd16}};  // 1.0 in Q4.4
    assign ln_beta = {EMBED_DIM*ACT_WIDTH{1'b0}};
    
    layer_norm #(
        .DIM(EMBED_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .PARALLEL(PARALLEL)
    ) u_final_norm (
        .clk(clk),
        .rst_n(rst_n),
        .x_in(xform_out),
        .valid_in(xform_out_valid & (layer_cnt == NUM_LAYERS - 1)),
        .ready_in(final_norm_ready),
        .gamma(ln_gamma),
        .beta(ln_beta),
        .y_out(final_norm_out),
        .valid_out(final_norm_valid),
        .ready_out(1'b1)
    );
    
    // =========================================================================
    // Main Control Logic
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            layer_cnt <= 0;
            patch_cnt <= 0;
            out_cnt <= 0;
            done <= 0;
            embed_ready <= 0;
            xform_in_valid <= 0;
            xform_out_ready <= 0;
        end else begin
            done <= 0;
            embed_ready <= 0;
            xform_in_valid <= 0;
            
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        state <= ST_PATCH_EMBED;
                        patch_cnt <= 0;
                        layer_cnt <= 0;
                    end
                end
                
                ST_PATCH_EMBED: begin
                    embed_ready <= 1;
                    
                    if (embed_valid && embed_ready) begin
                        patch_cnt <= patch_cnt + 1;
                        if (patch_cnt == NUM_PATCHES - 1) begin
                            state <= ST_TRANSFORMER;
                            patch_cnt <= 0;
                        end
                    end
                end
                
                ST_TRANSFORMER: begin
                    // Feed tokens to transformer
                    if (xform_in_ready && patch_cnt < NUM_PATCHES) begin
                        xform_in <= embed_buffer[patch_cnt];
                        xform_in_idx <= patch_cnt;
                        xform_in_valid <= 1;
                        patch_cnt <= patch_cnt + 1;
                    end
                    
                    // Collect transformer output and write back
                    xform_out_ready <= 1;
                    if (xform_out_valid && xform_out_ready) begin
                        embed_buffer[xform_out_idx] <= xform_out;
                    end
                    
                    // Check for layer completion
                    if (xform_layer_done) begin
                        layer_cnt <= layer_cnt + 1;
                        patch_cnt <= 0;
                        
                        if (layer_cnt == NUM_LAYERS - 1) begin
                            state <= ST_OUTPUT;
                            out_cnt <= 0;
                        end
                    end
                end
                
                ST_OUTPUT: begin
                    // Stream out final tokens
                    if (token_ready && out_cnt < NUM_PATCHES) begin
                        out_cnt <= out_cnt + 1;
                        if (out_cnt == NUM_PATCHES - 1) begin
                            state <= ST_DONE;
                        end
                    end
                end
                
                ST_DONE: begin
                    done <= 1;
                    if (!start) begin
                        state <= ST_IDLE;
                    end
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Output Assignment
    // =========================================================================
    
    assign token_out = embed_buffer[out_cnt];
    assign token_idx = out_cnt;
    assign token_valid = (state == ST_OUTPUT);
    
    // Memory interface (not used in this simplified version)
    assign mem_addr = 0;
    assign mem_wdata = 0;
    assign mem_rd = 0;
    assign mem_wr = 0;

endmodule

// =============================================================================
// Position Embedding ROM (Placeholder)
// =============================================================================

module position_embedding_rom #(
    parameter NUM_POS = 576,
    parameter EMBED_DIM = 768,
    parameter ACT_WIDTH = 8,
    parameter POS_IDX_WIDTH = 10
)(
    input  wire                         clk,
    input  wire [POS_IDX_WIDTH-1:0]     pos_idx,
    output wire [EMBED_DIM*ACT_WIDTH-1:0] embed_out
);
    // In actual implementation, this would be ROM or hardwired
    // For now, output zeros (position embeddings would be trained values)
    assign embed_out = {(EMBED_DIM*ACT_WIDTH){1'b0}};
endmodule

// =============================================================================
// Ternary Matrix Multiply (Placeholder)
// =============================================================================

module ternary_matmul #(
    parameter M = 1,
    parameter K = 768,
    parameter N = 768,
    parameter ACT_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter PARALLEL = 64
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire [M*K*ACT_WIDTH-1:0]     x_in,
    input  wire                         x_valid,
    output wire                         x_ready,
    input  wire [K*N*2-1:0]             weights,
    output reg  [M*N*ACT_WIDTH-1:0]     y_out,
    output reg                          y_valid,
    input  wire                         y_ready
);
    // Simplified placeholder - actual impl uses ternary_mac
    assign x_ready = y_ready;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out <= 0;
            y_valid <= 0;
        end else begin
            y_valid <= x_valid;
            if (x_valid) begin
                y_out <= x_in[M*N*ACT_WIDTH-1:0];  // Placeholder
            end
        end
    end
endmodule

// =============================================================================
// Transformer Block (Placeholder)
// =============================================================================

module transformer_block #(
    parameter SEQ_LEN = 576,
    parameter EMBED_DIM = 768,
    parameter NUM_HEADS = 12,
    parameter MLP_DIM = 3072,
    parameter ACT_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter PARALLEL = 64,
    parameter SEQ_IDX_WIDTH = 10
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    input  wire [3:0]                   layer_idx,
    
    input  wire [EMBED_DIM*ACT_WIDTH-1:0] x_in,
    input  wire [SEQ_IDX_WIDTH-1:0]     x_idx,
    input  wire                         x_valid,
    output wire                         x_ready,
    
    output wire [EMBED_DIM*ACT_WIDTH-1:0] y_out,
    output wire [SEQ_IDX_WIDTH-1:0]     y_idx,
    output wire                         y_valid,
    input  wire                         y_ready,
    
    output wire                         layer_done
);
    // Placeholder - passthrough
    assign x_ready = y_ready;
    assign y_out = x_in;
    assign y_idx = x_idx;
    assign y_valid = x_valid;
    
    // Layer done after processing all tokens
    reg [SEQ_IDX_WIDTH-1:0] cnt;
    reg done_r;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= 0;
            done_r <= 0;
        end else begin
            done_r <= 0;
            if (x_valid && x_ready) begin
                cnt <= cnt + 1;
                if (cnt == SEQ_LEN - 1) begin
                    cnt <= 0;
                    done_r <= 1;
                end
            end
        end
    end
    
    assign layer_done = done_r;
endmodule

`default_nettype wire
