// =============================================================================
// SiLens - Patch Embedding Module
// =============================================================================
// Extracts patches from input image and projects to embedding dimension.
//
// Input: 384x384 RGB image (8-bit per channel)
// Process:
//   1. Extract 24x24 grid of 16x16 patches (576 patches total)
//   2. Linear projection: 16x16x3 = 768 values -> 768-dim embedding
//   3. Add learnable positional embeddings
//
// Output: 576 tokens of 768 dimensions each
//
// Architecture parameters:
//   - Image size: 384x384
//   - Patch size: 16x16
//   - Grid: 24x24 = 576 patches
//   - Embedding dim: 768
//
// License: Apache 2.0
// =============================================================================

module patch_embed #(
    parameter IMG_SIZE    = 384,                    // Input image size
    parameter PATCH_SIZE  = 16,                     // Patch size
    parameter IN_CHANNELS = 3,                      // RGB channels
    parameter EMBED_DIM   = 768,                    // Output embedding dimension
    parameter ACT_WIDTH   = 8,                      // Activation bit width
    parameter ACC_WIDTH   = 32,                     // Accumulator bit width
    parameter PARALLEL    = 16                      // Parallel MAC operations
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Image input interface (streaming pixels row by row)
    input  wire [IN_CHANNELS*ACT_WIDTH-1:0] pixel_in,       // RGB pixel
    input  wire                         pixel_valid,
    output wire                         pixel_ready,
    
    // Start of frame signal
    input  wire                         frame_start,
    
    // Ternary projection weights (hardwired, but exposed for flexibility)
    // Projection: (PATCH_SIZE * PATCH_SIZE * IN_CHANNELS) -> EMBED_DIM
    // 768 -> 768, weights are 768 * 768 * 2 bits = 1.125 Mbit
    input  wire [EMBED_DIM*PATCH_SIZE*PATCH_SIZE*IN_CHANNELS*2-1:0] proj_weights,
    
    // Positional embeddings (fixed, loaded at init)
    input  wire [NUM_PATCHES*EMBED_DIM*ACT_WIDTH-1:0] pos_embed,
    
    // Output interface (one token at a time)
    output reg  [EMBED_DIM*ACT_WIDTH-1:0] token_out,        // Embedded token
    output reg  [$clog2(NUM_PATCHES)-1:0] token_idx,        // Token index (0-575)
    output reg                          token_valid,
    input  wire                         token_ready
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    
    localparam GRID_SIZE = IMG_SIZE / PATCH_SIZE;   // 24
    localparam NUM_PATCHES = GRID_SIZE * GRID_SIZE; // 576
    localparam PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS; // 768
    
    localparam NUM_PROJ_ITERS = (PATCH_PIXELS + PARALLEL - 1) / PARALLEL;
    
    // Weight encoding
    localparam W_ZERO = 2'b00;
    localparam W_POS  = 2'b01;
    localparam W_NEG  = 2'b10;
    
    // =========================================================================
    // FSM states
    // =========================================================================
    
    localparam STATE_IDLE      = 3'd0;
    localparam STATE_LOAD      = 3'd1;  // Load patch pixels
    localparam STATE_PROJECT   = 3'd2;  // Linear projection
    localparam STATE_POS_ADD   = 3'd3;  // Add positional embedding
    localparam STATE_OUTPUT    = 3'd4;  // Output token
    
    reg [2:0] state;
    
    // =========================================================================
    // Pixel buffer for current patch
    // =========================================================================
    
    reg [ACT_WIDTH-1:0] patch_buffer [0:PATCH_PIXELS-1];
    
    // Pixel loading counters
    reg [$clog2(IMG_SIZE)-1:0] pixel_x, pixel_y;
    reg [$clog2(GRID_SIZE)-1:0] patch_x, patch_y;
    reg [$clog2(PATCH_SIZE)-1:0] patch_px, patch_py;
    
    // Projection counters
    reg [$clog2(EMBED_DIM)-1:0] embed_idx;
    reg [$clog2(NUM_PROJ_ITERS)-1:0] proj_iter;
    
    // Line buffer for storing partial rows
    // Need to buffer PATCH_SIZE rows to extract patches
    reg [IN_CHANNELS*ACT_WIDTH-1:0] line_buffer [0:IMG_SIZE*PATCH_SIZE-1];
    reg [$clog2(PATCH_SIZE)-1:0] line_buf_row;
    reg [$clog2(IMG_SIZE)-1:0] line_buf_col;
    reg line_buf_ready;
    
    // Current patch being processed
    reg [$clog2(NUM_PATCHES)-1:0] current_patch;
    
    // =========================================================================
    // Projection result buffer
    // =========================================================================
    
    reg signed [ACC_WIDTH-1:0] embed_accum [0:EMBED_DIM-1];
    
    // =========================================================================
    // Ready signal
    // =========================================================================
    
    assign pixel_ready = (state == STATE_LOAD) || (state == STATE_IDLE && frame_start);
    
    // =========================================================================
    // Pixel loading logic
    // =========================================================================
    
    integer load_i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            pixel_x <= 0;
            pixel_y <= 0;
            line_buf_row <= 0;
            line_buf_col <= 0;
            line_buf_ready <= 1'b0;
        end else if (state == STATE_IDLE && frame_start) begin
            pixel_x <= 0;
            pixel_y <= 0;
            line_buf_row <= 0;
            line_buf_col <= 0;
            line_buf_ready <= 1'b0;
        end else if (state == STATE_LOAD && pixel_valid && pixel_ready) begin
            // Store pixel in line buffer
            line_buffer[line_buf_row * IMG_SIZE + line_buf_col] <= pixel_in;
            
            // Update column
            if (line_buf_col == IMG_SIZE - 1) begin
                line_buf_col <= 0;
                // Update row within current set of PATCH_SIZE rows
                if (line_buf_row == PATCH_SIZE - 1) begin
                    line_buf_row <= 0;
                    line_buf_ready <= 1'b1;
                end else begin
                    line_buf_row <= line_buf_row + 1;
                end
            end else begin
                line_buf_col <= line_buf_col + 1;
            end
            
            // Track absolute pixel position
            if (pixel_x == IMG_SIZE - 1) begin
                pixel_x <= 0;
                pixel_y <= pixel_y + 1;
            end else begin
                pixel_x <= pixel_x + 1;
            end
        end
    end
    
    // =========================================================================
    // Extract patch from line buffer
    // =========================================================================
    
    // Extract pixels for current patch when line buffer is ready
    integer px, py, ch;
    
    always @(posedge clk) begin
        if (line_buf_ready && state == STATE_LOAD) begin
            // Extract current patch
            for (py = 0; py < PATCH_SIZE; py = py + 1) begin
                for (px = 0; px < PATCH_SIZE; px = px + 1) begin
                    for (ch = 0; ch < IN_CHANNELS; ch = ch + 1) begin
                        patch_buffer[(py * PATCH_SIZE + px) * IN_CHANNELS + ch] <=
                            line_buffer[py * IMG_SIZE + patch_x * PATCH_SIZE + px][ch*ACT_WIDTH +: ACT_WIDTH];
                    end
                end
            end
        end
    end
    
    // =========================================================================
    // Ternary MAC for projection
    // =========================================================================
    
    // Parallel MAC computation for projection
    wire signed [ACC_WIDTH-1:0] mac_partial [0:EMBED_DIM-1];
    
    genvar g_e, g_p;
    generate
        for (g_e = 0; g_e < EMBED_DIM; g_e = g_e + 1) begin : gen_embed
            // Accumulate contribution from PARALLEL input elements per cycle
            reg signed [ACC_WIDTH-1:0] partial_sum;
            integer p_idx;
            
            always @(*) begin
                partial_sum = 0;
                for (p_idx = 0; p_idx < PARALLEL; p_idx = p_idx + 1) begin
                    if (proj_iter * PARALLEL + p_idx < PATCH_PIXELS) begin
                        // Get weight for this (input, output) pair
                        // Weight index: g_e * PATCH_PIXELS + (proj_iter * PARALLEL + p_idx)
                        // Each weight is 2 bits
                        case (proj_weights[(g_e * PATCH_PIXELS + proj_iter * PARALLEL + p_idx) * 2 +: 2])
                            W_POS: partial_sum = partial_sum + 
                                   $signed({1'b0, patch_buffer[proj_iter * PARALLEL + p_idx]});
                            W_NEG: partial_sum = partial_sum - 
                                   $signed({1'b0, patch_buffer[proj_iter * PARALLEL + p_idx]});
                            default: ; // W_ZERO: no contribution
                        endcase
                    end
                end
            end
            
            assign mac_partial[g_e] = partial_sum;
        end
    endgenerate
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    integer acc_i, out_i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            current_patch <= 0;
            patch_x <= 0;
            patch_y <= 0;
            embed_idx <= 0;
            proj_iter <= 0;
            token_valid <= 1'b0;
            token_idx <= 0;
            
            for (acc_i = 0; acc_i < EMBED_DIM; acc_i = acc_i + 1) begin
                embed_accum[acc_i] <= 0;
            end
        end else begin
            case (state)
                STATE_IDLE: begin
                    token_valid <= 1'b0;
                    if (frame_start) begin
                        state <= STATE_LOAD;
                        current_patch <= 0;
                        patch_x <= 0;
                        patch_y <= 0;
                    end
                end
                
                STATE_LOAD: begin
                    // Wait for line buffer to be ready (PATCH_SIZE rows loaded)
                    if (line_buf_ready) begin
                        state <= STATE_PROJECT;
                        proj_iter <= 0;
                        
                        // Clear accumulators
                        for (acc_i = 0; acc_i < EMBED_DIM; acc_i = acc_i + 1) begin
                            embed_accum[acc_i] <= 0;
                        end
                    end
                end
                
                STATE_PROJECT: begin
                    // Accumulate MAC results
                    for (acc_i = 0; acc_i < EMBED_DIM; acc_i = acc_i + 1) begin
                        embed_accum[acc_i] <= embed_accum[acc_i] + mac_partial[acc_i];
                    end
                    
                    if (proj_iter >= NUM_PROJ_ITERS - 1) begin
                        state <= STATE_POS_ADD;
                    end else begin
                        proj_iter <= proj_iter + 1;
                    end
                end
                
                STATE_POS_ADD: begin
                    // Add positional embedding and saturate to ACT_WIDTH
                    for (out_i = 0; out_i < EMBED_DIM; out_i = out_i + 1) begin
                        // Get positional embedding value
                        // Saturate and store result
                        token_out[out_i*ACT_WIDTH +: ACT_WIDTH] <= 
                            saturate_to_width(
                                embed_accum[out_i] + 
                                $signed(pos_embed[(current_patch * EMBED_DIM + out_i) * ACT_WIDTH +: ACT_WIDTH])
                            );
                    end
                    token_idx <= current_patch;
                    state <= STATE_OUTPUT;
                end
                
                STATE_OUTPUT: begin
                    token_valid <= 1'b1;
                    
                    if (token_ready) begin
                        token_valid <= 1'b0;
                        
                        // Move to next patch
                        if (current_patch >= NUM_PATCHES - 1) begin
                            state <= STATE_IDLE;
                        end else begin
                            current_patch <= current_patch + 1;
                            
                            // Update patch coordinates
                            if (patch_x >= GRID_SIZE - 1) begin
                                patch_x <= 0;
                                patch_y <= patch_y + 1;
                            end else begin
                                patch_x <= patch_x + 1;
                            end
                            
                            // If we need more rows, go back to LOAD
                            // Otherwise we can process next patch from same line buffer
                            if (patch_x >= GRID_SIZE - 1) begin
                                state <= STATE_LOAD;
                            end else begin
                                state <= STATE_PROJECT;
                                proj_iter <= 0;
                                for (acc_i = 0; acc_i < EMBED_DIM; acc_i = acc_i + 1) begin
                                    embed_accum[acc_i] <= 0;
                                end
                            end
                        end
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end
    
    // =========================================================================
    // Saturation function
    // =========================================================================
    
    function [ACT_WIDTH-1:0] saturate_to_width;
        input signed [ACC_WIDTH-1:0] val;
        reg signed [ACT_WIDTH-1:0] max_val;
        reg signed [ACT_WIDTH-1:0] min_val;
        begin
            max_val = {1'b0, {(ACT_WIDTH-1){1'b1}}};  // 127 for 8-bit
            min_val = {1'b1, {(ACT_WIDTH-1){1'b0}}};  // -128 for 8-bit
            
            if (val > max_val)
                saturate_to_width = max_val;
            else if (val < min_val)
                saturate_to_width = min_val;
            else
                saturate_to_width = val[ACT_WIDTH-1:0];
        end
    endfunction

endmodule

// =============================================================================
// Testbench
// =============================================================================
`ifdef SIMULATION

module patch_embed_tb;
    parameter IMG_SIZE = 32;      // Smaller for testing
    parameter PATCH_SIZE = 8;
    parameter IN_CHANNELS = 3;
    parameter EMBED_DIM = 64;
    parameter ACT_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter PARALLEL = 8;
    
    localparam GRID_SIZE = IMG_SIZE / PATCH_SIZE;
    localparam NUM_PATCHES = GRID_SIZE * GRID_SIZE;
    localparam PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS;
    
    reg clk, rst_n;
    reg [IN_CHANNELS*ACT_WIDTH-1:0] pixel_in;
    reg pixel_valid;
    wire pixel_ready;
    reg frame_start;
    reg [EMBED_DIM*PATCH_PIXELS*2-1:0] proj_weights;
    reg [NUM_PATCHES*EMBED_DIM*ACT_WIDTH-1:0] pos_embed;
    wire [EMBED_DIM*ACT_WIDTH-1:0] token_out;
    wire [$clog2(NUM_PATCHES)-1:0] token_idx;
    wire token_valid;
    reg token_ready;
    
    patch_embed #(
        .IMG_SIZE(IMG_SIZE),
        .PATCH_SIZE(PATCH_SIZE),
        .IN_CHANNELS(IN_CHANNELS),
        .EMBED_DIM(EMBED_DIM),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PARALLEL(PARALLEL)
    ) dut (.*);
    
    always #5 clk = ~clk;
    
    integer i, j, tokens_received;
    
    initial begin
        $display("Patch Embedding Testbench");
        $display("=========================");
        $display("Image: %0dx%0d, Patch: %0dx%0d, Grid: %0dx%0d = %0d patches",
                 IMG_SIZE, IMG_SIZE, PATCH_SIZE, PATCH_SIZE, GRID_SIZE, GRID_SIZE, NUM_PATCHES);
        
        clk = 0;
        rst_n = 0;
        pixel_in = 0;
        pixel_valid = 0;
        frame_start = 0;
        token_ready = 1;
        
        // Initialize weights to all +1 (simple test)
        proj_weights = {(EMBED_DIM*PATCH_PIXELS){2'b01}};
        
        // Initialize positional embeddings to 0
        pos_embed = 0;
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Start frame
        frame_start = 1;
        @(posedge clk);
        frame_start = 0;
        
        // Send image pixels (simple gradient pattern)
        tokens_received = 0;
        
        for (i = 0; i < IMG_SIZE * IMG_SIZE; i = i + 1) begin
            @(posedge clk);
            pixel_in = {8'd10, 8'd20, 8'd30};  // Simple RGB values
            pixel_valid = 1;
            
            // Check for token outputs while loading
            if (token_valid) begin
                tokens_received = tokens_received + 1;
                $display("Token %0d received at pixel %0d", token_idx, i);
            end
        end
        
        @(posedge clk);
        pixel_valid = 0;
        
        // Wait for all tokens
        repeat(1000) begin
            @(posedge clk);
            if (token_valid) begin
                tokens_received = tokens_received + 1;
                $display("Token %0d received", token_idx);
            end
        end
        
        $display("Received %0d tokens (expected %0d)", tokens_received, NUM_PATCHES);
        
        $display("Testbench complete");
        $finish;
    end
    
    initial begin
        #500000;
        $display("TIMEOUT");
        $finish;
    end
endmodule

`endif
