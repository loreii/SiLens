// =============================================================================
// SiLens Power Management Controller
// =============================================================================
// Dynamic power management for the SiLens accelerator with:
//   - Clock gating for idle compute units
//   - Dynamic voltage/frequency scaling (DVFS) interface
//   - Power domain isolation
//   - Thermal throttling interface
//   - Activity monitoring for adaptive power states
//
// Power States:
//   00 - FULL_POWER:   All units active, max performance
//   01 - BALANCED:     Vision or LLM inactive based on phase
//   10 - LOW_POWER:    Reduced clock, non-essential units gated
//   11 - SLEEP:        Minimal power, only control logic active
//
// License: Apache 2.0
// =============================================================================

module power_controller #(
    parameter NUM_COMPUTE_UNITS = 16,
    parameter NUM_MEMORY_BANKS = 4,
    parameter IDLE_THRESHOLD_CYCLES = 1000,
    parameter THERMAL_BITS = 8,
    parameter ACTIVITY_WINDOW = 256
)(
    // Clock and reset
    input  wire         clk,
    input  wire         rst_n,
    
    // Power state control
    input  wire [1:0]   power_state_req,
    output reg  [1:0]   power_state_ack,
    input  wire         auto_power_mgmt_en,
    
    // Activity monitoring
    input  wire [NUM_COMPUTE_UNITS-1:0] compute_active,
    input  wire [NUM_MEMORY_BANKS-1:0]  memory_active,
    input  wire         vision_active,
    input  wire         llm_active,
    input  wire         dma_active,
    
    // Thermal interface
    input  wire [THERMAL_BITS-1:0] temperature,
    input  wire [THERMAL_BITS-1:0] thermal_limit,
    input  wire         thermal_shutdown_req,
    output reg          thermal_throttle,
    
    // Clock gating outputs
    output reg  [NUM_COMPUTE_UNITS-1:0] compute_clk_en,
    output reg  [NUM_MEMORY_BANKS-1:0]  memory_clk_en,
    output reg          vision_clk_en,
    output reg          llm_clk_en,
    output reg          pcie_clk_en,
    
    // DVFS interface
    output reg  [2:0]   dvfs_freq_sel,      // Frequency selection
    output reg  [2:0]   dvfs_voltage_sel,   // Voltage selection
    output reg          dvfs_change_req,
    input  wire         dvfs_change_ack,
    
    // Power domain isolation
    output reg          iso_vision,
    output reg          iso_llm,
    output reg          iso_memory,
    
    // Power good inputs
    input  wire         pg_core,
    input  wire         pg_memory,
    input  wire         pg_io,
    
    // Status/Debug
    output wire [31:0]  power_status,
    output wire [15:0]  power_consumed_mw,
    output reg          power_fault
);

    // =========================================================================
    // Power state definitions
    // =========================================================================
    
    localparam PSTATE_FULL_POWER = 2'b00;
    localparam PSTATE_BALANCED   = 2'b01;
    localparam PSTATE_LOW_POWER  = 2'b10;
    localparam PSTATE_SLEEP      = 2'b11;
    
    // DVFS levels
    localparam DVFS_MAX     = 3'b000;  // 100MHz, 1.0V
    localparam DVFS_HIGH    = 3'b001;  // 80MHz,  0.95V
    localparam DVFS_MEDIUM  = 3'b010;  // 60MHz,  0.9V
    localparam DVFS_LOW     = 3'b011;  // 40MHz,  0.85V
    localparam DVFS_MIN     = 3'b100;  // 20MHz,  0.8V
    
    // =========================================================================
    // State machine for power transitions
    // =========================================================================
    
    localparam PS_IDLE          = 3'd0;
    localparam PS_GATE_CLOCKS   = 3'd1;
    localparam PS_ISOLATE       = 3'd2;
    localparam PS_DVFS_CHANGE   = 3'd3;
    localparam PS_WAIT_STABLE   = 3'd4;
    localparam PS_DEISOLATE     = 3'd5;
    localparam PS_UNGATE_CLOCKS = 3'd6;
    localparam PS_COMPLETE      = 3'd7;
    
    reg [2:0] ps_state;
    reg [1:0] current_power_state;
    reg [1:0] target_power_state;
    reg [15:0] stabilization_counter;
    
    // =========================================================================
    // Activity monitoring
    // =========================================================================
    
    reg [$clog2(ACTIVITY_WINDOW)-1:0] activity_sample_cnt;
    reg [15:0] compute_activity_acc;
    reg [15:0] memory_activity_acc;
    reg [7:0]  compute_utilization;
    reg [7:0]  memory_utilization;
    
    // Idle detection counters
    reg [15:0] compute_idle_cycles [NUM_COMPUTE_UNITS-1:0];
    reg [15:0] global_idle_cycles;
    
    // =========================================================================
    // Thermal management
    // =========================================================================
    
    reg [7:0] thermal_history [3:0];
    reg [1:0] thermal_idx;
    wire [9:0] thermal_avg;
    
    assign thermal_avg = (thermal_history[0] + thermal_history[1] + 
                          thermal_history[2] + thermal_history[3]);
    
    // Thermal throttling hysteresis
    reg thermal_throttle_prev;
    wire thermal_above_limit;
    wire thermal_below_safe;
    
    assign thermal_above_limit = (thermal_avg >> 2) > thermal_limit;
    assign thermal_below_safe  = (thermal_avg >> 2) < (thermal_limit - 8'd10);
    
    // =========================================================================
    // Activity monitoring logic
    // =========================================================================
    
    integer i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            activity_sample_cnt <= 0;
            compute_activity_acc <= 0;
            memory_activity_acc <= 0;
            compute_utilization <= 0;
            memory_utilization <= 0;
            global_idle_cycles <= 0;
            
            for (i = 0; i < NUM_COMPUTE_UNITS; i = i + 1) begin
                compute_idle_cycles[i] <= 0;
            end
        end else begin
            // Sample activity
            activity_sample_cnt <= activity_sample_cnt + 1;
            
            // Accumulate compute activity
            compute_activity_acc <= compute_activity_acc + 
                                    {{(16-NUM_COMPUTE_UNITS){1'b0}}, compute_active};
            memory_activity_acc <= memory_activity_acc + 
                                   {{(16-NUM_MEMORY_BANKS){1'b0}}, memory_active};
            
            // Calculate utilization at end of window
            if (activity_sample_cnt == ACTIVITY_WINDOW - 1) begin
                compute_utilization <= compute_activity_acc[15:8];
                memory_utilization <= memory_activity_acc[15:8];
                compute_activity_acc <= 0;
                memory_activity_acc <= 0;
            end
            
            // Track idle cycles per compute unit
            for (i = 0; i < NUM_COMPUTE_UNITS; i = i + 1) begin
                if (compute_active[i]) begin
                    compute_idle_cycles[i] <= 0;
                end else if (compute_idle_cycles[i] < 16'hFFFF) begin
                    compute_idle_cycles[i] <= compute_idle_cycles[i] + 1;
                end
            end
            
            // Global idle tracking
            if (|compute_active || |memory_active || vision_active || llm_active || dma_active) begin
                global_idle_cycles <= 0;
            end else if (global_idle_cycles < 16'hFFFF) begin
                global_idle_cycles <= global_idle_cycles + 1;
            end
        end
    end
    
    // =========================================================================
    // Thermal management
    // =========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            thermal_history[0] <= 0;
            thermal_history[1] <= 0;
            thermal_history[2] <= 0;
            thermal_history[3] <= 0;
            thermal_idx <= 0;
            thermal_throttle <= 0;
            thermal_throttle_prev <= 0;
        end else begin
            // Sample temperature periodically (every 65536 cycles)
            if (&activity_sample_cnt) begin
                thermal_history[thermal_idx] <= temperature;
                thermal_idx <= thermal_idx + 1;
            end
            
            // Thermal throttling with hysteresis
            thermal_throttle_prev <= thermal_throttle;
            
            if (thermal_shutdown_req) begin
                thermal_throttle <= 1'b1;
            end else if (thermal_above_limit && !thermal_throttle) begin
                thermal_throttle <= 1'b1;
            end else if (thermal_below_safe && thermal_throttle) begin
                thermal_throttle <= 1'b0;
            end
        end
    end
    
    // =========================================================================
    // Automatic power state selection
    // =========================================================================
    
    reg [1:0] auto_power_state;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            auto_power_state <= PSTATE_FULL_POWER;
        end else if (auto_power_mgmt_en) begin
            // Thermal override
            if (thermal_throttle) begin
                auto_power_state <= PSTATE_LOW_POWER;
            end
            // High utilization -> full power
            else if (compute_utilization > 8'd200 || memory_utilization > 8'd200) begin
                auto_power_state <= PSTATE_FULL_POWER;
            end
            // Medium utilization -> balanced
            else if (compute_utilization > 8'd100 || memory_utilization > 8'd100) begin
                auto_power_state <= PSTATE_BALANCED;
            end
            // Extended idle -> sleep
            else if (global_idle_cycles > IDLE_THRESHOLD_CYCLES * 10) begin
                auto_power_state <= PSTATE_SLEEP;
            end
            // Short idle -> low power
            else if (global_idle_cycles > IDLE_THRESHOLD_CYCLES) begin
                auto_power_state <= PSTATE_LOW_POWER;
            end
            // Default -> balanced
            else begin
                auto_power_state <= PSTATE_BALANCED;
            end
        end
    end
    
    // =========================================================================
    // Power state machine
    // =========================================================================
    
    wire [1:0] effective_power_state_req;
    assign effective_power_state_req = auto_power_mgmt_en ? auto_power_state : power_state_req;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            ps_state <= PS_IDLE;
            current_power_state <= PSTATE_FULL_POWER;
            target_power_state <= PSTATE_FULL_POWER;
            power_state_ack <= PSTATE_FULL_POWER;
            stabilization_counter <= 0;
            
            // Default: all clocks enabled
            compute_clk_en <= {NUM_COMPUTE_UNITS{1'b1}};
            memory_clk_en <= {NUM_MEMORY_BANKS{1'b1}};
            vision_clk_en <= 1'b1;
            llm_clk_en <= 1'b1;
            pcie_clk_en <= 1'b1;
            
            // Default: no isolation
            iso_vision <= 1'b0;
            iso_llm <= 1'b0;
            iso_memory <= 1'b0;
            
            // Default: max frequency
            dvfs_freq_sel <= DVFS_MAX;
            dvfs_voltage_sel <= DVFS_MAX;
            dvfs_change_req <= 1'b0;
            
            power_fault <= 1'b0;
            
        end else begin
            case (ps_state)
                PS_IDLE: begin
                    if (effective_power_state_req != current_power_state) begin
                        target_power_state <= effective_power_state_req;
                        ps_state <= PS_GATE_CLOCKS;
                    end
                    power_state_ack <= current_power_state;
                end
                
                PS_GATE_CLOCKS: begin
                    // Gate clocks based on target state
                    case (target_power_state)
                        PSTATE_FULL_POWER: begin
                            compute_clk_en <= {NUM_COMPUTE_UNITS{1'b1}};
                            memory_clk_en <= {NUM_MEMORY_BANKS{1'b1}};
                            vision_clk_en <= 1'b1;
                            llm_clk_en <= 1'b1;
                        end
                        
                        PSTATE_BALANCED: begin
                            // Gate idle compute units
                            for (i = 0; i < NUM_COMPUTE_UNITS; i = i + 1) begin
                                compute_clk_en[i] <= (compute_idle_cycles[i] < IDLE_THRESHOLD_CYCLES);
                            end
                            memory_clk_en <= {NUM_MEMORY_BANKS{1'b1}};
                            vision_clk_en <= vision_active;
                            llm_clk_en <= llm_active;
                        end
                        
                        PSTATE_LOW_POWER: begin
                            // Gate most units
                            compute_clk_en <= {{(NUM_COMPUTE_UNITS-4){1'b0}}, 4'b1111};
                            memory_clk_en <= {{(NUM_MEMORY_BANKS-1){1'b0}}, 1'b1};
                            vision_clk_en <= 1'b0;
                            llm_clk_en <= 1'b0;
                        end
                        
                        PSTATE_SLEEP: begin
                            compute_clk_en <= {NUM_COMPUTE_UNITS{1'b0}};
                            memory_clk_en <= {NUM_MEMORY_BANKS{1'b0}};
                            vision_clk_en <= 1'b0;
                            llm_clk_en <= 1'b0;
                        end
                    endcase
                    
                    ps_state <= PS_ISOLATE;
                end
                
                PS_ISOLATE: begin
                    // Apply isolation for powered-down domains
                    case (target_power_state)
                        PSTATE_LOW_POWER, PSTATE_SLEEP: begin
                            iso_vision <= 1'b1;
                            iso_llm <= 1'b1;
                        end
                        default: begin
                            iso_vision <= 1'b0;
                            iso_llm <= 1'b0;
                        end
                    endcase
                    
                    ps_state <= PS_DVFS_CHANGE;
                end
                
                PS_DVFS_CHANGE: begin
                    // Select DVFS level based on power state
                    case (target_power_state)
                        PSTATE_FULL_POWER: begin
                            dvfs_freq_sel <= thermal_throttle ? DVFS_MEDIUM : DVFS_MAX;
                            dvfs_voltage_sel <= thermal_throttle ? DVFS_MEDIUM : DVFS_MAX;
                        end
                        PSTATE_BALANCED: begin
                            dvfs_freq_sel <= DVFS_HIGH;
                            dvfs_voltage_sel <= DVFS_HIGH;
                        end
                        PSTATE_LOW_POWER: begin
                            dvfs_freq_sel <= DVFS_LOW;
                            dvfs_voltage_sel <= DVFS_LOW;
                        end
                        PSTATE_SLEEP: begin
                            dvfs_freq_sel <= DVFS_MIN;
                            dvfs_voltage_sel <= DVFS_MIN;
                        end
                    endcase
                    
                    dvfs_change_req <= 1'b1;
                    ps_state <= PS_WAIT_STABLE;
                    stabilization_counter <= 0;
                end
                
                PS_WAIT_STABLE: begin
                    if (dvfs_change_ack) begin
                        dvfs_change_req <= 1'b0;
                    end
                    
                    stabilization_counter <= stabilization_counter + 1;
                    
                    // Wait for voltage/frequency to stabilize
                    if (stabilization_counter > 16'd1000 && !dvfs_change_req) begin
                        ps_state <= PS_DEISOLATE;
                    end
                end
                
                PS_DEISOLATE: begin
                    // Remove isolation for active domains
                    case (target_power_state)
                        PSTATE_FULL_POWER, PSTATE_BALANCED: begin
                            iso_vision <= 1'b0;
                            iso_llm <= 1'b0;
                            iso_memory <= 1'b0;
                        end
                        default: ;  // Keep isolation
                    endcase
                    
                    ps_state <= PS_UNGATE_CLOCKS;
                end
                
                PS_UNGATE_CLOCKS: begin
                    // PCIe clock always enabled (except in deep sleep)
                    pcie_clk_en <= (target_power_state != PSTATE_SLEEP);
                    ps_state <= PS_COMPLETE;
                end
                
                PS_COMPLETE: begin
                    current_power_state <= target_power_state;
                    power_state_ack <= target_power_state;
                    ps_state <= PS_IDLE;
                end
                
                default: ps_state <= PS_IDLE;
            endcase
            
            // Power fault detection
            if (!pg_core || !pg_memory || !pg_io) begin
                power_fault <= 1'b1;
            end
        end
    end
    
    // =========================================================================
    // Power consumption estimation
    // =========================================================================
    
    // Simplified power model (milliwatts)
    // Real implementation would use calibrated values
    
    reg [15:0] power_estimate;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            power_estimate <= 0;
        end else begin
            case (current_power_state)
                PSTATE_FULL_POWER: power_estimate <= 16'd2500;  // ~2.5W
                PSTATE_BALANCED:   power_estimate <= 16'd1500;  // ~1.5W
                PSTATE_LOW_POWER:  power_estimate <= 16'd500;   // ~0.5W
                PSTATE_SLEEP:      power_estimate <= 16'd50;    // ~50mW
            endcase
            
            // Adjust based on activity
            power_estimate <= power_estimate + 
                              (compute_utilization * 8) + 
                              (memory_utilization * 4);
        end
    end
    
    assign power_consumed_mw = power_estimate;
    
    // =========================================================================
    // Status output
    // =========================================================================
    
    assign power_status = {
        2'b0,
        thermal_throttle,
        power_fault,
        pg_io,
        pg_memory,
        pg_core,
        auto_power_mgmt_en,
        current_power_state,
        target_power_state,
        ps_state,
        dvfs_freq_sel,
        compute_utilization
    };

endmodule
