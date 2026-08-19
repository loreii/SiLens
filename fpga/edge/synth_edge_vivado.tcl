# =============================================================================
# SiLens Edge Vivado Synthesis Script
# =============================================================================
# Creates a Vivado project for SiLens Edge FPGA prototyping.
#
# Target: NanoViT-12M + Classifier on compact FPGA platforms
# - Primary: Artix-7 35T (Arty A7-35T, Basys 3) - lower cost, tighter fit
# - Alternate: Artix-7 100T (Arty A7-100T, Nexys A7) - more headroom
#
# Usage:
#   vivado -mode tcl -source synth_edge_vivado.tcl
#   or
#   vivado -mode batch -source synth_edge_vivado.tcl
#
# License: Apache 2.0
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================

set project_name "silens_edge_fpga"
set project_dir "./vivado_project_edge"

# Path configuration (relative to this script location)
set script_dir [file dirname [info script]]
set fpga_edge_dir $script_dir
set repo_root [file normalize "$script_dir/../.."]

# RTL source paths
set edge_openlane_dir "$repo_root/variants/silens-edge/openlane"
set shared_openlane_dir "$repo_root/openlane"

# Select target device (uncomment one)
set target_part "xc7a35ticsg324-1L"     ;# Artix-7 35T (Arty A7-35T, low power)
# set target_part "xc7a100tcsg324-1"    ;# Artix-7 100T (Arty A7-100T, more resources)

# Select constraints file based on target
if {[string match "*35t*" $target_part]} {
    set constraints_file "$fpga_edge_dir/silens_edge_artix7_35t.xdc"
    set target_name "Artix-7 35T"
} else {
    # For 100T, use same constraints with adjusted utilization targets
    set constraints_file "$fpga_edge_dir/silens_edge_artix7_35t.xdc"
    set target_name "Artix-7 100T"
}

# Number of parallel jobs for synthesis/implementation
set num_jobs 8

# =============================================================================
# Create Project
# =============================================================================

puts "=============================================="
puts "SiLens Edge FPGA Synthesis"
puts "=============================================="
puts "Target: $target_name ($target_part)"
puts "Project: $project_name"
puts ""

create_project $project_name $project_dir -part $target_part -force

# Set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Verilog [current_project]
set_property default_lib work [current_project]

# =============================================================================
# Add Verilog Define for Xilinx Conditional Compilation
# =============================================================================

set_property verilog_define {XILINX=1} [current_fileset]

# =============================================================================
# Add RTL Source Files - Hierarchical Organization
# =============================================================================

puts "Adding RTL sources..."

# -----------------------------------------------------------------------------
# Level 1: Shared Primitives (from openlane/level1/)
# -----------------------------------------------------------------------------
puts "  Level 1 primitives..."

# Attention head
if {[file exists "$shared_openlane_dir/level1/attention_head/src/attention_head.v"]} {
    add_files -norecurse "$shared_openlane_dir/level1/attention_head/src/attention_head.v"
}

# Layer normalization block
if {[file exists "$shared_openlane_dir/level1/layer_norm_block/src/layer_norm_block.v"]} {
    add_files -norecurse "$shared_openlane_dir/level1/layer_norm_block/src/layer_norm_block.v"
}

# MLP block
if {[file exists "$shared_openlane_dir/level1/mlp_block/src/mlp_block.v"]} {
    add_files -norecurse "$shared_openlane_dir/level1/mlp_block/src/mlp_block.v"
}

# RMS normalization block
if {[file exists "$shared_openlane_dir/level1/rms_norm_block/src/rms_norm_block.v"]} {
    add_files -norecurse "$shared_openlane_dir/level1/rms_norm_block/src/rms_norm_block.v"
}

# SiLU activation unit
if {[file exists "$shared_openlane_dir/level1/silu_unit/src/silu_unit.v"]} {
    add_files -norecurse "$shared_openlane_dir/level1/silu_unit/src/silu_unit.v"
}

# Softmax unit
if {[file exists "$shared_openlane_dir/level1/softmax_unit/src/softmax_unit.v"]} {
    add_files -norecurse "$shared_openlane_dir/level1/softmax_unit/src/softmax_unit.v"
}

# Ternary MAC array (core compute primitive)
if {[file exists "$shared_openlane_dir/level1/ternary_mac_array_64/src/ternary_mac_array_64.v"]} {
    add_files -norecurse "$shared_openlane_dir/level1/ternary_mac_array_64/src/ternary_mac_array_64.v"
}

# -----------------------------------------------------------------------------
# Level 2: Shared Blocks (from openlane/level2/)
# -----------------------------------------------------------------------------
puts "  Level 2 blocks..."

# Embedding block
if {[file exists "$shared_openlane_dir/level2/embedding_block/src/embedding_block.v"]} {
    add_files -norecurse "$shared_openlane_dir/level2/embedding_block/src/embedding_block.v"
}

# Projector block
if {[file exists "$shared_openlane_dir/level2/projector_block/src/projector_block.v"]} {
    add_files -norecurse "$shared_openlane_dir/level2/projector_block/src/projector_block.v"
}

# Transformer block for LLM
if {[file exists "$shared_openlane_dir/level2/transformer_block_llm/src/transformer_block_llm.v"]} {
    add_files -norecurse "$shared_openlane_dir/level2/transformer_block_llm/src/transformer_block_llm.v"
}

# Transformer block for vision
if {[file exists "$shared_openlane_dir/level2/transformer_block_vision/src/transformer_block_vision.v"]} {
    add_files -norecurse "$shared_openlane_dir/level2/transformer_block_vision/src/transformer_block_vision.v"
}

# -----------------------------------------------------------------------------
# Level 3: Edge-Specific Modules (from variants/silens-edge/openlane/level3/)
# -----------------------------------------------------------------------------
puts "  Level 3 Edge modules..."

# Classifier head (Edge-specific)
if {[file exists "$edge_openlane_dir/level3/classifier_head/src/classifier_head.v"]} {
    add_files -norecurse "$edge_openlane_dir/level3/classifier_head/src/classifier_head.v"
}

# IO Edge module (SPI/I2C interfaces)
if {[file exists "$edge_openlane_dir/level3/io_edge/src/io_edge.v"]} {
    add_files -norecurse "$edge_openlane_dir/level3/io_edge/src/io_edge.v"
}

# SRAM 256KB (Edge memory subsystem)
if {[file exists "$edge_openlane_dir/level3/sram_256kb/src/sram_256kb.v"]} {
    add_files -norecurse "$edge_openlane_dir/level3/sram_256kb/src/sram_256kb.v"
}

# Vision Nano module (NanoViT encoder)
if {[file exists "$edge_openlane_dir/level3/vision_nano/src/vision_nano.v"]} {
    add_files -norecurse "$edge_openlane_dir/level3/vision_nano/src/vision_nano.v"
}

# -----------------------------------------------------------------------------
# Level 4: Edge SoC Top (from variants/silens-edge/openlane/level4/)
# -----------------------------------------------------------------------------
puts "  Level 4 Edge SoC..."

if {[file exists "$edge_openlane_dir/level4/silens_edge_soc/src/silens_edge_soc.v"]} {
    add_files -norecurse "$edge_openlane_dir/level4/silens_edge_soc/src/silens_edge_soc.v"
}

# -----------------------------------------------------------------------------
# FPGA Wrapper (from fpga/edge/)
# -----------------------------------------------------------------------------
puts "  FPGA wrapper..."

add_files -norecurse "$fpga_edge_dir/silens_edge_fpga_wrapper.v"

# Set top module
set_property top silens_edge_fpga_wrapper [current_fileset]

puts "  RTL sources added successfully."

# =============================================================================
# Add Constraints
# =============================================================================

puts "Adding constraints: $constraints_file"
add_files -fileset constrs_1 -norecurse $constraints_file
set_property target_constrs_file $constraints_file [current_fileset -constrset]

# =============================================================================
# Synthesis Settings - Optimized for AREA (smaller FPGA)
# =============================================================================

puts "Configuring synthesis for area optimization..."

# Primary synthesis directives for area optimization
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING on [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION one_hot [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING on [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.SHREG_MIN_SIZE 5 [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.MAX_BRAM 50 [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.MAX_DSP 90 [get_runs synth_1]

# For 100T target, allow more resources
if {[string match "*100t*" $target_part]} {
    set_property STEPS.SYNTH_DESIGN.ARGS.MAX_BRAM 135 [get_runs synth_1]
    set_property STEPS.SYNTH_DESIGN.ARGS.MAX_DSP 240 [get_runs synth_1]
}

# =============================================================================
# Implementation Settings - Optimized for Area and Timing Closure
# =============================================================================

puts "Configuring implementation settings..."

# Optimization phase
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE ExploreArea [get_runs impl_1]

# Placement phase - optimize for area with timing awareness
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraPostPlacementOpt [get_runs impl_1]

# Physical optimization
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

# Routing phase
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]

# Post-route physical optimization for timing closure
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

# =============================================================================
# Utility Procedures
# =============================================================================

proc run_synth {} {
    global num_jobs project_dir project_name
    
    puts ""
    puts "=============================================="
    puts "Starting Synthesis..."
    puts "=============================================="
    
    launch_runs synth_1 -jobs $num_jobs
    wait_on_run synth_1
    
    # Check synthesis status
    set synth_status [get_property STATUS [get_runs synth_1]]
    set synth_progress [get_property PROGRESS [get_runs synth_1]]
    
    if {$synth_progress != "100%"} {
        puts "ERROR: Synthesis failed!"
        puts "Status: $synth_status"
        return -1
    }
    
    open_run synth_1
    
    # Generate reports
    set reports_dir "$project_dir/reports"
    file mkdir $reports_dir
    
    report_utilization -file "$reports_dir/utilization_synth.rpt"
    report_timing_summary -file "$reports_dir/timing_synth.rpt"
    report_design_analysis -file "$reports_dir/design_analysis_synth.rpt"
    
    puts ""
    puts "Synthesis complete!"
    puts "Reports saved to: $reports_dir"
    
    # Print utilization summary
    puts ""
    puts "Resource Utilization Summary (Post-Synthesis):"
    report_utilization -hierarchical -hierarchical_depth 2
    
    return 0
}

proc run_impl {} {
    global num_jobs project_dir project_name
    
    puts ""
    puts "=============================================="
    puts "Starting Implementation..."
    puts "=============================================="
    
    launch_runs impl_1 -jobs $num_jobs
    wait_on_run impl_1
    
    # Check implementation status
    set impl_status [get_property STATUS [get_runs impl_1]]
    set impl_progress [get_property PROGRESS [get_runs impl_1]]
    
    if {$impl_progress != "100%"} {
        puts "ERROR: Implementation failed!"
        puts "Status: $impl_status"
        return -1
    }
    
    open_run impl_1
    
    # Generate reports
    set reports_dir "$project_dir/reports"
    file mkdir $reports_dir
    
    report_utilization -file "$reports_dir/utilization_impl.rpt"
    report_timing_summary -file "$reports_dir/timing_impl.rpt"
    report_power -file "$reports_dir/power_impl.rpt"
    report_clock_utilization -file "$reports_dir/clock_utilization.rpt"
    report_drc -file "$reports_dir/drc_impl.rpt"
    
    puts ""
    puts "Implementation complete!"
    puts "Reports saved to: $reports_dir"
    
    # Print timing summary
    puts ""
    puts "Timing Summary (Post-Implementation):"
    report_timing_summary -max_paths 5
    
    return 0
}

proc run_bitstream {} {
    global num_jobs project_dir project_name
    
    puts ""
    puts "=============================================="
    puts "Generating Bitstream..."
    puts "=============================================="
    
    launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
    wait_on_run impl_1
    
    # Check if bitstream generation succeeded
    set bit_file "$project_dir/${project_name}.runs/impl_1/silens_edge_fpga_wrapper.bit"
    
    if {[file exists $bit_file]} {
        puts ""
        puts "Bitstream generated successfully!"
        puts "Location: $bit_file"
        
        # Copy to output directory
        file mkdir "$project_dir/output"
        file copy -force $bit_file "$project_dir/output/silens_edge.bit"
        puts "Copied to: $project_dir/output/silens_edge.bit"
        
        return 0
    } else {
        puts "ERROR: Bitstream generation failed!"
        return -1
    }
}

proc run_all {} {
    puts ""
    puts "=============================================="
    puts "Running Full Build Flow"
    puts "=============================================="
    
    set start_time [clock seconds]
    
    if {[run_synth] != 0} {
        puts "Build failed at synthesis stage."
        return -1
    }
    
    if {[run_impl] != 0} {
        puts "Build failed at implementation stage."
        return -1
    }
    
    if {[run_bitstream] != 0} {
        puts "Build failed at bitstream stage."
        return -1
    }
    
    set end_time [clock seconds]
    set elapsed [expr {$end_time - $start_time}]
    set minutes [expr {$elapsed / 60}]
    set seconds [expr {$elapsed % 60}]
    
    puts ""
    puts "=============================================="
    puts "Build Complete!"
    puts "Total time: ${minutes}m ${seconds}s"
    puts "=============================================="
    
    return 0
}

proc check_resources {} {
    puts ""
    puts "=============================================="
    puts "Resource Availability Check"
    puts "=============================================="
    
    # Get device info
    set part [get_property PART [current_project]]
    puts "Target device: $part"
    puts ""
    
    # Print available resources
    puts "Available resources:"
    report_property [get_parts $part] {AVAILABLE_IOBS BLOCKRAM_SITES CLB_COUNT DSP_SITES FLIPFLOP_COUNT LUT_COUNT}
}

proc export_hw {} {
    global project_dir
    
    puts ""
    puts "Exporting hardware for SDK..."
    
    set hw_file "$project_dir/output/silens_edge_hw.xsa"
    write_hw_platform -fixed -force -include_bit -file $hw_file
    
    puts "Hardware exported to: $hw_file"
}

# =============================================================================
# Print Summary
# =============================================================================

puts ""
puts "=============================================="
puts "SiLens Edge FPGA Project Created Successfully"
puts "=============================================="
puts ""
puts "Project Configuration:"
puts "  Project Name:   $project_name"
puts "  Project Dir:    $project_dir"
puts "  Target Device:  $target_part ($target_name)"
puts "  Constraints:    $constraints_file"
puts ""
puts "Source Hierarchy:"
puts "  Level 1 (Primitives):  $shared_openlane_dir/level1/"
puts "  Level 2 (Blocks):      $shared_openlane_dir/level2/"
puts "  Level 3 (Edge):        $edge_openlane_dir/level3/"
puts "  Level 4 (SoC):         $edge_openlane_dir/level4/"
puts "  FPGA Wrapper:          $fpga_edge_dir/silens_edge_fpga_wrapper.v"
puts ""
puts "Synthesis Strategy: Area-Optimized (AreaOptimized_high)"
puts "  - Resource sharing enabled"
puts "  - Retiming enabled"
puts "  - BRAM limit: [expr {[string match "*100t*" $target_part] ? 135 : 50}]"
puts "  - DSP limit:  [expr {[string match "*100t*" $target_part] ? 240 : 90}]"
puts ""
puts "Available Commands:"
puts "  run_synth      - Run synthesis only"
puts "  run_impl       - Run implementation (after synth)"
puts "  run_bitstream  - Generate bitstream (after impl)"
puts "  run_all        - Run complete flow (synth → impl → bitstream)"
puts "  check_resources - Display available FPGA resources"
puts "  export_hw      - Export hardware platform (.xsa)"
puts ""
puts "Define: XILINX=1 (for conditional compilation)"
puts ""
puts "=============================================="
puts "Ready! Run 'run_all' to build complete design."
puts "=============================================="
