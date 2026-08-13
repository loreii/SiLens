# =============================================================================
# SiLens Vivado Synthesis Script
# =============================================================================
# Creates a Vivado project for SiLens FPGA prototyping
#
# Usage:
#   vivado -mode tcl -source synth_vivado.tcl
#   or
#   vivado -mode batch -source synth_vivado.tcl
#
# Target: Artix-7 200T or Kintex-7 325T
# =============================================================================

# Configuration
set project_name "silens_fpga"
set project_dir "./vivado_project"
set rtl_dir "../../rtl"

# Select target device (uncomment one)
set target_part "xc7a200tfbg676-1"    ;# Artix-7 200T (Nexys Video)
# set target_part "xc7k325tffg900-2"  ;# Kintex-7 325T (KC705)

# Select constraints file based on target
if {[string match "xc7a*" $target_part]} {
    set constraints_file "silens_artix7.xdc"
} else {
    set constraints_file "silens_kintex7.xdc"
}

# Create project
create_project $project_name $project_dir -part $target_part -force

# Set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Verilog [current_project]

# =============================================================================
# Add RTL source files
# =============================================================================

# Common modules
add_files -norecurse [glob -nocomplain $rtl_dir/common/*.v]

# Vision encoder modules
add_files -norecurse [glob -nocomplain $rtl_dir/vision_encoder/*.v]

# Projector modules
add_files -norecurse [glob -nocomplain $rtl_dir/projector/*.v]

# Language model modules
add_files -norecurse [glob -nocomplain $rtl_dir/language_model/*.v]

# Memory modules
add_files -norecurse [glob -nocomplain $rtl_dir/memory/*.v]

# PCIe modules
add_files -norecurse [glob -nocomplain $rtl_dir/pcie/*.v]

# Top-level modules
add_files -norecurse [glob -nocomplain $rtl_dir/top/*.v]

# FPGA wrapper
add_files -norecurse "silens_fpga_wrapper.v"

# Set top module
set_property top silens_fpga_wrapper [current_fileset]

# =============================================================================
# Add constraints
# =============================================================================

add_files -fileset constrs_1 -norecurse $constraints_file
set_property target_constrs_file $constraints_file [current_fileset -constrset]


# =============================================================================
# Synthesis settings
# =============================================================================

set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING on [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION one_hot [get_runs synth_1]

# =============================================================================
# Implementation settings
# =============================================================================

set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraPostPlacementOpt [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]

# =============================================================================
# Run synthesis (optional - uncomment to run automatically)
# =============================================================================

# puts "Starting synthesis..."
# launch_runs synth_1 -jobs 8
# wait_on_run synth_1

# Check synthesis results
# if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
#     error "Synthesis failed!"
# }

# puts "Synthesis complete. Starting implementation..."
# launch_runs impl_1 -jobs 8
# wait_on_run impl_1

# Generate bitstream
# puts "Generating bitstream..."
# launch_runs impl_1 -to_step write_bitstream -jobs 8
# wait_on_run impl_1

# =============================================================================
# Utility procedures
# =============================================================================

proc run_synth {} {
    launch_runs synth_1 -jobs 8
    wait_on_run synth_1
    open_run synth_1
    report_utilization -file utilization_synth.rpt
    report_timing_summary -file timing_synth.rpt
}

proc run_impl {} {
    launch_runs impl_1 -jobs 8
    wait_on_run impl_1
    open_run impl_1
    report_utilization -file utilization_impl.rpt
    report_timing_summary -file timing_impl.rpt
    report_power -file power_impl.rpt
}

proc run_bitstream {} {
    launch_runs impl_1 -to_step write_bitstream -jobs 8
    wait_on_run impl_1
    puts "Bitstream generated: $project_dir/$project_name.runs/impl_1/silens_fpga_wrapper.bit"
}

proc run_all {} {
    run_synth
    run_impl
    run_bitstream
}

# =============================================================================
# Print summary
# =============================================================================

puts ""
puts "=========================================="
puts "SiLens FPGA Project Created Successfully"
puts "=========================================="
puts "Project: $project_name"
puts "Directory: $project_dir"
puts "Target: $target_part"
puts "Constraints: $constraints_file"
puts ""
puts "To synthesize, run: run_synth"
puts "To implement, run: run_impl"
puts "To generate bitstream, run: run_bitstream"
puts "To run all steps, run: run_all"
puts "=========================================="
