/*
 * SiLens Firmware Startup Code
 *
 * Minimal RISC-V startup code for bare-metal environment.
 * Initializes stack, clears BSS, and jumps to main().
 */

.section .text.startup, "ax"
.global _start
.type _start, @function

_start:
    /* Disable interrupts */
    csrci mstatus, 0x8
    
    /* Set up stack pointer */
    la sp, __stack_top
    
    /* Set up global pointer (for relaxation) */
    .option push
    .option norelax
    la gp, __global_pointer
    .option pop
    
    /* Clear BSS section */
    la t0, __bss_start
    la t1, __bss_end
clear_bss:
    bge t0, t1, bss_done
    sw zero, 0(t0)
    addi t0, t0, 4
    j clear_bss
bss_done:
    
    /* Copy initialized data from ROM to RAM if needed */
    /* (Skip if running from RAM) */
    
    /* Jump to main */
    call main
    
    /* If main returns, loop forever */
halt:
    wfi
    j halt

.size _start, . - _start

/*
 * Trap handler (minimal)
 */
.section .text.trap, "ax"
.global _trap_handler
.type _trap_handler, @function

_trap_handler:
    /* Save registers */
    addi sp, sp, -64
    sw ra, 0(sp)
    sw t0, 4(sp)
    sw t1, 8(sp)
    sw t2, 12(sp)
    sw a0, 16(sp)
    sw a1, 20(sp)
    sw a2, 24(sp)
    sw a3, 28(sp)
    
    /* Get trap cause */
    csrr t0, mcause
    
    /* Check if interrupt or exception */
    bltz t0, handle_interrupt
    
handle_exception:
    /* Exception handling - for now, just hang */
    j halt
    
handle_interrupt:
    /* Clear MSB to get interrupt number */
    andi t0, t0, 0x7FFFFFFF
    
    /* Handle timer interrupt (interrupt 7) */
    li t1, 7
    beq t0, t1, timer_interrupt
    
    /* Handle external interrupt (interrupt 11) */
    li t1, 11
    beq t0, t1, external_interrupt
    
    /* Unknown interrupt */
    j trap_return
    
timer_interrupt:
    /* Handle timer interrupt */
    j trap_return
    
external_interrupt:
    /* Handle external interrupt */
    j trap_return
    
trap_return:
    /* Restore registers */
    lw ra, 0(sp)
    lw t0, 4(sp)
    lw t1, 8(sp)
    lw t2, 12(sp)
    lw a0, 16(sp)
    lw a1, 20(sp)
    lw a2, 24(sp)
    lw a3, 28(sp)
    addi sp, sp, 64
    
    mret

.size _trap_handler, . - _trap_handler
