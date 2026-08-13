/*
 * SiLens Firmware Linker Script
 *
 * Memory layout for soft-core processor on FPGA.
 * Adjust addresses and sizes for your specific implementation.
 */

ENTRY(_start)

/* Memory regions */
MEMORY
{
    /* Boot ROM - contains startup code */
    ROM (rx)  : ORIGIN = 0x00000000, LENGTH = 16K
    
    /* Main RAM - code and data */
    RAM (rwx) : ORIGIN = 0x00010000, LENGTH = 64K
    
    /* Host-visible registers (BAR0 mapping) */
    REGS (rw) : ORIGIN = 0x00001000, LENGTH = 4K
    
    /* Stack */
    STACK (rw) : ORIGIN = 0x0001F000, LENGTH = 4K
}

/* Stack size */
__stack_size = 4K;

SECTIONS
{
    /* Startup code goes to ROM */
    .text.startup :
    {
        *(.text.startup)
        *(.text.startup.*)
    } > ROM
    
    /* Main code */
    .text :
    {
        *(.text)
        *(.text.*)
        *(.rodata)
        *(.rodata.*)
    } > RAM
    
    /* Initialized data */
    .data :
    {
        __data_start = .;
        *(.data)
        *(.data.*)
        __data_end = .;
    } > RAM
    
    /* Uninitialized data */
    .bss (NOLOAD) :
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM
    
    /* Stack */
    .stack (NOLOAD) :
    {
        __stack_bottom = .;
        . = . + __stack_size;
        __stack_top = .;
    } > STACK
    
    /* Heap (remaining RAM) */
    .heap (NOLOAD) :
    {
        __heap_start = .;
        . = ORIGIN(RAM) + LENGTH(RAM);
        __heap_end = .;
    } > RAM
}

/* Provide symbols for startup code */
PROVIDE(__global_pointer = .);
PROVIDE(_stack = __stack_top);
