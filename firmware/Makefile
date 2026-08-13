# SiLens Card Firmware Makefile
#
# This Makefile is configured for building firmware targeting a
# soft-core RISC-V processor on FPGA. Adjust toolchain and flags
# as needed for your specific target.
#

# Project name
PROJECT = silens_fw

# Source files
SRCS = main.c

# Object files
OBJS = $(SRCS:.c=.o)

# Toolchain (RISC-V)
# Adjust this to match your installed toolchain
CROSS_COMPILE ?= riscv32-unknown-elf-
CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump
SIZE = $(CROSS_COMPILE)size

# Architecture flags (RV32I minimal, adjust for your core)
ARCH_FLAGS = -march=rv32i -mabi=ilp32

# Compiler flags
CFLAGS = $(ARCH_FLAGS)
CFLAGS += -Wall -Wextra -Werror
CFLAGS += -Os -ffunction-sections -fdata-sections
CFLAGS += -ffreestanding -nostdlib
CFLAGS += -I.

# Debug build
ifdef DEBUG
CFLAGS += -g -DDEBUG
endif

# Linker script
LDSCRIPT = linker.ld

# Linker flags
LDFLAGS = $(ARCH_FLAGS)
LDFLAGS += -T$(LDSCRIPT)
LDFLAGS += -nostdlib -nostartfiles
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,-Map=$(PROJECT).map

# Output files
ELF = $(PROJECT).elf
BIN = $(PROJECT).bin
HEX = $(PROJECT).hex
DIS = $(PROJECT).dis
MEM = $(PROJECT).mem

# Default target
all: $(BIN) $(HEX) $(MEM) $(DIS) size

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link
$(ELF): $(OBJS) $(LDSCRIPT)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

# Generate binary
$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@

# Generate Intel HEX
$(HEX): $(ELF)
	$(OBJCOPY) -O ihex $< $@

# Generate memory initialization file (for Verilog $readmemh)
$(MEM): $(BIN)
	hexdump -v -e '1/4 "%08x\n"' $< > $@

# Generate disassembly
$(DIS): $(ELF)
	$(OBJDUMP) -d -S $< > $@

# Show size
size: $(ELF)
	$(SIZE) $<

# Clean
clean:
	rm -f $(OBJS) $(ELF) $(BIN) $(HEX) $(MEM) $(DIS) $(PROJECT).map

# Flash to device (placeholder - depends on hardware)
flash: $(BIN)
	@echo "Flash command not configured. Adjust for your hardware."
	@echo "Example: openocd -f interface/ftdi/... -c 'program $(BIN) verify reset exit'"

# Simulation target (for testing with simulator)
sim: $(ELF)
	@echo "Simulation not configured. Use spike, qemu-riscv32, or similar."

.PHONY: all clean size flash sim
