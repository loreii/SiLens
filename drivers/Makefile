# SPDX-License-Identifier: GPL-2.0
#
# SiLens Linux Kernel Driver Makefile
#
# Build instructions:
#   make              - Build the kernel module
#   make clean        - Clean build files
#   make install      - Install the module
#   make uninstall    - Remove the module
#   make load         - Load the module
#   make unload       - Unload the module
#

# Module name
MODULE_NAME := silens

# Source files
obj-m := $(MODULE_NAME).o
$(MODULE_NAME)-objs := silens_drv.o

# Kernel build directory
KDIR ?= /lib/modules/$(shell uname -r)/build

# Current directory
PWD := $(shell pwd)

# Compiler flags
ccflags-y := -Wall -Werror

# Debug build (set DEBUG=1 to enable)
ifdef DEBUG
ccflags-y += -DDEBUG -g
endif

# Default target
all: modules

# Build kernel module
modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Clean build files
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f Module.symvers modules.order

# Install module
install: modules
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

# Uninstall module
uninstall:
	rm -f /lib/modules/$(shell uname -r)/extra/$(MODULE_NAME).ko
	depmod -a

# Load module
load: modules
	@if lsmod | grep -q "^$(MODULE_NAME)"; then \
		echo "Module already loaded, reloading..."; \
		rmmod $(MODULE_NAME); \
	fi
	insmod $(MODULE_NAME).ko debug_level=1

# Unload module
unload:
	@if lsmod | grep -q "^$(MODULE_NAME)"; then \
		rmmod $(MODULE_NAME); \
		echo "Module unloaded"; \
	else \
		echo "Module not loaded"; \
	fi

# Create udev rule for device permissions
udev-install:
	@echo 'KERNEL=="silens[0-9]*", MODE="0666"' > /etc/udev/rules.d/99-silens.rules
	udevadm control --reload-rules
	@echo "Udev rules installed"

udev-uninstall:
	rm -f /etc/udev/rules.d/99-silens.rules
	udevadm control --reload-rules
	@echo "Udev rules removed"

# Check kernel configuration
check-config:
	@echo "Checking kernel configuration..."
	@grep -q "CONFIG_PCI=y" /boot/config-$(shell uname -r) && echo "  PCI support: OK" || echo "  PCI support: MISSING"
	@grep -q "CONFIG_DMA_" /boot/config-$(shell uname -r) && echo "  DMA support: OK" || echo "  DMA support: MISSING"

# Show module info
info:
	@if [ -f $(MODULE_NAME).ko ]; then \
		modinfo $(MODULE_NAME).ko; \
	else \
		echo "Module not built yet. Run 'make' first."; \
	fi

# Test build (for CI)
test: modules
	@echo "Build successful"

.PHONY: all modules clean install uninstall load unload udev-install udev-uninstall check-config info test
