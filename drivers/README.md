# SiLens Linux Kernel Driver

Linux kernel module for the SiLens Vision-Language AI Accelerator.

## Overview

This driver provides:
- PCIe device registration and BAR mapping
- MSI/MSI-X interrupt handling
- DMA buffer allocation and management
- Character device interface (`/dev/silensN`)
- Memory mapping for direct register and buffer access

## Requirements

- Linux kernel 5.4 or later
- Kernel headers installed
- Root privileges for loading/installation

## Building

### Quick Build

```bash
make
```

### Debug Build

```bash
make DEBUG=1
```

### Clean

```bash
make clean
```

## Installation

### Temporary (until reboot)

```bash
sudo make load
```

### Permanent

```bash
sudo make install
sudo make udev-install  # For non-root access
```

## Usage

### Loading the Module

```bash
# Load with default settings
sudo insmod silens.ko

# Load with debug output
sudo insmod silens.ko debug_level=2
```

### Module Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `debug_level` | 0 | Debug output level (0=off, 1=basic, 2=verbose) |

### Check if Loaded

```bash
lsmod | grep silens
dmesg | grep -i silens
ls -la /dev/silens*
```

### Unloading

```bash
sudo rmmod silens
```


## IOCTL Interface

The driver provides the following IOCTL commands via `silens_ioctl.h`:

### Version Query

```c
#include "silens_ioctl.h"

int fd = open("/dev/silens0", O_RDWR);
struct silens_version ver;
ioctl(fd, SILENS_IOCTL_GET_VERSION, &ver);
printf("Driver: %d.%d.%d\n", ver.driver_major, ver.driver_minor, ver.driver_patch);
printf("Hardware: %d.%d.%d\n", ver.hw_major, ver.hw_minor, ver.hw_patch);
```

### Register Access

```c
// Read register
struct silens_reg_access reg = { .offset = 0x004 };  // STATUS register
ioctl(fd, SILENS_IOCTL_READ_REG, &reg);
printf("Status: 0x%08x\n", reg.value);

// Write register
reg.offset = 0x000;  // CTRL register
reg.value = 0x01;    // Enable
ioctl(fd, SILENS_IOCTL_WRITE_REG, &reg);
```

### DMA Buffer Allocation

```c
// Allocate buffer
struct silens_dma_alloc alloc = { .size = 1024 * 1024 };  // 1MB
ioctl(fd, SILENS_IOCTL_ALLOC_DMA, &alloc);
printf("Handle: %d, Physical: 0x%llx\n", alloc.handle, alloc.phys_addr);

// Memory-map the buffer
void *buf = mmap(NULL, alloc.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                 fd, (alloc.handle + 1) * 0x1000);

// Use the buffer...

// Free buffer
ioctl(fd, SILENS_IOCTL_FREE_DMA, &alloc.handle);
munmap(buf, alloc.size);
```

### Wait for Interrupt

```c
struct silens_wait_int wait = {
    .mask = 0x20,        // TOKEN_READY bit
    .timeout_ms = 5000   // 5 second timeout
};
int ret = ioctl(fd, SILENS_IOCTL_WAIT_INTERRUPT, &wait);
if (ret == 0) {
    printf("Interrupt received: 0x%08x\n", wait.status);
} else if (ret == -ETIMEDOUT) {
    printf("Timeout waiting for interrupt\n");
}
```

## Register Map

| Offset | Name | Description |
|--------|------|-------------|
| 0x000 | CTRL | Control register |
| 0x004 | STATUS | Status register |
| 0x008 | IMG_ADDR | Image buffer address |
| 0x00C | IMG_SIZE | Image dimensions |
| 0x010 | OUT_ADDR | Output buffer address |
| 0x014 | OUT_LEN | Output length |
| 0x018 | TOKEN_OUT | Current output token |
| 0x01C | TOKEN_VALID | Token valid flag |
| 0x028 | INT_STATUS | Interrupt status |
| 0x02C | INT_ENABLE | Interrupt enable mask |
| 0x100 | DMA_CTRL | DMA control |
| 0x104 | DMA_STATUS | DMA status |
| 0x1F0 | VERSION | Hardware version |

## Troubleshooting

### Device not detected

```bash
# Check PCI devices
lspci -d 1e88:0001

# Check kernel messages
dmesg | tail -50
```

### Permission denied

```bash
# Install udev rules for non-root access
sudo make udev-install

# Or manually:
sudo chmod 666 /dev/silens0
```

### Module won't load

```bash
# Check for conflicts
lsmod | grep silens

# Check kernel log
dmesg | tail -20

# Verify kernel headers
make check-config
```

## Files

| File | Description |
|------|-------------|
| `silens_drv.c` | Main driver source |
| `silens_ioctl.h` | IOCTL definitions (userspace API) |
| `Makefile` | Build system |
| `README.md` | This file |

## License

GPL v2. See LICENSE file in the project root.
