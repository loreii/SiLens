/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * SiLens IOCTL Definitions
 *
 * User-space interface definitions for the SiLens driver.
 *
 * Copyright (C) 2024 SiLens Team
 */

#ifndef _SILENS_IOCTL_H_
#define _SILENS_IOCTL_H_

#include <linux/types.h>
#include <linux/ioctl.h>

/* IOCTL magic number */
#define SILENS_IOC_MAGIC    'S'

/* Version information */
struct silens_version {
    __u8 driver_major;
    __u8 driver_minor;
    __u8 driver_patch;
    __u8 hw_major;
    __u8 hw_minor;
    __u8 hw_patch;
    __u8 reserved[2];
};

/* Register access */
struct silens_reg_access {
    __u32 offset;       /* Register offset */
    __u32 value;        /* Value to read/write */
};

/* DMA buffer allocation */
struct silens_dma_alloc {
    __u64 size;         /* Requested size in bytes */
    __u64 phys_addr;    /* Physical/DMA address (output) */
    __s32 handle;       /* Buffer handle (output) */
    __s32 reserved;
};

/* Wait for interrupt */
struct silens_wait_int {
    __u32 mask;         /* Interrupt mask to wait for */
    __u32 timeout_ms;   /* Timeout in milliseconds */
    __u32 status;       /* Interrupt status (output) */
    __u32 reserved;
};

/* DMA transfer request */
struct silens_dma_xfer {
    __s32 buffer_handle;    /* DMA buffer handle */
    __u32 direction;        /* 0 = to device, 1 = from device */
    __u64 device_addr;      /* Address in device memory */
    __u64 offset;           /* Offset in DMA buffer */
    __u64 size;             /* Transfer size */
};

/* Inference request */
struct silens_inference_req {
    __s32 image_handle;     /* DMA buffer handle for image */
    __s32 output_handle;    /* DMA buffer handle for output */
    __u32 img_width;        /* Image width */
    __u32 img_height;       /* Image height */
    __u32 max_tokens;       /* Maximum output tokens */
    __u32 flags;            /* Inference flags */
#define SILENS_INF_FLAG_STREAMING   (1 << 0)
#define SILENS_INF_FLAG_VISION_ONLY (1 << 1)
};

/* Device status */
struct silens_status {
    __u32 status_reg;       /* Raw status register value */
    __u32 ctrl_reg;         /* Raw control register value */
    __u32 version;          /* Hardware version */
    __u32 temperature;      /* Temperature in millidegrees C (if available) */
    __u64 total_inferences; /* Total inference count */
    __u64 total_tokens;     /* Total tokens generated */
};

/* IOCTL commands */

/* Get driver and hardware version */
#define SILENS_IOCTL_GET_VERSION    _IOR(SILENS_IOC_MAGIC, 1, struct silens_version)

/* Read a register */
#define SILENS_IOCTL_READ_REG       _IOWR(SILENS_IOC_MAGIC, 2, struct silens_reg_access)

/* Write a register */
#define SILENS_IOCTL_WRITE_REG      _IOW(SILENS_IOC_MAGIC, 3, struct silens_reg_access)

/* Allocate DMA buffer */
#define SILENS_IOCTL_ALLOC_DMA      _IOWR(SILENS_IOC_MAGIC, 4, struct silens_dma_alloc)

/* Free DMA buffer */
#define SILENS_IOCTL_FREE_DMA       _IOW(SILENS_IOC_MAGIC, 5, int)

/* Start DMA transfer */
#define SILENS_IOCTL_DMA_XFER       _IOW(SILENS_IOC_MAGIC, 6, struct silens_dma_xfer)

/* Wait for interrupt */
#define SILENS_IOCTL_WAIT_INTERRUPT _IOWR(SILENS_IOC_MAGIC, 7, struct silens_wait_int)

/* Reset device */
#define SILENS_IOCTL_RESET          _IO(SILENS_IOC_MAGIC, 8)

/* Start inference */
#define SILENS_IOCTL_START_INFERENCE _IOW(SILENS_IOC_MAGIC, 9, struct silens_inference_req)

/* Get device status */
#define SILENS_IOCTL_GET_STATUS     _IOR(SILENS_IOC_MAGIC, 10, struct silens_status)

/* Enable/disable interrupts */
#define SILENS_IOCTL_SET_INT_ENABLE _IOW(SILENS_IOC_MAGIC, 11, __u32)

#endif /* _SILENS_IOCTL_H_ */
