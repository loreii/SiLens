// SPDX-License-Identifier: GPL-2.0-only
/*
 * SiLens PCIe Driver
 *
 * Linux kernel driver for SiLens Vision-Language AI Accelerator.
 * Provides:
 *   - PCIe device registration and BAR mapping
 *   - Interrupt handling (MSI/MSI-X)
 *   - DMA buffer allocation
 *   - Character device interface for userspace
 *
 * Copyright (C) 2024 SiLens Team
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/interrupt.h>
#include <linux/dma-mapping.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/wait.h>
#include <linux/poll.h>

#include "silens_ioctl.h"

#define DRIVER_NAME     "silens"
#define DRIVER_VERSION  "0.1.0"
#define DRIVER_DESC     "SiLens Vision-Language AI Accelerator Driver"

/* PCI Device IDs */
#define SILENS_VENDOR_ID    0x1E88
#define SILENS_DEVICE_ID    0x0001

/* BAR definitions */
#define SILENS_BAR_REGS     0   /* BAR0: Registers */
#define SILENS_BAR_DMA      1   /* BAR1: DMA buffers */
#define SILENS_BAR0_SIZE    0x1000      /* 4KB */
#define SILENS_BAR1_SIZE    0x10000000  /* 256MB */

/* Register offsets */
#define REG_CTRL            0x000
#define REG_STATUS          0x004
#define REG_IMG_ADDR        0x008
#define REG_IMG_SIZE        0x00C
#define REG_OUT_ADDR        0x010
#define REG_OUT_LEN         0x014
#define REG_TOKEN_OUT       0x018
#define REG_TOKEN_VALID     0x01C
#define REG_INT_STATUS      0x028
#define REG_INT_ENABLE      0x02C
#define REG_DMA_CTRL        0x100
#define REG_DMA_STATUS      0x104
#define REG_VERSION         0x1F0


/* Status bits */
#define STATUS_READY        (1 << 0)
#define STATUS_BUSY         (1 << 1)
#define STATUS_ERROR        (1 << 2)
#define STATUS_VISION_DONE  (1 << 3)
#define STATUS_LLM_DONE     (1 << 4)
#define STATUS_TOKEN_READY  (1 << 5)

/* Control bits */
#define CTRL_ENABLE         (1 << 0)
#define CTRL_START          (1 << 1)
#define CTRL_RESET          (1 << 2)

/* Maximum devices supported */
#define SILENS_MAX_DEVICES  8

/* DMA buffer limits */
#define SILENS_MAX_DMA_BUFFERS  16
#define SILENS_MAX_DMA_SIZE     (64 * 1024 * 1024)  /* 64MB max per buffer */

/* Module parameters */
static int debug_level = 0;
module_param(debug_level, int, 0644);
MODULE_PARM_DESC(debug_level, "Debug output level (0=off, 1=basic, 2=verbose)");

/* Global state */
static dev_t silens_devno;
static struct class *silens_class;
static int silens_num_devices;

/* Per-device DMA buffer tracking */
struct silens_dma_buffer {
    void *cpu_addr;
    dma_addr_t dma_addr;
    size_t size;
    int in_use;
};

/* Per-device structure */
struct silens_device {
    struct pci_dev *pdev;
    struct cdev cdev;
    struct device *dev;
    int minor;
    
    /* BAR mappings */
    void __iomem *bar0;     /* Registers */
    resource_size_t bar0_start;
    resource_size_t bar0_len;
    
    /* Interrupt handling */
    int irq;
    int msi_enabled;
    
    /* DMA buffers */
    struct silens_dma_buffer dma_buffers[SILENS_MAX_DMA_BUFFERS];
    spinlock_t dma_lock;
    
    /* Wait queue for blocking operations */
    wait_queue_head_t wait_queue;
    
    /* Device state */
    atomic_t open_count;
    struct mutex mutex;
    unsigned int int_status;
};

/* Forward declarations */
static int silens_open(struct inode *inode, struct file *filp);
static int silens_release(struct inode *inode, struct file *filp);
static long silens_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);
static int silens_mmap(struct file *filp, struct vm_area_struct *vma);
static unsigned int silens_poll(struct file *filp, poll_table *wait);


/* File operations */
static const struct file_operations silens_fops = {
    .owner          = THIS_MODULE,
    .open           = silens_open,
    .release        = silens_release,
    .unlocked_ioctl = silens_ioctl,
    .mmap           = silens_mmap,
    .poll           = silens_poll,
};

/* PCI device ID table */
static const struct pci_device_id silens_pci_ids[] = {
    { PCI_DEVICE(SILENS_VENDOR_ID, SILENS_DEVICE_ID) },
    { 0, }
};
MODULE_DEVICE_TABLE(pci, silens_pci_ids);

/* ============================================================================
 * Register Access
 * ============================================================================ */

static inline u32 silens_read_reg(struct silens_device *sdev, u32 offset)
{
    return ioread32(sdev->bar0 + offset);
}

static inline void silens_write_reg(struct silens_device *sdev, u32 offset, u32 value)
{
    iowrite32(value, sdev->bar0 + offset);
}

/* ============================================================================
 * Interrupt Handler
 * ============================================================================ */

static irqreturn_t silens_irq_handler(int irq, void *data)
{
    struct silens_device *sdev = data;
    u32 int_status;
    
    /* Read and acknowledge interrupt status */
    int_status = silens_read_reg(sdev, REG_INT_STATUS);
    if (!int_status)
        return IRQ_NONE;
    
    /* Clear interrupts */
    silens_write_reg(sdev, REG_INT_STATUS, int_status);
    
    /* Store status for userspace */
    sdev->int_status |= int_status;
    
    /* Wake up waiting processes */
    wake_up_interruptible(&sdev->wait_queue);
    
    if (debug_level >= 2)
        dev_dbg(&sdev->pdev->dev, "IRQ: status=0x%08x\n", int_status);
    
    return IRQ_HANDLED;
}


/* ============================================================================
 * DMA Buffer Management
 * ============================================================================ */

static int silens_alloc_dma_buffer(struct silens_device *sdev,
                                   struct silens_dma_alloc *alloc)
{
    struct silens_dma_buffer *buf;
    unsigned long flags;
    int i;
    
    if (alloc->size > SILENS_MAX_DMA_SIZE)
        return -EINVAL;
    
    spin_lock_irqsave(&sdev->dma_lock, flags);
    
    /* Find free slot */
    for (i = 0; i < SILENS_MAX_DMA_BUFFERS; i++) {
        if (!sdev->dma_buffers[i].in_use)
            break;
    }
    
    if (i >= SILENS_MAX_DMA_BUFFERS) {
        spin_unlock_irqrestore(&sdev->dma_lock, flags);
        return -ENOMEM;
    }
    
    buf = &sdev->dma_buffers[i];
    buf->in_use = 1;
    spin_unlock_irqrestore(&sdev->dma_lock, flags);
    
    /* Allocate coherent DMA buffer */
    buf->cpu_addr = dma_alloc_coherent(&sdev->pdev->dev, alloc->size,
                                       &buf->dma_addr, GFP_KERNEL);
    if (!buf->cpu_addr) {
        spin_lock_irqsave(&sdev->dma_lock, flags);
        buf->in_use = 0;
        spin_unlock_irqrestore(&sdev->dma_lock, flags);
        return -ENOMEM;
    }
    
    buf->size = alloc->size;
    
    /* Return info to userspace */
    alloc->handle = i;
    alloc->phys_addr = buf->dma_addr;
    
    if (debug_level >= 1)
        dev_info(&sdev->pdev->dev, "Allocated DMA buffer %d: %zu bytes at 0x%llx\n",
                 i, buf->size, (unsigned long long)buf->dma_addr);
    
    return 0;
}

static int silens_free_dma_buffer(struct silens_device *sdev, int handle)
{
    struct silens_dma_buffer *buf;
    unsigned long flags;
    
    if (handle < 0 || handle >= SILENS_MAX_DMA_BUFFERS)
        return -EINVAL;
    
    spin_lock_irqsave(&sdev->dma_lock, flags);
    buf = &sdev->dma_buffers[handle];
    
    if (!buf->in_use) {
        spin_unlock_irqrestore(&sdev->dma_lock, flags);
        return -EINVAL;
    }
    
    spin_unlock_irqrestore(&sdev->dma_lock, flags);
    
    /* Free DMA buffer */
    if (buf->cpu_addr) {
        dma_free_coherent(&sdev->pdev->dev, buf->size,
                          buf->cpu_addr, buf->dma_addr);
    }
    
    spin_lock_irqsave(&sdev->dma_lock, flags);
    buf->cpu_addr = NULL;
    buf->dma_addr = 0;
    buf->size = 0;
    buf->in_use = 0;
    spin_unlock_irqrestore(&sdev->dma_lock, flags);
    
    if (debug_level >= 1)
        dev_info(&sdev->pdev->dev, "Freed DMA buffer %d\n", handle);
    
    return 0;
}


/* ============================================================================
 * Character Device Operations
 * ============================================================================ */

static int silens_open(struct inode *inode, struct file *filp)
{
    struct silens_device *sdev;
    
    sdev = container_of(inode->i_cdev, struct silens_device, cdev);
    filp->private_data = sdev;
    
    atomic_inc(&sdev->open_count);
    
    if (debug_level >= 1)
        dev_info(&sdev->pdev->dev, "Device opened (count=%d)\n",
                 atomic_read(&sdev->open_count));
    
    return 0;
}

static int silens_release(struct inode *inode, struct file *filp)
{
    struct silens_device *sdev = filp->private_data;
    
    atomic_dec(&sdev->open_count);
    
    if (debug_level >= 1)
        dev_info(&sdev->pdev->dev, "Device closed (count=%d)\n",
                 atomic_read(&sdev->open_count));
    
    return 0;
}

static long silens_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct silens_device *sdev = filp->private_data;
    void __user *argp = (void __user *)arg;
    int ret = 0;
    
    mutex_lock(&sdev->mutex);
    
    switch (cmd) {
    case SILENS_IOCTL_GET_VERSION: {
        struct silens_version ver;
        u32 hw_ver;
        
        hw_ver = silens_read_reg(sdev, REG_VERSION);
        ver.driver_major = 0;
        ver.driver_minor = 1;
        ver.driver_patch = 0;
        ver.hw_major = (hw_ver >> 16) & 0xFF;
        ver.hw_minor = (hw_ver >> 8) & 0xFF;
        ver.hw_patch = hw_ver & 0xFF;
        
        if (copy_to_user(argp, &ver, sizeof(ver)))
            ret = -EFAULT;
        break;
    }
    
    case SILENS_IOCTL_READ_REG: {
        struct silens_reg_access reg;
        
        if (copy_from_user(&reg, argp, sizeof(reg))) {
            ret = -EFAULT;
            break;
        }
        
        if (reg.offset >= SILENS_BAR0_SIZE) {
            ret = -EINVAL;
            break;
        }
        
        reg.value = silens_read_reg(sdev, reg.offset);
        
        if (copy_to_user(argp, &reg, sizeof(reg)))
            ret = -EFAULT;
        break;
    }
    
    case SILENS_IOCTL_WRITE_REG: {
        struct silens_reg_access reg;
        
        if (copy_from_user(&reg, argp, sizeof(reg))) {
            ret = -EFAULT;
            break;
        }
        
        if (reg.offset >= SILENS_BAR0_SIZE) {
            ret = -EINVAL;
            break;
        }
        
        silens_write_reg(sdev, reg.offset, reg.value);
        break;
    }

    
    case SILENS_IOCTL_ALLOC_DMA: {
        struct silens_dma_alloc alloc;
        
        if (copy_from_user(&alloc, argp, sizeof(alloc))) {
            ret = -EFAULT;
            break;
        }
        
        ret = silens_alloc_dma_buffer(sdev, &alloc);
        if (ret)
            break;
        
        if (copy_to_user(argp, &alloc, sizeof(alloc)))
            ret = -EFAULT;
        break;
    }
    
    case SILENS_IOCTL_FREE_DMA: {
        int handle;
        
        if (get_user(handle, (int __user *)argp)) {
            ret = -EFAULT;
            break;
        }
        
        ret = silens_free_dma_buffer(sdev, handle);
        break;
    }
    
    case SILENS_IOCTL_WAIT_INTERRUPT: {
        struct silens_wait_int wait_int;
        
        if (copy_from_user(&wait_int, argp, sizeof(wait_int))) {
            ret = -EFAULT;
            break;
        }
        
        /* Wait for specified interrupt */
        mutex_unlock(&sdev->mutex);
        
        ret = wait_event_interruptible_timeout(
            sdev->wait_queue,
            (sdev->int_status & wait_int.mask) != 0,
            msecs_to_jiffies(wait_int.timeout_ms)
        );
        
        mutex_lock(&sdev->mutex);
        
        if (ret < 0)
            break;
        
        if (ret == 0) {
            ret = -ETIMEDOUT;
            break;
        }
        
        /* Return and clear interrupt status */
        wait_int.status = sdev->int_status;
        sdev->int_status &= ~wait_int.mask;
        
        if (copy_to_user(argp, &wait_int, sizeof(wait_int)))
            ret = -EFAULT;
        else
            ret = 0;
        break;
    }
    
    case SILENS_IOCTL_RESET: {
        /* Reset device */
        silens_write_reg(sdev, REG_CTRL, CTRL_RESET);
        msleep(100);
        silens_write_reg(sdev, REG_CTRL, 0);
        break;
    }
    
    default:
        ret = -ENOTTY;
        break;
    }
    
    mutex_unlock(&sdev->mutex);
    return ret;
}


static int silens_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct silens_device *sdev = filp->private_data;
    unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;
    unsigned long size = vma->vm_end - vma->vm_start;
    
    /* Offset 0: BAR0 registers */
    if (offset == 0) {
        if (size > sdev->bar0_len)
            return -EINVAL;
        
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        
        if (io_remap_pfn_range(vma, vma->vm_start,
                               sdev->bar0_start >> PAGE_SHIFT,
                               size, vma->vm_page_prot))
            return -EAGAIN;
        
        return 0;
    }
    
    /* Offset > 0: DMA buffer */
    {
        struct silens_dma_buffer *buf;
        int handle = (offset - PAGE_SIZE) / SILENS_MAX_DMA_SIZE;
        
        if (handle < 0 || handle >= SILENS_MAX_DMA_BUFFERS)
            return -EINVAL;
        
        buf = &sdev->dma_buffers[handle];
        
        if (!buf->in_use || size > buf->size)
            return -EINVAL;
        
        /* Map DMA buffer to userspace */
        return dma_mmap_coherent(&sdev->pdev->dev, vma,
                                 buf->cpu_addr, buf->dma_addr, buf->size);
    }
}

static unsigned int silens_poll(struct file *filp, poll_table *wait)
{
    struct silens_device *sdev = filp->private_data;
    unsigned int mask = 0;
    
    poll_wait(filp, &sdev->wait_queue, wait);
    
    if (sdev->int_status)
        mask |= POLLIN | POLLRDNORM;
    
    return mask;
}

/* ============================================================================
 * PCI Driver Callbacks
 * ============================================================================ */

static int silens_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct silens_device *sdev;
    int ret;
    
    dev_info(&pdev->dev, "SiLens device found: %04x:%04x\n",
             pdev->vendor, pdev->device);
    
    /* Allocate device structure */
    sdev = kzalloc(sizeof(*sdev), GFP_KERNEL);
    if (!sdev)
        return -ENOMEM;
    
    sdev->pdev = pdev;
    sdev->minor = silens_num_devices;
    mutex_init(&sdev->mutex);
    spin_lock_init(&sdev->dma_lock);
    init_waitqueue_head(&sdev->wait_queue);
    atomic_set(&sdev->open_count, 0);
    
    pci_set_drvdata(pdev, sdev);
    
    /* Enable device */
    ret = pci_enable_device(pdev);
    if (ret) {
        dev_err(&pdev->dev, "Failed to enable PCI device\n");
        goto err_free;
    }
    
    /* Request regions */
    ret = pci_request_regions(pdev, DRIVER_NAME);
    if (ret) {
        dev_err(&pdev->dev, "Failed to request PCI regions\n");
        goto err_disable;
    }

    
    /* Map BAR0 */
    sdev->bar0_start = pci_resource_start(pdev, SILENS_BAR_REGS);
    sdev->bar0_len = pci_resource_len(pdev, SILENS_BAR_REGS);
    
    sdev->bar0 = pci_iomap(pdev, SILENS_BAR_REGS, sdev->bar0_len);
    if (!sdev->bar0) {
        dev_err(&pdev->dev, "Failed to map BAR0\n");
        ret = -ENOMEM;
        goto err_regions;
    }
    
    /* Enable bus mastering for DMA */
    pci_set_master(pdev);
    
    /* Set DMA mask */
    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
    if (ret) {
        ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
        if (ret) {
            dev_err(&pdev->dev, "Failed to set DMA mask\n");
            goto err_unmap;
        }
    }
    
    /* Setup MSI interrupts */
    ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSI | PCI_IRQ_LEGACY);
    if (ret < 0) {
        dev_err(&pdev->dev, "Failed to allocate IRQ vectors\n");
        goto err_unmap;
    }
    
    sdev->irq = pci_irq_vector(pdev, 0);
    sdev->msi_enabled = (ret == 1 && pdev->msi_enabled);
    
    ret = request_irq(sdev->irq, silens_irq_handler,
                      sdev->msi_enabled ? 0 : IRQF_SHARED,
                      DRIVER_NAME, sdev);
    if (ret) {
        dev_err(&pdev->dev, "Failed to request IRQ %d\n", sdev->irq);
        goto err_irq_vec;
    }
    
    /* Create character device */
    cdev_init(&sdev->cdev, &silens_fops);
    sdev->cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&sdev->cdev, MKDEV(MAJOR(silens_devno), sdev->minor), 1);
    if (ret) {
        dev_err(&pdev->dev, "Failed to add cdev\n");
        goto err_irq;
    }
    
    /* Create device node */
    sdev->dev = device_create(silens_class, &pdev->dev,
                              MKDEV(MAJOR(silens_devno), sdev->minor),
                              sdev, "silens%d", sdev->minor);
    if (IS_ERR(sdev->dev)) {
        ret = PTR_ERR(sdev->dev);
        dev_err(&pdev->dev, "Failed to create device\n");
        goto err_cdev;
    }
    
    /* Read and display hardware version */
    {
        u32 version = silens_read_reg(sdev, REG_VERSION);
        dev_info(&pdev->dev, "Hardware version: %d.%d.%d\n",
                 (version >> 16) & 0xFF,
                 (version >> 8) & 0xFF,
                 version & 0xFF);
    }
    
    silens_num_devices++;
    
    dev_info(&pdev->dev, "SiLens device %d initialized successfully\n", sdev->minor);
    
    return 0;

err_cdev:
    cdev_del(&sdev->cdev);
err_irq:
    free_irq(sdev->irq, sdev);
err_irq_vec:
    pci_free_irq_vectors(pdev);
err_unmap:
    pci_iounmap(pdev, sdev->bar0);
err_regions:
    pci_release_regions(pdev);
err_disable:
    pci_disable_device(pdev);
err_free:
    kfree(sdev);
    return ret;
}


static void silens_remove(struct pci_dev *pdev)
{
    struct silens_device *sdev = pci_get_drvdata(pdev);
    int i;
    
    dev_info(&pdev->dev, "Removing SiLens device %d\n", sdev->minor);
    
    /* Destroy device node */
    device_destroy(silens_class, MKDEV(MAJOR(silens_devno), sdev->minor));
    
    /* Remove cdev */
    cdev_del(&sdev->cdev);
    
    /* Free all DMA buffers */
    for (i = 0; i < SILENS_MAX_DMA_BUFFERS; i++) {
        if (sdev->dma_buffers[i].in_use)
            silens_free_dma_buffer(sdev, i);
    }
    
    /* Free IRQ */
    free_irq(sdev->irq, sdev);
    pci_free_irq_vectors(pdev);
    
    /* Unmap BAR0 */
    pci_iounmap(pdev, sdev->bar0);
    
    /* Release regions */
    pci_release_regions(pdev);
    
    /* Disable device */
    pci_disable_device(pdev);
    
    kfree(sdev);
    silens_num_devices--;
}

static struct pci_driver silens_pci_driver = {
    .name     = DRIVER_NAME,
    .id_table = silens_pci_ids,
    .probe    = silens_probe,
    .remove   = silens_remove,
};

/* ============================================================================
 * Module Init/Exit
 * ============================================================================ */

static int __init silens_init(void)
{
    int ret;
    
    pr_info("SiLens driver v%s loading\n", DRIVER_VERSION);
    
    /* Allocate character device numbers */
    ret = alloc_chrdev_region(&silens_devno, 0, SILENS_MAX_DEVICES, DRIVER_NAME);
    if (ret) {
        pr_err("Failed to allocate chrdev region\n");
        return ret;
    }
    
    /* Create device class */
    silens_class = class_create(DRIVER_NAME);
    if (IS_ERR(silens_class)) {
        ret = PTR_ERR(silens_class);
        pr_err("Failed to create device class\n");
        goto err_chrdev;
    }
    
    /* Register PCI driver */
    ret = pci_register_driver(&silens_pci_driver);
    if (ret) {
        pr_err("Failed to register PCI driver\n");
        goto err_class;
    }
    
    pr_info("SiLens driver loaded successfully\n");
    
    return 0;

err_class:
    class_destroy(silens_class);
err_chrdev:
    unregister_chrdev_region(silens_devno, SILENS_MAX_DEVICES);
    return ret;
}

static void __exit silens_exit(void)
{
    pci_unregister_driver(&silens_pci_driver);
    class_destroy(silens_class);
    unregister_chrdev_region(silens_devno, SILENS_MAX_DEVICES);
    
    pr_info("SiLens driver unloaded\n");
}

module_init(silens_init);
module_exit(silens_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("SiLens Team");
MODULE_DESCRIPTION(DRIVER_DESC);
MODULE_VERSION(DRIVER_VERSION);
