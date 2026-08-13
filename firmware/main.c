/*
 * SiLens Card Firmware
 *
 * Firmware for SiLens FPGA prototype / ASIC card.
 * Handles PCIe configuration, command processing, and hardware control.
 *
 * Target: Soft-core processor (RISC-V or similar) on FPGA
 *         or dedicated control processor on ASIC
 *
 * Copyright (C) 2024 SiLens Team
 * License: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* ============================================================================
 * Hardware Register Definitions
 * ============================================================================ */

/* Base addresses for memory-mapped peripherals */
#define PCIE_BASE       0x10000000
#define VISION_BASE     0x20000000
#define LLM_BASE        0x30000000
#define DMA_BASE        0x40000000
#define UART_BASE       0x50000000

/* PCIe registers */
#define PCIE_CTRL       (*(volatile uint32_t *)(PCIE_BASE + 0x000))
#define PCIE_STATUS     (*(volatile uint32_t *)(PCIE_BASE + 0x004))
#define PCIE_INT_STATUS (*(volatile uint32_t *)(PCIE_BASE + 0x008))
#define PCIE_INT_ENABLE (*(volatile uint32_t *)(PCIE_BASE + 0x00C))
#define PCIE_BAR0_ADDR  (*(volatile uint32_t *)(PCIE_BASE + 0x010))
#define PCIE_BAR1_ADDR  (*(volatile uint32_t *)(PCIE_BASE + 0x014))

/* Host-visible registers (BAR0) */
typedef struct {
    volatile uint32_t ctrl;         /* 0x000 */
    volatile uint32_t status;       /* 0x004 */
    volatile uint32_t img_addr;     /* 0x008 */
    volatile uint32_t img_size;     /* 0x00C */
    volatile uint32_t out_addr;     /* 0x010 */
    volatile uint32_t out_len;      /* 0x014 */
    volatile uint32_t token_out;    /* 0x018 */
    volatile uint32_t token_valid;  /* 0x01C */
    volatile uint32_t kv_cache_addr;/* 0x020 */
    volatile uint32_t kv_cache_size;/* 0x024 */
    volatile uint32_t int_status;   /* 0x028 */
    volatile uint32_t int_enable;   /* 0x02C */
    uint32_t reserved[52];          /* 0x030-0x0FC */
    volatile uint32_t dma_ctrl;     /* 0x100 */
    volatile uint32_t dma_status;   /* 0x104 */
    volatile uint32_t dma_src_addr; /* 0x108 */
    volatile uint32_t dma_dst_addr; /* 0x10C */
    volatile uint32_t dma_length;   /* 0x110 */
    uint32_t reserved2[55];         /* 0x114-0x1EC */
    volatile uint32_t version;      /* 0x1F0 */
    volatile uint32_t build_date;   /* 0x1F4 */
} host_regs_t;

#define HOST_REGS ((host_regs_t *)0x00001000)


/* Vision encoder control */
typedef struct {
    volatile uint32_t ctrl;
    volatile uint32_t status;
    volatile uint32_t img_ptr;
    volatile uint32_t out_ptr;
    volatile uint32_t config[4];
} vision_ctrl_t;

#define VISION_CTRL ((vision_ctrl_t *)VISION_BASE)

/* LLM control */
typedef struct {
    volatile uint32_t ctrl;
    volatile uint32_t status;
    volatile uint32_t token_in;
    volatile uint32_t token_out;
    volatile uint32_t kv_cache_ptr;
    volatile uint32_t seq_len;
    volatile uint32_t config[4];
} llm_ctrl_t;

#define LLM_CTRL ((llm_ctrl_t *)LLM_BASE)

/* DMA engine */
typedef struct {
    volatile uint32_t ctrl;
    volatile uint32_t status;
    volatile uint32_t src_addr;
    volatile uint32_t dst_addr;
    volatile uint32_t length;
} dma_ctrl_t;

#define DMA_CTRL ((dma_ctrl_t *)DMA_BASE)

/* Control bits */
#define CTRL_ENABLE         (1 << 0)
#define CTRL_START          (1 << 1)
#define CTRL_RESET          (1 << 2)
#define CTRL_STREAMING      (1 << 5)

/* Status bits */
#define STATUS_READY        (1 << 0)
#define STATUS_BUSY         (1 << 1)
#define STATUS_ERROR        (1 << 2)
#define STATUS_VISION_DONE  (1 << 3)
#define STATUS_LLM_DONE     (1 << 4)
#define STATUS_TOKEN_READY  (1 << 5)
#define STATUS_DMA_ACTIVE   (1 << 6)
#define STATUS_INIT_DONE    (1 << 7)

/* Interrupt bits */
#define INT_INFERENCE_DONE  (1 << 0)
#define INT_TOKEN_READY     (1 << 1)
#define INT_ERROR           (1 << 2)
#define INT_DMA_DONE        (1 << 3)

/* ============================================================================
 * Global State
 * ============================================================================ */

typedef enum {
    STATE_IDLE = 0,
    STATE_LOAD_IMAGE,
    STATE_VISION,
    STATE_PROJECT,
    STATE_LLM,
    STATE_OUTPUT,
    STATE_ERROR
} fw_state_t;

static fw_state_t current_state = STATE_IDLE;
static uint32_t tokens_generated = 0;
static uint32_t max_tokens = 256;
static bool streaming_mode = false;


/* ============================================================================
 * Utility Functions
 * ============================================================================ */

static void delay_us(uint32_t us)
{
    /* Simple delay loop - calibrate for actual clock speed */
    volatile uint32_t count = us * 10;
    while (count--);
}

static void send_interrupt(uint32_t bits)
{
    HOST_REGS->int_status |= bits;
    
    /* Trigger MSI if enabled */
    if (HOST_REGS->int_enable & bits) {
        PCIE_INT_STATUS = bits;
    }
}

/* ============================================================================
 * Hardware Initialization
 * ============================================================================ */

static void init_pcie(void)
{
    /* Wait for PCIe link to be up */
    while (!(PCIE_STATUS & 0x01)) {
        delay_us(1000);
    }
    
    /* Enable BAR0 and BAR1 */
    PCIE_CTRL = 0x03;
    
    /* Enable interrupts */
    PCIE_INT_ENABLE = 0xFF;
}

static void init_vision_encoder(void)
{
    /* Reset vision encoder */
    VISION_CTRL->ctrl = CTRL_RESET;
    delay_us(100);
    VISION_CTRL->ctrl = 0;
    
    /* Configure for 384x384 input, 576 output tokens */
    VISION_CTRL->config[0] = 384;       /* Image size */
    VISION_CTRL->config[1] = 16;        /* Patch size */
    VISION_CTRL->config[2] = 768;       /* Hidden dim */
    VISION_CTRL->config[3] = 12;        /* Num layers */
    
    /* Enable */
    VISION_CTRL->ctrl = CTRL_ENABLE;
}

static void init_llm(void)
{
    /* Reset LLM */
    LLM_CTRL->ctrl = CTRL_RESET;
    delay_us(100);
    LLM_CTRL->ctrl = 0;
    
    /* Configure for SmolLM2-135M */
    LLM_CTRL->config[0] = 576;          /* Hidden dim */
    LLM_CTRL->config[1] = 30;           /* Num layers */
    LLM_CTRL->config[2] = 49152;        /* Vocab size */
    LLM_CTRL->config[3] = 8192;         /* Max seq len */
    
    /* Enable */
    LLM_CTRL->ctrl = CTRL_ENABLE;
}

static void init_dma(void)
{
    DMA_CTRL->ctrl = CTRL_RESET;
    delay_us(10);
    DMA_CTRL->ctrl = CTRL_ENABLE;
}

static void hardware_init(void)
{
    /* Initialize all subsystems */
    init_pcie();
    init_vision_encoder();
    init_llm();
    init_dma();
    
    /* Set version info */
    HOST_REGS->version = 0x00010000;    /* v0.1.0 */
    HOST_REGS->build_date = 0x20240101;
    
    /* Mark as ready */
    HOST_REGS->status = STATUS_READY | STATUS_INIT_DONE;
}


/* ============================================================================
 * DMA Operations
 * ============================================================================ */

static bool dma_transfer(uint32_t src, uint32_t dst, uint32_t len)
{
    /* Configure DMA */
    DMA_CTRL->src_addr = src;
    DMA_CTRL->dst_addr = dst;
    DMA_CTRL->length = len;
    
    /* Start transfer */
    DMA_CTRL->ctrl = CTRL_ENABLE | CTRL_START;
    
    /* Wait for completion */
    uint32_t timeout = 10000;
    while ((DMA_CTRL->status & STATUS_BUSY) && timeout--) {
        delay_us(1);
    }
    
    if (timeout == 0) {
        HOST_REGS->status |= STATUS_ERROR;
        return false;
    }
    
    return !(DMA_CTRL->status & STATUS_ERROR);
}

/* ============================================================================
 * Inference Pipeline
 * ============================================================================ */

static void run_vision_encoder(void)
{
    /* Set input pointer to image buffer */
    VISION_CTRL->img_ptr = HOST_REGS->img_addr;
    
    /* Start vision processing */
    VISION_CTRL->ctrl = CTRL_ENABLE | CTRL_START;
    
    /* Wait for completion */
    while (VISION_CTRL->status & STATUS_BUSY) {
        delay_us(10);
    }
    
    if (VISION_CTRL->status & STATUS_ERROR) {
        current_state = STATE_ERROR;
        HOST_REGS->status |= STATUS_ERROR;
        send_interrupt(INT_ERROR);
        return;
    }
    
    HOST_REGS->status |= STATUS_VISION_DONE;
    current_state = STATE_PROJECT;
}

static void run_projector(void)
{
    /* Projector is hardwired - just set up the data flow */
    /* Vision output feeds directly to projector, then to LLM */
    
    current_state = STATE_LLM;
}

static void run_llm_step(void)
{
    /* Start one token generation step */
    LLM_CTRL->ctrl = CTRL_ENABLE | CTRL_START;
    
    /* Wait for token */
    while (LLM_CTRL->status & STATUS_BUSY) {
        delay_us(1);
    }
    
    if (LLM_CTRL->status & STATUS_ERROR) {
        current_state = STATE_ERROR;
        HOST_REGS->status |= STATUS_ERROR;
        send_interrupt(INT_ERROR);
        return;
    }
    
    /* Get generated token */
    uint32_t token = LLM_CTRL->token_out;
    HOST_REGS->token_out = token;
    HOST_REGS->token_valid = 1;
    HOST_REGS->status |= STATUS_TOKEN_READY;
    
    tokens_generated++;
    
    /* Send interrupt for streaming mode */
    if (streaming_mode) {
        send_interrupt(INT_TOKEN_READY);
    }
    
    /* Check for end conditions */
    if (token == 2 /* EOS */ || tokens_generated >= max_tokens) {
        current_state = STATE_OUTPUT;
        HOST_REGS->status |= STATUS_LLM_DONE;
    }
    
    /* Feed token back for next step */
    LLM_CTRL->token_in = token;
    LLM_CTRL->seq_len++;
}


static void complete_inference(void)
{
    /* Store output length */
    HOST_REGS->out_len = tokens_generated;
    
    /* Clear busy, set done */
    HOST_REGS->status &= ~STATUS_BUSY;
    HOST_REGS->status |= STATUS_READY;
    
    /* Send completion interrupt */
    send_interrupt(INT_INFERENCE_DONE);
    
    current_state = STATE_IDLE;
}

/* ============================================================================
 * Command Processing
 * ============================================================================ */

static void handle_host_command(void)
{
    uint32_t ctrl = HOST_REGS->ctrl;
    
    /* Reset command */
    if (ctrl & CTRL_RESET) {
        HOST_REGS->ctrl = 0;
        current_state = STATE_IDLE;
        tokens_generated = 0;
        HOST_REGS->status = STATUS_READY | STATUS_INIT_DONE;
        return;
    }
    
    /* Start inference command */
    if ((ctrl & (CTRL_ENABLE | CTRL_START)) == (CTRL_ENABLE | CTRL_START)) {
        if (current_state != STATE_IDLE) {
            return;  /* Already running */
        }
        
        /* Parse configuration */
        streaming_mode = (ctrl & CTRL_STREAMING) != 0;
        max_tokens = 256;  /* Could be configurable */
        
        /* Start inference */
        HOST_REGS->status = STATUS_BUSY;
        HOST_REGS->token_valid = 0;
        tokens_generated = 0;
        LLM_CTRL->seq_len = 0;
        
        current_state = STATE_VISION;
    }
    
    /* Handle DMA commands */
    if (HOST_REGS->dma_ctrl & CTRL_START) {
        uint32_t direction = (HOST_REGS->dma_ctrl >> 4) & 0x03;
        
        if (direction == 0x01) {
            /* Host to device */
            dma_transfer(HOST_REGS->dma_src_addr, 
                        HOST_REGS->dma_dst_addr,
                        HOST_REGS->dma_length);
        } else if (direction == 0x02) {
            /* Device to host */
            dma_transfer(HOST_REGS->dma_src_addr,
                        HOST_REGS->dma_dst_addr,
                        HOST_REGS->dma_length);
        }
        
        HOST_REGS->dma_status = 0x01;  /* Complete */
        HOST_REGS->dma_ctrl &= ~CTRL_START;
        send_interrupt(INT_DMA_DONE);
    }
}

/* ============================================================================
 * Main Loop
 * ============================================================================ */

static void state_machine_step(void)
{
    switch (current_state) {
        case STATE_IDLE:
            /* Waiting for commands */
            break;
            
        case STATE_VISION:
            run_vision_encoder();
            break;
            
        case STATE_PROJECT:
            run_projector();
            break;
            
        case STATE_LLM:
            run_llm_step();
            break;
            
        case STATE_OUTPUT:
            complete_inference();
            break;
            
        case STATE_ERROR:
            /* Stay in error until reset */
            break;
            
        default:
            current_state = STATE_IDLE;
            break;
    }
}

int main(void)
{
    /* Initialize hardware */
    hardware_init();
    
    /* Main loop */
    while (1) {
        /* Process host commands */
        handle_host_command();
        
        /* Run state machine */
        state_machine_step();
    }
    
    return 0;
}
