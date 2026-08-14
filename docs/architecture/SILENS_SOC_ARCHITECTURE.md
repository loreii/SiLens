# SiLens SoC Architecture

> **Target:** SkyWater SKY130 130nm CMOS  
> **Die Size:** ~800mm² (26mm × 30.77mm, fits 26×32mm reticle)  
> **Model:** SmolVLM-256M (246M ternary parameters)  
> **Package:** FCBGA-900 (30mm × 30mm)

---

## 1. Executive Summary

The SiLens SoC is a full-custom ASIC designed to run SmolVLM-256M vision-language inference entirely on-chip, with weights hardwired as metal routing. This approach eliminates weight memory access and achieves high energy efficiency.

### Key Specifications

| Parameter | Value |
|-----------|-------|
| Process | SkyWater SKY130 (130nm) |
| Die Size | 800mm² (26mm × 30.77mm) |
| Core Voltage | 1.8V |
| I/O Voltage | 3.3V |
| Clock Frequency | 100 MHz (target) |
| Power Budget | 25W TDP |
| Package | FCBGA-900 |
| Model | SmolVLM-256M (246M params) |
| Quantization | Ternary (-1, 0, +1) |
| Weight Storage | 61.5MB hardwired |

---

## 2. System Block Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           SiLens SoC (800mm², SKY130)                          │
│                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │                     VISION ENCODER (SigLIP-B/16)                       │    │
│  │                          93M Ternary Weights                           │    │
│  │    ┌──────────┐   ┌──────────┐   ┌──────────┐       ┌──────────┐     │    │
│  │    │  Patch   │──▶│  Embed   │──▶│  Xformer │──...──│  Xformer │     │    │
│  │    │ Extract  │   │ + PosEmb │   │ Block 1  │       │ Block 12 │     │    │
│  │    └──────────┘   └──────────┘   └──────────┘       └──────────┘     │    │
│  │                         (~250mm² area)                                │    │
│  └─────────────────────────────────────┬─────────────────────────────────┘    │
│                                        │                                       │
│  ┌─────────────────────────────────────▼─────────────────────────────────┐    │
│  │                    MULTIMODAL PROJECTOR (18M weights)                  │    │
│  │                         768 → 576 dimension                            │    │
│  │                           (~50mm² area)                                │    │
│  └─────────────────────────────────────┬─────────────────────────────────┘    │
│                                        │                                       │
│  ┌─────────────────────────────────────▼─────────────────────────────────┐    │
│  │                   LANGUAGE MODEL (SmolLM2-135M)                        │    │
│  │                        135M Ternary Weights                            │    │
│  │    ┌──────────┐   ┌──────────┐   ┌──────────┐       ┌──────────┐     │    │
│  │    │  Token   │──▶│  Xformer │──▶│  Xformer │──...──│  Xformer │──▶  │    │
│  │    │  Embed   │   │ Block 1  │   │ Block 2  │       │ Block 30 │     │    │
│  │    └──────────┘   └──────────┘   └──────────┘       └──────────┘     │    │
│  │                                                            │          │    │
│  │    ┌──────────────────────────────────────────────────────▼────────┐ │    │
│  │    │                   LM HEAD (576 → 49152)                        │ │    │
│  │    │                    Token ID Output                             │ │    │
│  │    └───────────────────────────────────────────────────────────────┘ │    │
│  │                         (~400mm² area)                                │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   DDR3 PHY   │  │   Parallel   │  │    Clock     │  │    Power     │      │
│  │   (32-bit)   │  │   Host I/F   │  │  Generation  │  │  Management  │      │
│  │   (~30mm²)   │  │   (~20mm²)   │  │   (~10mm²)   │  │   (~10mm²)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  └──────────────┘      │
│         │                 │                                                    │
│  ┌──────┴─────────────────┴────────────────────────────────────────────┐      │
│  │                            IO RING (~30mm²)                          │      │
│  │   DDR3 (62 pins)  |  Host Parallel (83 pins)  |  GPIO/Debug (40)    │      │
│  └──────────────────────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────────────────────┘
                    │                                │
                    ▼                                ▼
            ┌──────────────┐                ┌──────────────┐
            │  DDR3 SDRAM  │                │  FPGA Bridge │
            │   256MB-1GB  │                │   (PCIe x4)  │
            │  (KV Cache)  │                │   to Host    │
            └──────────────┘                └──────────────┘
```

---

## 3. Module Hierarchy

```
silens_soc (top)
├── silens_clock_gen         # Clock generation (PLL)
├── silens_reset_sync (×3)   # Reset synchronizers
├── silens_ddr3_controller   # DDR3-1066 memory controller
├── silens_host_interface    # Parallel bus + FIFOs + CDC
├── silens_spi_slave         # SPI configuration interface
├── silens_vlm_core          # VLM processing core
│   ├── vision_encoder_top   # SigLIP-B/16 encoder
│   │   ├── patch_extractor
│   │   ├── position_embedding_rom
│   │   ├── ternary_matmul (×many)
│   │   ├── transformer_block (×12)
│   │   │   ├── rms_norm
│   │   │   ├── self_attention
│   │   │   └── mlp
│   │   └── layer_norm
│   ├── projector            # 768→576 MLP
│   └── llm_decoder_top      # SmolLM2-135M decoder
│       ├── embedding_table
│       ├── transformer_block (×30)
│       │   ├── rms_norm (×2)
│       │   ├── grouped_query_attention
│       │   │   ├── q_proj (ternary matmul)
│       │   │   ├── k_proj (ternary matmul)
│       │   │   ├── v_proj (ternary matmul)
│       │   │   ├── attention_scores
│       │   │   ├── softmax_approx
│       │   │   └── o_proj (ternary matmul)
│       │   └── gated_mlp
│       │       ├── gate_proj + silu
│       │       ├── up_proj
│       │       └── down_proj
│       ├── final_rms_norm
│       └── lm_head + argmax
└── gpio_controller
```

---

## 4. Area Breakdown

| Block | Area (mm²) | Percentage | Notes |
|-------|------------|------------|-------|
| Vision Encoder (SigLIP-B/16) | 250 | 31.3% | 12 transformer blocks, 93M params |
| Language Model (SmolLM2-135M) | 400 | 50.0% | 30 transformer blocks, 135M params |
| Multimodal Projector | 50 | 6.3% | 2-layer MLP, 18M params |
| DDR3 PHY + Controller | 30 | 3.8% | 32-bit interface |
| Host Interface | 20 | 2.5% | Parallel bus, FIFOs |
| Clock/Power Management | 20 | 2.5% | PLL, PMU, monitors |
| IO Ring | 30 | 3.8% | Pads, ESD, level shifters |
| **Total** | **800** | **100%** | |

---

## 5. Memory Architecture

### 5.1 On-Chip Memory

| Memory | Size | Type | Purpose |
|--------|------|------|---------|
| Weights | 61.5 MB | Hardwired (metal) | Model parameters |
| Vision Tokens | 3.5 MB | SRAM | 576 × 768 × 8-bit |
| Activation Buffer | 12 MB | SRAM | Layer intermediates |
| Embedding Table | 27 MB | ROM | 49152 × 576 × 8-bit |

### 5.2 External Memory (DDR3)

| Memory | Size | Purpose |
|--------|------|---------|
| KV Cache | ~70 MB | 30 layers × 2K seq × 9 heads × 64 dim × 2 (K+V) |
| Image Buffer | 0.5 MB | 384 × 384 × 3 bytes |
| **Required** | **256 MB** | Minimum DDR3 |
| **Recommended** | **512 MB-1GB** | Headroom for longer context |

---

## 6. Interface Specifications

### 6.1 DDR3-1066 Interface

| Parameter | Value |
|-----------|-------|
| Data Width | 32 bits |
| Clock | 533 MHz (DDR) |
| Bandwidth | 4.3 GB/s peak |
| Latency | ~50 ns typical |
| Protocol | DDR3 JEDEC |
| Pins | 62 |

### 6.2 Parallel Host Interface

| Parameter | Value |
|-----------|-------|
| Data Width | 32 bits |
| Address Width | 16 bits |
| Clock | 100 MHz |
| Bandwidth | 400 MB/s peak |
| Protocol | Async parallel bus |
| Pins | 83 |

**Register Map:**

| Address | Name | R/W | Description |
|---------|------|-----|-------------|
| 0x0000 | CONTROL | RW | Start/stop/abort control |
| 0x0004 | STATUS | RO | State, busy, error flags |
| 0x0008 | TOKEN_WR | WO | Input token FIFO write |
| 0x000C | TOKEN_RD | RO | Output token FIFO read |
| 0x0010 | DMA_BASE | RW | Image DMA base address |
| 0x0014 | DMA_LEN | RW | Image DMA length |
| 0x0018 | DMA_CTRL | RW | DMA control |
| 0x001C | IRQ_ENABLE | RW | Interrupt enable |
| 0x0020 | IRQ_STATUS | RO/W1C | Interrupt status |
| 0x0024 | VERSION | RO | Hardware version |
| 0x0028 | PIXEL_WR | WO | Pixel FIFO write |
| 0x002C | FIFO_STATUS | RO | FIFO full/empty flags |

### 6.3 SPI Configuration Interface

| Parameter | Value |
|-----------|-------|
| Mode | 0 (CPOL=0, CPHA=0) |
| Max Clock | 10 MHz |
| Data Order | MSB first |
| Protocol | 8-bit register R/W |
| Pins | 4 |

### 6.4 JTAG Debug Interface

| Signal | Direction |
|--------|-----------|
| TCK | Input |
| TMS | Input |
| TDI | Input |
| TDO | Output |
| TRST_N | Input |

---

## 7. Power Budget

| Domain | Voltage | Current | Power |
|--------|---------|---------|-------|
| Core Logic | 1.8V | 10A | 18W |
| I/O Ring | 3.3V | 1.5A | 5W |
| DDR3 PHY | 1.5V | 0.5A | 0.75W |
| Analog (PLL) | 1.8V | 0.1A | 0.18W |
| **Total** | | | **~24W TDP** |

### Power Density

- Die area: 800mm²
- Power: 24W
- **Power density: 3.0 W/cm²** (well below 10 W/cm² thermal limit)
- Passive heatsink sufficient

---

## 8. Clock Domains

| Domain | Frequency | Purpose |
|--------|-----------|---------|
| clk_core | 100 MHz | Main compute logic |
| clk_ddr | 533 MHz | DDR3 interface (2× data rate) |
| host_clk | 100 MHz | Host parallel interface |
| spi_clk | ≤10 MHz | SPI configuration (external) |
| jtag_tck | ≤20 MHz | JTAG debug (external) |

### Clock Domain Crossings

| From | To | Mechanism |
|------|-----|-----------|
| host_clk | clk_core | Async FIFOs (Gray coded) |
| clk_core | host_clk | Double-FF synchronizers |
| spi_clk | clk_core | Edge detection + sync |

---

## 9. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Vision encoding | 50ms | 384×384 image |
| Prefill latency | 20ms | 576 vision + 100 text tokens |
| Token throughput | 80-150 tok/s | Autoregressive decode |
| First token latency | 70ms | Vision + prefill |
| Power efficiency | 3-6 tok/s/W | At 24W TDP |

---

## 10. File Listing

### RTL Files

| File | Description |
|------|-------------|
| `rtl/top/silens_soc.v` | Top-level SoC |
| `rtl/core/silens_vlm_core.v` | VLM processing core |
| `rtl/vision/vision_encoder_top.v` | Vision encoder |
| `rtl/llm/llm_decoder_top.v` | LLM decoder |
| `rtl/memory/ddr3_controller.v` | DDR3 controller |
| `rtl/interfaces/silens_host_interface.v` | Host parallel interface |
| `rtl/interfaces/silens_spi_slave.v` | SPI slave |
| `rtl/common/clock_gen.v` | Clock generation |
| `rtl/common/reset_sync.v` | Reset synchronizer |
| `rtl/common/ternary_mac.v` | Ternary MAC unit |
| `rtl/common/rms_norm.v` | RMS normalization |
| `rtl/common/softmax_approx.v` | Softmax approximation |
| `rtl/llm/projector.v` | Multimodal projector |

### Synthesis Configuration

| File | Description |
|------|-------------|
| `openlane/silens_soc/config.json` | OpenLane config |
| `openlane/silens_soc/pin_order.cfg` | Pin placement |

---

## 11. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-08-14 | SiLens Team | Initial architecture |

---

*Document generated as part of SiLens SoC design flow*
