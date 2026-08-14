# SkyWater SKY130 Pin Mapping for SiLens

## Caravel Harness Overview

The SkyWater SKY130 PDK uses the **Caravel** harness which provides a standardized interface for user projects. This document defines how SiLens NanoLM maps to the Caravel I/O.

## Available Caravel Resources

| Resource | Count | Description |
|----------|-------|-------------|
| GPIOs | 38 | `io_in/io_out/io_oeb[37:0]` |
| Logic Analyzer | 128 bits | Debug/monitoring interface |
| Wishbone Bus | 32-bit | Register access from management SoC |
| IRQ Lines | 3 | Interrupt signals |
| User Clock | 1 | Independent clock input |

---

## GPIO Pin Mapping (38 pins)

### Reserved Pins (Directly from Caravel - DO NOT USE)

| Pin | Signal | Direction | Notes |
|-----|--------|-----------|-------|
| `io[0]` | JTAG | - | Reserved for debug |
| `io[1]` | SDO | - | Reserved |
| `io[2]` | SDI | - | Reserved |
| `io[3]` | CSB | - | Reserved |
| `io[4]` | SCK | - | Reserved |
| `io[5]` | ser_rx | - | Management UART |
| `io[6]` | ser_tx | - | Management UART |
| `io[7]` | IRQ | - | Reserved |

### SPI Interface (Data Streaming)

| Pin | Signal | Direction | Description |
|-----|--------|-----------|-------------|
| `io[8]` | `spi_clk` | Input | SPI clock (up to 25 MHz) |
| `io[9]` | `spi_mosi` | Input | SPI data in (tokens, pixels) |
| `io[10]` | `spi_miso` | Output | SPI data out (generated tokens) |
| `io[11]` | `spi_cs_n` | Input | Chip select (active low) |

### UART Interface (Alternative I/O)

| Pin | Signal | Direction | Description |
|-----|--------|-----------|-------------|
| `io[12]` | `uart_rx` | Input | UART receive (115200 baud default) |
| `io[13]` | `uart_tx` | Output | UART transmit |

### Control Signals

| Pin | Signal | Direction | Description |
|-----|--------|-----------|-------------|
| `io[14]` | `frame_start` | Input | Trigger image inference (rising edge) |
| `io[15]` | `seq_start` | Input | Start new text sequence |
| `io[16]` | `gen_start` | Input | Start autoregressive generation |
| `io[17]` | `inference_done` | Output | High when inference complete |
| `io[18]` | `busy` | Output | High during processing |
| `io[19]` | `error_flag` | Output | High on error |

### Status LEDs

| Pin | Signal | Direction | Description |
|-----|--------|-----------|-------------|
| `io[20]` | `status[0]` | Output | FSM state bit 0 |
| `io[21]` | `status[1]` | Output | FSM state bit 1 |
| `io[22]` | `status[2]` | Output | FSM state bit 2 |
| `io[23]` | `status[3]` | Output | FSM state bit 3 |

### Token Output (Parallel Interface)

| Pin | Signal | Direction | Description |
|-----|--------|-----------|-------------|
| `io[24]` | `token_out[0]` | Output | Token bit 0 (LSB) |
| `io[25]` | `token_out[1]` | Output | Token bit 1 |
| `io[26]` | `token_out[2]` | Output | Token bit 2 |
| `io[27]` | `token_out[3]` | Output | Token bit 3 |
| `io[28]` | `token_out[4]` | Output | Token bit 4 |
| `io[29]` | `token_out[5]` | Output | Token bit 5 |
| `io[30]` | `token_out[6]` | Output | Token bit 6 |
| `io[31]` | `token_out[7]` | Output | Token bit 7 |
| `io[32]` | `token_out[8]` | Output | Token bit 8 |
| `io[33]` | `token_out[9]` | Output | Token bit 9 |
| `io[34]` | `token_out[10]` | Output | Token bit 10 |
| `io[35]` | `token_out[11]` | Output | Token bit 11 (MSB, 4K vocab) |
| `io[36]` | `token_valid` | Output | Token strobe (high = valid token) |
| `io[37]` | `token_ready` | Input | Consumer ready for next token |

---

## Wishbone Register Map

Base address: `0x3000_0000`

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| `0x00` | `CTRL` | R/W | Control register |
| `0x04` | `STATUS` | RO | Status register |
| `0x08` | `TOKEN_IN` | WO | Write token to input buffer |
| `0x0C` | `TOKEN_OUT` | RO | Read generated token |
| `0x10` | `CONFIG` | R/W | Configuration register |
| `0x100` | `WEIGHT_ADDR` | WO | Weight load address |
| `0x104` | `WEIGHT_DATA` | WO | Weight load data |

### Control Register (0x00)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `frame_start` | Trigger image inference |
| 1 | `seq_start` | Start text sequence |
| 2 | `gen_start` | Start generation |
| 3 | `soft_reset` | Reset inference engine |
| 31:4 | Reserved | - |

### Status Register (0x04)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `token_valid` | Output token available |
| 1 | `inference_done` | Inference complete |
| 2 | `busy` | Processing in progress |
| 3 | `error_flag` | Error occurred |
| 7:4 | `fsm_state` | Current FSM state |
| 31:8 | Reserved | - |

---

## Logic Analyzer Mapping (128 bits)

| Bits | Signal | Description |
|------|--------|-------------|
| `[31:0]` | `cycle_counter` | Clock cycle count |
| `[63:32]` | `inference_cycles` | Cycles for last inference |
| `[67:64]` | `fsm_state` | Internal FSM state |
| `[79:68]` | `current_token` | Current token being processed |
| `[95:80]` | `token_count` | Tokens generated count |
| `[127:96]` | Reserved | Custom debug signals |

---

## IRQ Mapping

| IRQ | Signal | Description |
|-----|--------|-------------|
| `irq[0]` | `inference_done` | Inference complete interrupt |
| `irq[1]` | `error_flag` | Error interrupt |
| `irq[2]` | `token_valid` | Token ready interrupt |

---

## Clock Configuration

| Clock | Source | Typical Frequency | Notes |
|-------|--------|-------------------|-------|
| `wb_clk_i` | Caravel | 10-40 MHz | Wishbone/system clock |
| `user_clock2` | External | Up to 100 MHz | Optional high-speed clock |

The design defaults to using `wb_clk_i` for simplicity. For higher performance, `user_clock2` can be configured.

---

## Physical Connection Example

### Minimal Test Setup

```
Raspberry Pi / MCU            SiLens (Caravel)
─────────────────            ─────────────────
GPIO (SPI CLK)  ───────────► io[8]  (spi_clk)
GPIO (MOSI)     ───────────► io[9]  (spi_mosi)
GPIO (MISO)     ◄─────────── io[10] (spi_miso)
GPIO (CS)       ───────────► io[11] (spi_cs_n)
GPIO (START)    ───────────► io[14] (frame_start)
GPIO (DONE)     ◄─────────── io[17] (inference_done)
GPIO (BUSY)     ◄─────────── io[18] (busy)
GND             ───────────── vssd1
3.3V            ───────────── vccd1
```

### Full Token Output Setup

```
MCU / FPGA                    SiLens (Caravel)
───────────                   ─────────────────
12-bit parallel bus ◄──────── io[35:24] (token_out[11:0])
Token strobe        ◄──────── io[36] (token_valid)
Ready signal        ────────► io[37] (token_ready)
```

---

## Data Transfer Protocol

### SPI Token Input (via io[8:11])

1. Assert `spi_cs_n` low
2. Send command byte: `0x01` = write token
3. Send 2 bytes: token ID (12-bit, MSB first)
4. Deassert `spi_cs_n`

### SPI Token Output (via io[8:11])

1. Assert `spi_cs_n` low
2. Send command byte: `0x02` = read token
3. Read 2 bytes: token ID (12-bit, MSB first)
4. Deassert `spi_cs_n`

### Parallel Token Output (via io[24:37])

1. Wait for `token_valid` (io[36]) high
2. Read `token_out[11:0]` from io[35:24]
3. Assert `token_ready` (io[37]) high
4. Wait for `token_valid` to go low
5. Deassert `token_ready`

---

## Power Requirements

| Supply | Voltage | Typical Current | Notes |
|--------|---------|-----------------|-------|
| `vccd1` | 1.8V | TBD (~50-100mA) | Digital core |
| `vssd1` | GND | - | Digital ground |

---

## Design Constraints

1. **38 GPIOs max** - Cannot exceed Caravel's GPIO count
2. **~2.9mm × 3.5mm user area** - Die area constraint
3. **SKY130 standard cells** - Must use sky130_fd_sc_hd library
4. **No external SRAM** - All memory on-die
5. **Max ~10 mm²** - Practical die size for NanoLM-5M

---

## Files

| File | Description |
|------|-------------|
| `user_project_wrapper.v` | Top-level Caravel wrapper |
| `silens_nanolm_core.v` | NanoLM inference core (to be created) |
| `config.json` | Caravel configuration |
| `pin_order.cfg` | Pin placement constraints |

---

*Last updated: Based on Caravel harness v2, SkyWater SKY130 PDK*
