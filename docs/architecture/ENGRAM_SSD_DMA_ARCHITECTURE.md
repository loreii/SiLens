# Engram N-gram Table: SSD-Resident DMA Architecture

## Executive Summary

The Qwen3.8-Flash-Next Engram table contains 51.2 billion parameters (320M rows × 160 dimensions) requiring ~102GB in FP16. This document proposes keeping the Engram table on NVMe SSD with hardware-accelerated hashing and DMA access, enabling inference on systems with limited DRAM while maintaining acceptable latency.

## Problem Statement

### Current Engram Access Pattern

```
Token sequence: [t₁, t₂, t₃, t₄, ...]

For each position i:
  1. Compute 8 bigram hashes:   hash(tᵢ₋₁, tᵢ) → row_ids[0:8]
  2. Compute 8 trigram hashes:  hash(tᵢ₋₂, tᵢ₋₁, tᵢ) → row_ids[8:16]
  3. Lookup 16 rows from Engram table
  4. Sum/aggregate embeddings (160 dims each)
  5. Add to hidden state before layer 2
```

### Key Observations

1. **Access is hash-indexed**: Not sequential, random access pattern
2. **Batch locality**: Nearby tokens share some n-grams (sliding window)
3. **Read-only during inference**: Table is frozen, no writes needed
4. **Latency budget**: ~100-500μs per token generation is acceptable

## Hardware Hashing Options

### Option 1: Intel Data Streaming Accelerator (DSA)

Available on Intel Xeon Scalable 4th Gen+ (Sapphire Rapids, Emerald Rapids).

```c
// DSA descriptor for CRC32-based hash computation
struct dsa_hw_desc {
    uint32_t flags;           // IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR
    uint32_t opcode;          // DSA_OPCODE_CRC64
    uint64_t src_addr;        // Token pair buffer
    uint64_t dst_addr;        // Hash result buffer
    uint32_t xfer_size;       // sizeof(token_pair)
    uint32_t crc_seed;        // Engram hash seed
};
```

**Performance characteristics:**
- Hash throughput: ~50 GB/s aggregate across all DSA engines
- Latency per hash: ~200-500ns
- Can chain: hash → DMA read in single descriptor batch

### Option 2: AMD EPYC I/O Acceleration

AMD EPYC 9004 series includes:
- **DRAM-less DMA**: Direct PCIe-to-NVMe transfers
- **Inline CRC**: Hardware CRC during DMA

```c
// AMD IOMMU DMA descriptor with inline CRC
struct amd_dma_desc {
    uint64_t src_phys;        // NVMe CMB address
    uint64_t dst_phys;        // GPU BAR or host memory
    uint32_t length;
    uint32_t flags;           // AMD_DMA_INLINE_CRC
    uint64_t crc_result;      // Written by hardware
};
```

### Option 3: FPGA-Based Hash Accelerator

For custom deployment (e.g., SiLens ASIC integration):

```
┌─────────────────────────────────────────────────────────┐
│                  FPGA Hash Accelerator                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Hash Unit 0 │  │ Hash Unit 1 │  │ ...             │  │
│  │ MurmurHash3 │  │ MurmurHash3 │  │ (16 parallel)   │  │
│  │ xxHash64    │  │ xxHash64    │  │                 │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                   │           │
│         └────────────────┼───────────────────┘           │
│                          ▼                               │
│              ┌───────────────────────┐                  │
│              │  Row Address Compute  │                  │
│              │  row = hash % 320M    │                  │
│              └───────────┬───────────┘                  │
│                          ▼                               │
│              ┌───────────────────────┐                  │
│              │  NVMe Command Queue   │                  │
│              │  (Batch read requests)│                  │
│              └───────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

## NVMe SSD Architecture for Engram

### Storage Layout

The Engram table is sharded across multiple NVMe SSDs for parallel access:

```
Engram Table: 320,001,536 rows × 160 dimensions × 2 bytes = 102.4 GB

Sharding Strategy (4 SSDs):
  SSD 0: rows 0 - 79,999,999          (25.6 GB)
  SSD 1: rows 80,000,000 - 159,999,999 (25.6 GB)
  SSD 2: rows 160,000,000 - 239,999,999 (25.6 GB)
  SSD 3: rows 240,000,000 - 320,001,535 (25.6 GB)

Row size: 160 × 2 bytes = 320 bytes
Aligned to: 512 bytes (NVMe sector) → 512 bytes/row with padding
```

### NVMe Access Optimization

```c
// Optimal NVMe access pattern
struct engram_nvme_config {
    // Use NVMe CMB (Controller Memory Buffer) for low-latency
    bool use_cmb;                    // Maps SSD internal DRAM to PCIe BAR
    
    // Queue configuration
    uint32_t io_queue_depth;         // 1024 (maximum parallel reads)
    uint32_t num_io_queues;          // 32 (one per CPU core)
    
    // Read coalescing
    uint32_t prefetch_window;        // 16 rows (5KB read for each hash batch)
    bool use_read_ahead;             // Enable for sequential token processing
    
    // DMA configuration  
    bool use_p2p_dma;                // Direct SSD→GPU without CPU bounce
    uint32_t dma_batch_size;         // 64 (rows per DMA batch)
};
```

### P2P DMA: SSD → GPU Direct Transfer

Using GPUDirect Storage (GDS) or SPDK with P2P:

```
                    No CPU involvement
                           │
┌──────────┐    PCIe P2P   │   ┌──────────┐
│  NVMe    │◄──────────────┼──►│   GPU    │
│  SSD     │               │   │  Memory  │
│          │               │   │          │
│ Engram   │  Direct DMA   │   │ Engram   │
│ Shard    │──────────────►│   │ Buffer   │
└──────────┘               │   └──────────┘
                           │
                    ~2-5μs latency
```

## Software Architecture

### Engram Manager Component

```c
// engram_ssd_manager.h

#ifndef ENGRAM_SSD_MANAGER_H
#define ENGRAM_SSD_MANAGER_H

#include <stdint.h>
#include <stdbool.h>

// Configuration
#define ENGRAM_NUM_ROWS      320001536
#define ENGRAM_ROW_DIM       160
#define ENGRAM_ROW_BYTES     320        // 160 * sizeof(fp16)
#define ENGRAM_ROW_ALIGNED   512        // NVMe sector alignment
#define ENGRAM_NUM_SHARDS    4
#define ENGRAM_BIGRAM_HASHES  8
#define ENGRAM_TRIGRAM_HASHES 8
#define ENGRAM_TOTAL_HASHES  16

// Hash seeds (from Qwen3.8-Flash-Next reference)
#define ENGRAM_HASH_SEED_BIGRAM   0x9E3779B97F4A7C15ULL
#define ENGRAM_HASH_SEED_TRIGRAM  0xC6A4A7935BD1E995ULL

// Shard information
typedef struct {
    int nvme_fd;                    // NVMe device file descriptor
    uint64_t base_row;              // First row in this shard
    uint64_t num_rows;              // Number of rows in shard
    void* cmb_mapping;              // CMB memory mapping (if available)
    uint64_t cmb_size;
} engram_shard_t;

// Manager state
typedef struct {
    engram_shard_t shards[ENGRAM_NUM_SHARDS];
    
    // Hardware acceleration
    bool use_dsa;                   // Intel DSA available
    int dsa_wq_fd;                  // DSA work queue file descriptor
    
    // Caching
    void* hot_cache;                // LRU cache for frequent rows
    uint64_t cache_size;            // Cache size in bytes
    
    // DMA configuration
    bool use_p2p;                   // P2P DMA to GPU
    int gpu_device_id;
    void* gpu_staging_buffer;       // GPU memory for Engram rows
} engram_manager_t;

// Initialization
int engram_manager_init(engram_manager_t* mgr, const char* config_path);
void engram_manager_destroy(engram_manager_t* mgr);

// Hash computation
void engram_compute_hashes(
    const uint32_t* tokens,         // Input token IDs
    int num_tokens,                 // Number of tokens
    uint64_t* row_ids,              // Output: [num_tokens][16] row IDs
    engram_manager_t* mgr           // Uses DSA if available
);

// Row lookup (async)
typedef void (*engram_callback_t)(void* user_data, const void* rows, int num_rows);

int engram_lookup_async(
    engram_manager_t* mgr,
    const uint64_t* row_ids,        // Row IDs to fetch
    int num_rows,                   // Number of rows
    void* dest_buffer,              // Destination (host or GPU memory)
    engram_callback_t callback,     // Completion callback
    void* user_data
);

// Synchronous lookup (blocks until complete)
int engram_lookup_sync(
    engram_manager_t* mgr,
    const uint64_t* row_ids,
    int num_rows,
    void* dest_buffer
);

// Batch processing for token sequence
int engram_process_sequence(
    engram_manager_t* mgr,
    const uint32_t* tokens,
    int seq_len,
    void* output_embeddings,        // [seq_len][160] fp16 output
    bool use_gpu_direct             // Use P2P DMA to GPU
);

#endif // ENGRAM_SSD_MANAGER_H
```

### Intel DSA Integration

```c
// engram_dsa_hash.c - Intel DSA hardware hashing

#include <linux/idxd.h>
#include <accel-config/libaccel_config.h>
#include "engram_ssd_manager.h"

// MurmurHash3 compatible with Engram reference implementation
static inline uint64_t engram_hash_bigram(uint32_t t1, uint32_t t2, uint64_t seed) {
    uint64_t h = seed;
    uint64_t k = ((uint64_t)t1 << 32) | t2;
    
    // MurmurHash3 64-bit finalizer
    k *= 0x87c37b91114253d5ULL;
    k = (k << 31) | (k >> 33);
    k *= 0x4cf5ad432745937fULL;
    h ^= k;
    h = (h << 27) | (h >> 37);
    h = h * 5 + 0x52dce729;
    
    // Finalize
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    
    return h % ENGRAM_NUM_ROWS;
}

// DSA-accelerated batch hashing
int engram_dsa_hash_batch(
    engram_manager_t* mgr,
    const uint32_t* tokens,
    int num_tokens,
    uint64_t* row_ids
) {
    if (!mgr->use_dsa) {
        // Fallback to software
        for (int i = 1; i < num_tokens; i++) {
            for (int j = 0; j < ENGRAM_BIGRAM_HASHES; j++) {
                uint64_t seed = ENGRAM_HASH_SEED_BIGRAM + j;
                row_ids[i * ENGRAM_TOTAL_HASHES + j] = 
                    engram_hash_bigram(tokens[i-1], tokens[i], seed);
            }
            for (int j = 0; j < ENGRAM_TRIGRAM_HASHES && i >= 2; j++) {
                uint64_t seed = ENGRAM_HASH_SEED_TRIGRAM + j;
                // Trigram: combine three tokens
                uint64_t h = engram_hash_bigram(tokens[i-2], tokens[i-1], seed);
                row_ids[i * ENGRAM_TOTAL_HASHES + ENGRAM_BIGRAM_HASHES + j] = 
                    engram_hash_bigram(h, tokens[i], seed);
            }
        }
        return 0;
    }
    
    // DSA batch submission
    struct dsa_hw_desc* descs = aligned_alloc(64, 
        num_tokens * ENGRAM_TOTAL_HASHES * sizeof(struct dsa_hw_desc));
    
    // Build descriptors for CRC64 operations
    // DSA CRC64 can be used as a fast hash function
    int desc_idx = 0;
    for (int i = 1; i < num_tokens; i++) {
        uint64_t token_pair[2] = {tokens[i-1], tokens[i]};
        
        for (int j = 0; j < ENGRAM_BIGRAM_HASHES; j++) {
            struct dsa_hw_desc* d = &descs[desc_idx++];
            d->opcode = DSA_OPCODE_CRC64;
            d->flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR;
            d->src_addr = (uint64_t)token_pair;
            d->dst_addr = (uint64_t)&row_ids[i * ENGRAM_TOTAL_HASHES + j];
            d->xfer_size = 16;
            d->crc_seed = ENGRAM_HASH_SEED_BIGRAM + j;
        }
    }
    
    // Submit batch to DSA
    struct dsa_batch_desc batch = {
        .desc_count = desc_idx,
        .descs = descs,
    };
    
    int ret = ioctl(mgr->dsa_wq_fd, DSA_SUBMIT_BATCH, &batch);
    
    // Wait for completion
    while (!batch.completed) {
        _mm_pause();
    }
    
    // Post-process: apply modulo for row index
    for (int i = 0; i < desc_idx; i++) {
        row_ids[i] = row_ids[i] % ENGRAM_NUM_ROWS;
    }
    
    free(descs);
    return ret;
}
```

### NVMe P2P DMA Implementation

```c
// engram_nvme_p2p.c - NVMe P2P DMA to GPU

#include <cuda_runtime.h>
#include <cufile.h>  // GPUDirect Storage
#include "engram_ssd_manager.h"

// Initialize GPUDirect Storage for Engram
int engram_init_gds(engram_manager_t* mgr, int gpu_id) {
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    
    // Initialize cuFile driver
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        return -1;
    }
    
    // Register NVMe devices for each shard
    for (int i = 0; i < ENGRAM_NUM_SHARDS; i++) {
        memset(&cf_descr, 0, sizeof(cf_descr));
        cf_descr.handle.fd = mgr->shards[i].nvme_fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        
        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
            return -1;
        }
        mgr->shards[i].cufile_handle = cf_handle;
    }
    
    // Allocate GPU staging buffer
    size_t buffer_size = 1024 * ENGRAM_ROW_ALIGNED;  // 1024 rows max batch
    cudaSetDevice(gpu_id);
    cudaMalloc(&mgr->gpu_staging_buffer, buffer_size);
    
    // Register GPU buffer with cuFile
    cuFileBufRegister(mgr->gpu_staging_buffer, buffer_size, 0);
    
    mgr->use_p2p = true;
    mgr->gpu_device_id = gpu_id;
    
    return 0;
}

// P2P DMA read: NVMe → GPU direct
int engram_p2p_read(
    engram_manager_t* mgr,
    const uint64_t* row_ids,
    int num_rows,
    void* gpu_dest_buffer
) {
    // Group row IDs by shard
    typedef struct {
        uint64_t row_id;
        int original_idx;
    } shard_request_t;
    
    shard_request_t* shard_requests[ENGRAM_NUM_SHARDS];
    int shard_counts[ENGRAM_NUM_SHARDS] = {0};
    
    // Allocate request arrays
    for (int i = 0; i < ENGRAM_NUM_SHARDS; i++) {
        shard_requests[i] = malloc(num_rows * sizeof(shard_request_t));
    }
    
    // Partition requests by shard
    for (int i = 0; i < num_rows; i++) {
        int shard = row_ids[i] / (ENGRAM_NUM_ROWS / ENGRAM_NUM_SHARDS);
        shard = (shard >= ENGRAM_NUM_SHARDS) ? ENGRAM_NUM_SHARDS - 1 : shard;
        
        shard_requests[shard][shard_counts[shard]].row_id = row_ids[i];
        shard_requests[shard][shard_counts[shard]].original_idx = i;
        shard_counts[shard]++;
    }
    
    // Submit parallel reads to each shard
    cudaStream_t streams[ENGRAM_NUM_SHARDS];
    for (int s = 0; s < ENGRAM_NUM_SHARDS; s++) {
        if (shard_counts[s] == 0) continue;
        
        cudaStreamCreate(&streams[s]);
        
        for (int i = 0; i < shard_counts[s]; i++) {
            uint64_t local_row = shard_requests[s][i].row_id - 
                                 mgr->shards[s].base_row;
            uint64_t file_offset = local_row * ENGRAM_ROW_ALIGNED;
            int dest_idx = shard_requests[s][i].original_idx;
            
            // P2P read: NVMe → GPU
            cuFileRead(
                mgr->shards[s].cufile_handle,
                (char*)gpu_dest_buffer + dest_idx * ENGRAM_ROW_BYTES,
                ENGRAM_ROW_BYTES,
                file_offset,
                0  // Use registered buffer
            );
        }
    }
    
    // Wait for all reads
    for (int s = 0; s < ENGRAM_NUM_SHARDS; s++) {
        if (shard_counts[s] > 0) {
            cudaStreamSynchronize(streams[s]);
            cudaStreamDestroy(streams[s]);
        }
        free(shard_requests[s]);
    }
    
    return 0;
}
```

## Performance Analysis

### Latency Breakdown

```
Operation                      Software    DSA+P2P DMA
─────────────────────────────────────────────────────
Hash computation (16 hashes)   ~500ns      ~100ns (DSA)
Row ID → Shard mapping         ~50ns       ~50ns
NVMe command submission        ~1μs        ~1μs
NVMe read latency              ~80μs       ~80μs
Data transfer to GPU           ~5μs (PCIe) ~2μs (P2P)
─────────────────────────────────────────────────────
Total per token                ~86μs       ~83μs
```

### Throughput Analysis

For batch processing with 16 row lookups per token:

```
NVMe Gen4 x4 SSD: 7 GB/s read
Row size: 320 bytes
Rows per second: 7 GB/s ÷ 320 bytes = 21.8M rows/s

With 4 SSDs in parallel: 87.5M rows/s
Tokens per second: 87.5M ÷ 16 = 5.47M tokens/s

At batch size 32: 5.47M ÷ 32 = 171K batches/s
Per-batch latency: 5.8μs (amortized)
```

### Caching Strategy

Hot n-grams can be cached in DRAM/HBM for frequent patterns:

```c
// LRU cache for hot Engram rows
typedef struct {
    uint64_t row_id;
    uint64_t access_count;
    uint64_t last_access;
    uint8_t data[ENGRAM_ROW_BYTES];
} cache_entry_t;

typedef struct {
    cache_entry_t* entries;
    uint64_t num_entries;
    uint64_t hits;
    uint64_t misses;
} engram_cache_t;

// Expected hit rate for natural language:
// - Top 10K bigrams: ~60% of occurrences
// - Top 100K bigrams: ~90% of occurrences
// 
// With 1GB cache: ~3M rows cached
// Expected hit rate: 70-80% for typical text
```

## Integration with SiLens

### Modified Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SiLens + Qwen3.8-Flash-Next                  │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Token     │    │   Engram    │    │   SiLens Ternary    │ │
│  │   Input     │───►│   Manager   │───►│   Accelerator       │ │
│  │             │    │   (SSD+DMA) │    │   (MoE inference)   │ │
│  └─────────────┘    └──────┬──────┘    └──────────┬──────────┘ │
│                            │                       │            │
│                            ▼                       │            │
│                    ┌───────────────┐              │            │
│                    │  NVMe SSDs    │              │            │
│                    │  (Engram      │              │            │
│                    │   Table)      │              │            │
│                    └───────────────┘              │            │
│                                                    ▼            │
│                                           ┌───────────────┐    │
│                                           │   Output      │    │
│                                           │   Tokens      │    │
│                                           └───────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Budget

| Component | Location | Size | Access Pattern |
|-----------|----------|------|----------------|
| Engram table | NVMe SSD | 102 GB | Random read, DMA |
| Hot Engram cache | DRAM | 1-4 GB | LRU, ~70% hit |
| Ternary MoE weights | SiLens ASIC | 31 GB | Hardwired |
| KV Cache | HBM/DRAM | 4-16 GB | Sequential |
| Activations | HBM | 2-4 GB | Streaming |

**Total DRAM requirement**: 7-24 GB (vs 102+ GB with full Engram in RAM)

## Conclusion

By leveraging:
1. **Hardware hashing** (Intel DSA, AMD DMA engines, or custom FPGA)
2. **NVMe P2P DMA** (GPUDirect Storage)
3. **Smart caching** (hot n-gram LRU)

We can keep the 51.2B Engram table on SSD while achieving:
- **~80-100μs latency per token** (acceptable for inference)
- **~170K tokens/second throughput** (with batching)
- **DRAM reduction from 102GB to <10GB**

This makes Qwen3.8-Flash-Next feasible on consumer/prosumer hardware with the SiLens ternary accelerator handling the main MoE computation.

## Next Steps

1. Prototype Intel DSA hash acceleration on Sapphire Rapids
2. Benchmark GPUDirect Storage with Samsung PM1733 SSDs
3. Implement hot n-gram cache with adaptive eviction
4. Integrate with SiLens inference pipeline
5. Measure end-to-end latency on target hardware

## References

- [Intel DSA Documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-data-streaming-accelerator-for-linux.html)
- [NVIDIA GPUDirect Storage](https://developer.nvidia.com/gpudirect-storage)
- [NVMe CMB Specification](https://nvmexpress.org/specifications/)
- [Engram Paper](https://arxiv.org/abs/2601.07372)
