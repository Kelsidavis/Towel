#pragma once

// TurboQuant KV cache compression for NVIDIA GPUs
//
// Algorithm: Two-stage quantization
//   Stage 1 (PolarQuant): Randomized Walsh-Hadamard rotation → radius extraction
//                          → uniform quantization to kv_bits (default 3)
//   Stage 2 (QJL):        JL projection of residual → 1-bit sign storage
//
// Compression ratio (D=256, 3-bit, qjl_ratio=0.5):
//   F16:         512 bytes/head
//   TQ (3-bit):  ~114 bytes/head  (~4.5x)
//
// Phase 1 (this implementation):
//   Compress/decompress as standalone CUDA ops.
//   Used by llama_kv_cache_tq for state serialization (4.5x smaller saves).
//
// Phase 2 (future):
//   Decompress in-tile inside fattn-tile.cu → true VRAM savings during inference.
//   The kernels below are designed to be callable from the FA tile loop.

#include "common.cuh"
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

// Number of uint32 words to pack D values at kv_bits bits/value
// Uses 32//kv_bits values per word (same as MLX reference implementation)
inline int tq_n_words_quant(int D, int kv_bits) {
    int vals_per_word = 32 / kv_bits;
    int D_padded = D + ((-D % vals_per_word) + vals_per_word) % vals_per_word;
    return D_padded / vals_per_word;
}

// Number of uint32 words to pack m sign bits
inline int tq_n_words_signs(int m) {
    return (m + 31) / 32;
}

// JL output dimension
inline int tq_jl_dim(int D, float qjl_ratio) {
    return (int)(D * qjl_ratio);
}

// Total compressed bytes per (head, cell) pair
inline size_t tq_bytes_per_cell(int D, int kv_bits, float qjl_ratio) {
    int m = tq_jl_dim(D, qjl_ratio);
    return sizeof(half)                                     // radius
         + (size_t)tq_n_words_quant(D, kv_bits) * 4        // packed quant
         + (size_t)tq_n_words_signs(m) * 4;                 // JL signs
}

// ---------------------------------------------------------------------------
// GPU-side random matrices (generated once, stored on device)
// ---------------------------------------------------------------------------

struct TQMatrices {
    float * rand_signs;  // [D]   random ±1 for FWHT rotation
    float * J;           // [D,m] JL projection matrix (scaled 1/sqrt(m))
    int     D;
    int     m;           // JL output dim = D * qjl_ratio
    int     kv_bits;
    float   quant_range; // 4.0f / sqrt(D)
    int     n_words_q;   // ceil(D * kv_bits / (32/kv_bits * kv_bits)) -- words for quant
    int     n_words_s;   // ceil(m / 32)  -- words for signs
};

// Allocate and fill TQMatrices on the current CUDA device
TQMatrices * tq_matrices_create(int D, float qjl_ratio, int kv_bits, int seed);
void         tq_matrices_free(TQMatrices * m);

// ---------------------------------------------------------------------------
// Compress: F16 KV → TQ compressed format
//
// kv:       [n_total, D]          F16 input (row-major)
// radii:    [n_total]             F16 output
// packed:   [n_total, n_words_q]  uint32 output (packed quantized values)
// signs:    [n_total, n_words_s]  uint32 output (packed JL signs)
//
// n_total = n_heads * n_tokens  (caller reshapes)
// ---------------------------------------------------------------------------
void tq_compress(
    const TQMatrices * mats,
    const half       * kv,
    int                n_total,
    half             * radii,
    uint32_t         * packed,
    uint32_t         * signs,
    cudaStream_t       stream);

// ---------------------------------------------------------------------------
// Decompress: TQ compressed format → F16 KV
//
// out:      [n_total, D]          F16 output
// ---------------------------------------------------------------------------
void tq_decompress(
    const TQMatrices * mats,
    const half       * radii,
    const uint32_t   * packed,
    const uint32_t   * signs,
    int                n_total,
    half             * out,
    cudaStream_t       stream);
