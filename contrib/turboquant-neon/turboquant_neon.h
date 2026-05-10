#pragma once

// TurboQuant KV cache compression — ARM NEON implementation
//
// Port of the CUDA TurboQuant (PolarQuant-only, no QJL) to ARM NEON
// for Raspberry Pi 5 (Cortex-A76) and other AArch64 targets.
//
// Algorithm: RWHT rotation → radius extraction → uniform quantization
// Compression ratio (D=128, 3-bit): ~4.2x vs FP16
//
// Drop-in replacement for ggml-turboquant.h on non-CUDA builds.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context
typedef struct tq_neon_context tq_neon_context;

// Layout helpers
int    tq_neon_n_words_quant(int head_dim, int kv_bits);
size_t tq_neon_bytes_per_cell(int head_dim, int kv_bits);
float  tq_neon_compression_ratio(int head_dim, int kv_bits);

// Lifecycle
tq_neon_context * tq_neon_create(int head_dim, int kv_bits, int seed);
void              tq_neon_free(tq_neon_context * ctx);

// Compress n_total KV vectors: F16 → TQ
//   kv_f16:  [n_total, head_dim]  input  (uint16_t holding IEEE 754 FP16)
//   radii:   [n_total]            output (FP16)
//   packed:  [n_total, n_words_q] output (uint32_t, bit-packed quantized values)
void tq_neon_compress(
    const tq_neon_context * ctx,
    const uint16_t        * kv_f16,
    int                     n_total,
    uint16_t              * radii,
    uint32_t              * packed);

// Decompress n_total KV vectors: TQ → F16
//   out_f16: [n_total, head_dim]  output (FP16)
void tq_neon_decompress(
    const tq_neon_context * ctx,
    const uint16_t        * radii,
    const uint32_t        * packed,
    int                     n_total,
    uint16_t              * out_f16);

// Fused attention: Q @ decompress(K)^T → softmax → @ decompress(V)
// Online softmax, streaming KV decompression, no F16 workspace.
//   Q:        [n_q, head_dim]       F16 input
//   K_radii:  [n_kv]               F16
//   K_packed: [n_kv, n_words_q]    uint32
//   V_radii:  [n_kv]               F16
//   V_packed: [n_kv, n_words_q]    uint32
//   out:      [n_q, head_dim]       F16 output
void tq_neon_fused_attention(
    const tq_neon_context * ctx,
    const uint16_t        * Q,
    const uint16_t        * K_radii,
    const uint32_t        * K_packed,
    const uint16_t        * V_radii,
    const uint32_t        * V_packed,
    int                     n_q,
    int                     n_kv,
    float                   scale,
    uint16_t              * out);

#ifdef __cplusplus
}
#endif
