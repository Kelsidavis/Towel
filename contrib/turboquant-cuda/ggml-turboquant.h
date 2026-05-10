#pragma once

// TurboQuant KV cache compression — public C API
// Backend-agnostic interface. Currently implemented for CUDA only.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to TQ matrices (rotation signs + JL projection)
typedef struct tq_context tq_context;

// Layout helpers (pure arithmetic, no GPU needed)
int    tq_calc_n_words_quant(int head_dim, int kv_bits);
int    tq_calc_n_words_signs(int head_dim, float qjl_ratio);
size_t tq_calc_bytes_per_cell(int head_dim, int kv_bits, float qjl_ratio);

// Create / destroy context (allocates GPU matrices)
tq_context * tq_context_create(int head_dim, float qjl_ratio, int kv_bits, int seed);
void         tq_context_free(tq_context * ctx);

// Compress:  kv [n_total, head_dim] F16  →  radii, packed, signs  (all device ptrs)
void tq_do_compress(
    const tq_context * ctx,
    const void       * kv,       // F16 device ptr
    int                n_total,
    void             * radii,    // F16 device ptr  [n_total]
    void             * packed,   // u32 device ptr  [n_total, n_words_q]
    void             * signs,    // u32 device ptr  [n_total, n_words_s]
    void             * stream);  // cudaStream_t (or NULL for default)

// Decompress:  radii, packed, signs  →  out [n_total, head_dim] F16
void tq_do_decompress(
    const tq_context * ctx,
    const void       * radii,
    const void       * packed,
    const void       * signs,
    int                n_total,
    void             * out,      // F16 device ptr
    void             * stream);

// ---------------------------------------------------------------------------
// Phase 2: Fused Flash Attention with in-kernel TQ decompression
//
// Decompresses K/V via FWHT in shared memory during attention, eliminating
// the pre-allocated F16 decompression workspace for true VRAM savings.
// ---------------------------------------------------------------------------

// Reference attention on F16 K/V (correctness baseline for fused kernel)
void tq_do_reference_attention(
    const tq_context * ctx,
    const void       * Q,        // F16 [n_q, head_dim]
    const void       * K,        // F16 [n_kv, head_dim]
    const void       * V,        // F16 [n_kv, head_dim]
    int                n_q,
    int                n_kv,
    float              scale,    // typically 1/sqrt(head_dim)
    void             * out,      // F16 [n_q, head_dim]
    void             * stream);

// Fused attention: reads compressed K/V, decompresses inline via FWHT
void tq_do_fused_attention(
    const tq_context * ctx,
    const void       * Q,        // F16 [n_q, head_dim]
    const void       * K_radii,  // F16 [n_kv]
    const void       * K_packed, // u32 [n_kv, n_words_q]
    const void       * K_signs,  // u32 [n_kv, n_words_s]  (NULL if use_qjl==0)
    const void       * V_radii,  // F16 [n_kv]
    const void       * V_packed, // u32 [n_kv, n_words_q]
    const void       * V_signs,  // u32 [n_kv, n_words_s]  (NULL if use_qjl==0)
    int                n_q,
    int                n_kv,
    float              scale,
    int                use_qjl,  // 0 = PolarQuant only, 1 = full TQ
    void             * out,      // F16 [n_q, head_dim]
    void             * stream);

#ifdef __cplusplus
}
#endif
