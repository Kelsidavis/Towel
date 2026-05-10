// TurboQuant KV cache compression — ARM NEON implementation
//
// PolarQuant-only port (no QJL): RWHT → radius → uniform quant → bit-pack
// Targets AArch64 NEON (Cortex-A76 / Raspberry Pi 5).
//
// The CUDA version uses one thread per vector element (blockDim=D).
// Here we process each vector sequentially with NEON 4-wide SIMD.
// D=128 is the primary target (Qwen3 head_dim), D=256 also supported.

#include "turboquant_neon.h"

#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct tq_neon_context {
    int     head_dim;
    int     kv_bits;
    int     n_words_q;
    float   quant_range;
    float * rand_signs;  // [head_dim]  ±1.0f
};

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

int tq_neon_n_words_quant(int D, int kv_bits) {
    int vpw = 32 / kv_bits;
    int D_padded = D + ((-D % vpw) + vpw) % vpw;
    return D_padded / vpw;
}

size_t tq_neon_bytes_per_cell(int D, int kv_bits) {
    return sizeof(uint16_t)  // radius (FP16)
         + (size_t)tq_neon_n_words_quant(D, kv_bits) * 4;  // packed quant
}

float tq_neon_compression_ratio(int D, int kv_bits) {
    size_t fp16_bytes = (size_t)D * 2;
    return (float)fp16_bytes / (float)tq_neon_bytes_per_cell(D, kv_bits);
}

// ---------------------------------------------------------------------------
// Deterministic random signs (must match CUDA for interop)
// ---------------------------------------------------------------------------

static void fill_rand_signs(float * out, int D, int seed) {
    uint64_t state = (uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL;
    for (int i = 0; i < D; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out[i] = (state >> 63) ? 1.0f : -1.0f;
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

tq_neon_context * tq_neon_create(int head_dim, int kv_bits, int seed) {
    tq_neon_context * ctx = (tq_neon_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->head_dim    = head_dim;
    ctx->kv_bits     = kv_bits;
    ctx->n_words_q   = tq_neon_n_words_quant(head_dim, kv_bits);
    ctx->quant_range = 4.0f / sqrtf((float)head_dim);

    ctx->rand_signs = (float *)aligned_alloc(64, (size_t)head_dim * sizeof(float));
    fill_rand_signs(ctx->rand_signs, head_dim, seed);

    return ctx;
}

void tq_neon_free(tq_neon_context * ctx) {
    if (!ctx) return;
    free(ctx->rand_signs);
    free(ctx);
}

// ---------------------------------------------------------------------------
// FP16 ↔ FP32 helpers (NEON)
// ---------------------------------------------------------------------------

static inline float f16_to_f32(uint16_t h) {
    float16_t fh;
    memcpy(&fh, &h, sizeof(h));
    return (float)fh;
}

static inline uint16_t f32_to_f16(float f) {
    float16_t fh = (float16_t)f;
    uint16_t h;
    memcpy(&h, &fh, sizeof(h));
    return h;
}

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform — NEON vectorized
//
// In-place on float buf[D]. Normalized by 1/sqrt(D).
// D must be a power of 2 and >= 4.
//
// For stride >= 4, we process 4 butterflies at once with NEON.
// For stride 1 and 2, we use vzip/vuzp patterns.
// ---------------------------------------------------------------------------

static void fwht_neon(float * buf, int D) {
    // Stride 1: pairs (0,1), (2,3), ...
    // Load 4 floats [a0 a1 a2 a3], deinterleave to [a0 a2] [a1 a3],
    // butterfly, reinterleave.
    for (int i = 0; i < D; i += 4) {
        float32x4_t v = vld1q_f32(buf + i);
        // v = [x0 x1 x2 x3]
        // pairs: (x0,x1) and (x2,x3)
        float32x2_t lo = vget_low_f32(v);   // [x0, x1]
        float32x2_t hi = vget_high_f32(v);  // [x2, x3]
        // a = [x0, x2], b = [x1, x3]
        float32x2x2_t t0 = vtrn_f32(lo, hi);
        float32x2_t a = t0.val[0]; // [x0, x2]
        float32x2_t b = t0.val[1]; // [x1, x3]
        float32x2_t sum  = vadd_f32(a, b);
        float32x2_t diff = vsub_f32(a, b);
        // re-interleave: [sum0, diff0, sum1, diff1]
        float32x2x2_t t1 = vtrn_f32(sum, diff);
        vst1q_f32(buf + i, vcombine_f32(t1.val[0], t1.val[1]));
    }

    // Stride 2: pairs (0,2),(1,3), (4,6),(5,7), ...
    for (int i = 0; i < D; i += 4) {
        float32x4_t v = vld1q_f32(buf + i);
        // v = [x0 x1 x2 x3], pairs: (x0,x2) and (x1,x3)
        float32x2_t lo = vget_low_f32(v);   // [x0, x1]
        float32x2_t hi = vget_high_f32(v);  // [x2, x3]
        float32x2_t sum  = vadd_f32(lo, hi);  // [x0+x2, x1+x3]
        float32x2_t diff = vsub_f32(lo, hi);  // [x0-x2, x1-x3]
        vst1q_f32(buf + i, vcombine_f32(sum, diff));
    }

    // Stride 4, 8, ..., D/2: contiguous blocks, straightforward NEON
    for (int stride = 4; stride < D; stride <<= 1) {
        int block_size = stride * 2;
        for (int base = 0; base < D; base += block_size) {
            float * a_ptr = buf + base;
            float * b_ptr = buf + base + stride;
            for (int j = 0; j < stride; j += 4) {
                float32x4_t a = vld1q_f32(a_ptr + j);
                float32x4_t b = vld1q_f32(b_ptr + j);
                vst1q_f32(a_ptr + j, vaddq_f32(a, b));
                vst1q_f32(b_ptr + j, vsubq_f32(a, b));
            }
        }
    }

    // Normalize by 1/sqrt(D)
    float inv_sqrt_D = 1.0f / sqrtf((float)D);
    float32x4_t norm_v = vdupq_n_f32(inv_sqrt_D);
    for (int i = 0; i < D; i += 4) {
        vst1q_f32(buf + i, vmulq_f32(vld1q_f32(buf + i), norm_v));
    }
}

// ---------------------------------------------------------------------------
// L2 norm — NEON
// ---------------------------------------------------------------------------

static float vec_l2_norm_neon(const float * v, int D) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < D; i += 4) {
        float32x4_t x = vld1q_f32(v + i);
        acc = vfmaq_f32(acc, x, x);
    }
    // Horizontal sum
    float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum2 = vpadd_f32(sum2, sum2);
    float total = vget_lane_f32(sum2, 0);
    return sqrtf(total > 1e-16f ? total : 1e-16f);
}

// ---------------------------------------------------------------------------
// Dot product — NEON
// ---------------------------------------------------------------------------

static float vec_dot_neon(const float * a, const float * b, int D) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < D; i += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}

// ---------------------------------------------------------------------------
// Compress one vector
// ---------------------------------------------------------------------------

static void compress_one(
    const tq_neon_context * ctx,
    const uint16_t * kv_f16,    // [D] input
    uint16_t       * radius_out,
    uint32_t       * packed_out) // [n_words_q] output
{
    const int D        = ctx->head_dim;
    const int kv_bits  = ctx->kv_bits;
    const int n_words  = ctx->n_words_q;
    const float qr     = ctx->quant_range;
    const float * signs = ctx->rand_signs;

    // Stack buffer for rotated vector
    float buf[512]; // max D=512

    // Load F16 → F32, apply random signs
    for (int i = 0; i < D; i += 4) {
        float16x4_t h = vld1_f16((const float16_t *)(kv_f16 + i));
        float32x4_t f = vcvt_f32_f16(h);
        float32x4_t s = vld1q_f32(signs + i);
        vst1q_f32(buf + i, vmulq_f32(f, s));
    }

    // FWHT
    fwht_neon(buf, D);

    // L2 norm → radius
    float norm = vec_l2_norm_neon(buf, D);
    *radius_out = f32_to_f16(norm);

    // Uniform quantize unit vector + bit-pack
    float inv_norm   = 1.0f / norm;
    float inv_2qr    = 1.0f / (2.0f * qr);
    int   n_levels   = (1 << kv_bits) - 1;
    float fn_levels  = (float)n_levels;
    int   vpw        = 32 / kv_bits;
    uint32_t bitmask = (1u << kv_bits) - 1u;

    memset(packed_out, 0, (size_t)n_words * sizeof(uint32_t));

    for (int i = 0; i < D; i++) {
        float unit = buf[i] * inv_norm;
        float scaled = (unit + qr) * inv_2qr;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 1.0f) scaled = 1.0f;
        uint32_t qval = (uint32_t)(scaled * fn_levels + 0.5f);

        int word_idx = i / vpw;
        int bit_off  = (i % vpw) * kv_bits;
        packed_out[word_idx] |= (qval & bitmask) << bit_off;
    }
}

// ---------------------------------------------------------------------------
// Decompress one vector into float buffer (internal, for fused attention)
// ---------------------------------------------------------------------------

static void decompress_one_f32(
    const tq_neon_context * ctx,
    uint16_t         radius_f16,
    const uint32_t * packed,
    float          * out_f32)
{
    const int D        = ctx->head_dim;
    const int kv_bits  = ctx->kv_bits;
    const float qr     = ctx->quant_range;
    const float * signs = ctx->rand_signs;

    int   n_levels   = (1 << kv_bits) - 1;
    float inv_nlev   = 1.0f / (float)n_levels;
    float qr2        = 2.0f * qr;
    float norm       = f16_to_f32(radius_f16);
    int   vpw        = 32 / kv_bits;
    uint32_t bitmask = (1u << kv_bits) - 1u;

    // Dequantize into out_f32
    for (int i = 0; i < D; i++) {
        int word_idx = i / vpw;
        int bit_off  = (i % vpw) * kv_bits;
        uint32_t qval = (packed[word_idx] >> bit_off) & bitmask;
        float y_unit = (float)qval * inv_nlev * qr2 - qr;
        out_f32[i] = y_unit * norm;
    }

    // Inverse FWHT
    fwht_neon(out_f32, D);

    // Undo random signs
    for (int i = 0; i < D; i += 4) {
        float32x4_t v = vld1q_f32(out_f32 + i);
        float32x4_t s = vld1q_f32(signs + i);
        vst1q_f32(out_f32 + i, vmulq_f32(v, s));
    }
}

// ---------------------------------------------------------------------------
// Public: compress
// ---------------------------------------------------------------------------

void tq_neon_compress(
    const tq_neon_context * ctx,
    const uint16_t        * kv_f16,
    int                     n_total,
    uint16_t              * radii,
    uint32_t              * packed)
{
    const int D    = ctx->head_dim;
    const int nwq  = ctx->n_words_q;

    for (int i = 0; i < n_total; i++) {
        compress_one(ctx,
                     kv_f16  + i * D,
                     radii   + i,
                     packed  + i * nwq);
    }
}

// ---------------------------------------------------------------------------
// Public: decompress
// ---------------------------------------------------------------------------

void tq_neon_decompress(
    const tq_neon_context * ctx,
    const uint16_t        * radii,
    const uint32_t        * packed,
    int                     n_total,
    uint16_t              * out_f16)
{
    const int D   = ctx->head_dim;
    const int nwq = ctx->n_words_q;

    float buf[512];

    for (int i = 0; i < n_total; i++) {
        decompress_one_f32(ctx, radii[i], packed + i * nwq, buf);

        // F32 → F16
        for (int j = 0; j < D; j += 4) {
            float32x4_t v = vld1q_f32(buf + j);
            float16x4_t h = vcvt_f16_f32(v);
            vst1_f16((float16_t *)(out_f16 + i * D + j), h);
        }
    }
}

// ---------------------------------------------------------------------------
// Public: fused attention (online softmax, streaming KV decompression)
//
// No workspace allocation — decompresses one KV vector at a time into a
// stack buffer, exactly like the CUDA Phase 2 kernel uses shared memory.
// ---------------------------------------------------------------------------

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
    uint16_t              * out)
{
    const int D   = ctx->head_dim;
    const int nwq = ctx->n_words_q;

    float q_buf[512];
    float k_buf[512];
    float v_buf[512];
    float o_buf[512];

    for (int qi = 0; qi < n_q; qi++) {
        // Load Q vector: F16 → F32, pre-scale
        for (int j = 0; j < D; j += 4) {
            float16x4_t h = vld1_f16((const float16_t *)(Q + qi * D + j));
            float32x4_t f = vcvt_f32_f16(h);
            vst1q_f32(q_buf + j, vmulq_f32(f, vdupq_n_f32(scale)));
        }

        // Online softmax state
        float m_prev = -1e30f;
        float l_prev = 0.0f;
        memset(o_buf, 0, (size_t)D * sizeof(float));

        for (int kvi = 0; kvi < n_kv; kvi++) {
            // Decompress K[kvi]
            decompress_one_f32(ctx, K_radii[kvi],
                               K_packed + kvi * nwq, k_buf);

            // score = Q · K
            float score = vec_dot_neon(q_buf, k_buf, D);

            // Online softmax update
            float new_max = score > m_prev ? score : m_prev;
            float rescale = expf(m_prev - new_max);
            float attn_w  = expf(score - new_max);

            // Rescale running output + accumulate
            float32x4_t rs_v = vdupq_n_f32(rescale);
            float32x4_t aw_v = vdupq_n_f32(attn_w);

            // Decompress V[kvi]
            decompress_one_f32(ctx, V_radii[kvi],
                               V_packed + kvi * nwq, v_buf);

            for (int j = 0; j < D; j += 4) {
                float32x4_t o = vld1q_f32(o_buf + j);
                float32x4_t v = vld1q_f32(v_buf + j);
                o = vfmaq_f32(vmulq_f32(o, rs_v), aw_v, v);
                vst1q_f32(o_buf + j, o);
            }

            l_prev = l_prev * rescale + attn_w;
            m_prev = new_max;
        }

        // Normalize and write F16 output
        float inv_l = 1.0f / (l_prev > 1e-16f ? l_prev : 1e-16f);
        float32x4_t inv_v = vdupq_n_f32(inv_l);
        for (int j = 0; j < D; j += 4) {
            float32x4_t o = vmulq_f32(vld1q_f32(o_buf + j), inv_v);
            float16x4_t h = vcvt_f16_f32(o);
            vst1_f16((float16_t *)(out + qi * D + j), h);
        }
    }
}
