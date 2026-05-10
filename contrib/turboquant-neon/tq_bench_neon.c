// TurboQuant NEON benchmark
//
// Tests compress/decompress accuracy and throughput on ARM.
// Build: cc -O3 -march=native -o tq_bench_neon tq_bench_neon.c turboquant_neon.c -lm

#include "turboquant_neon.h"

#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float f16_to_f32(uint16_t h) {
    float16_t fh;
    memcpy(&fh, &h, sizeof(h));
    return (float)fh;
}

static uint16_t f32_to_f16(float f) {
    float16_t fh = (float16_t)f;
    uint16_t h;
    memcpy(&h, &fh, sizeof(h));
    return h;
}

static float cosine_similarity(const float * a, const float * b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-16f);
}

static float rmse(const float * a, const float * b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum / n);
}

int main(void) {
    const int D       = 128;   // Qwen3 head_dim
    const int n_vecs  = 4096;  // simulate one layer's KV for 4K context
    const int n_q     = 1;
    const int seed    = 42;

    printf("TurboQuant NEON benchmark\n");
    printf("D=%d, n_vectors=%d\n\n", D, n_vecs);

    // Generate random F16 input
    srand(42);
    uint16_t * kv_f16 = (uint16_t *)malloc((size_t)n_vecs * D * sizeof(uint16_t));
    float    * kv_f32 = (float *)malloc((size_t)n_vecs * D * sizeof(float));
    for (int i = 0; i < n_vecs * D; i++) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        kv_f16[i] = f32_to_f16(v);
        kv_f32[i] = f16_to_f32(kv_f16[i]); // roundtrip for fair comparison
    }

    for (int kv_bits = 3; kv_bits <= 4; kv_bits++) {
        printf("── %d-bit ──\n", kv_bits);

        tq_neon_context * ctx = tq_neon_create(D, kv_bits, seed);
        int nwq = tq_neon_n_words_quant(D, kv_bits);

        uint16_t * radii  = (uint16_t *)malloc((size_t)n_vecs * sizeof(uint16_t));
        uint32_t * packed = (uint32_t *)malloc((size_t)n_vecs * nwq * sizeof(uint32_t));
        uint16_t * out_f16 = (uint16_t *)malloc((size_t)n_vecs * D * sizeof(uint16_t));

        // Compress
        double t0 = now_ms();
        tq_neon_compress(ctx, kv_f16, n_vecs, radii, packed);
        double t_compress = now_ms() - t0;

        // Decompress
        t0 = now_ms();
        tq_neon_decompress(ctx, radii, packed, n_vecs, out_f16);
        double t_decompress = now_ms() - t0;

        // Accuracy
        float * out_f32 = (float *)malloc((size_t)n_vecs * D * sizeof(float));
        for (int i = 0; i < n_vecs * D; i++)
            out_f32[i] = f16_to_f32(out_f16[i]);

        float avg_cos = 0, avg_rmse = 0;
        for (int i = 0; i < n_vecs; i++) {
            avg_cos  += cosine_similarity(kv_f32 + i * D, out_f32 + i * D, D);
            avg_rmse += rmse(kv_f32 + i * D, out_f32 + i * D, D);
        }
        avg_cos  /= n_vecs;
        avg_rmse /= n_vecs;

        float ratio = tq_neon_compression_ratio(D, kv_bits);
        size_t bpc  = tq_neon_bytes_per_cell(D, kv_bits);

        printf("  Ratio:      %.2fx (%zu bytes/vec vs %d F16)\n", ratio, bpc, D * 2);
        printf("  Cosine sim: %.4f\n", avg_cos);
        printf("  RMSE:       %.4f\n", avg_rmse);
        printf("  Compress:   %.1f ms (%d vecs)\n", t_compress, n_vecs);
        printf("  Decompress: %.1f ms (%d vecs)\n", t_decompress, n_vecs);

        // Fused attention benchmark
        uint16_t * Q_f16  = (uint16_t *)malloc((size_t)n_q * D * sizeof(uint16_t));
        uint16_t * attn_out = (uint16_t *)malloc((size_t)n_q * D * sizeof(uint16_t));
        for (int i = 0; i < n_q * D; i++)
            Q_f16[i] = f32_to_f16(((float)rand() / RAND_MAX - 0.5f) * 2.0f);

        float attn_scale = 1.0f / sqrtf((float)D);

        int n_kv_sizes[] = {512, 2048, 4096};
        for (int s = 0; s < 3; s++) {
            int nkv = n_kv_sizes[s];
            if (nkv > n_vecs) continue;

            t0 = now_ms();
            int iters = 10;
            for (int it = 0; it < iters; it++) {
                tq_neon_fused_attention(ctx, Q_f16,
                                        radii, packed,
                                        radii, packed,  // reuse K as V for bench
                                        n_q, nkv, attn_scale, attn_out);
            }
            double t_attn = (now_ms() - t0) / iters;
            printf("  Fused attn (%d KV): %.1f ms\n", nkv, t_attn);
        }

        printf("\n");

        free(Q_f16); free(attn_out);
        free(radii); free(packed); free(out_f16); free(out_f32);
        tq_neon_free(ctx);
    }

    // Memory estimate for 262K context
    printf("── Memory estimates (Qwen3-9B, 40 layers, 8 KV heads) ──\n");
    int layers = 40, kv_heads = 8;
    int ctx_lens[] = {4096, 32768, 65536, 131072, 262144};
    for (int c = 0; c < 5; c++) {
        int ctx_len = ctx_lens[c];
        size_t fp16_total = (size_t)2 * layers * kv_heads * ctx_len * D * 2;
        for (int bits = 3; bits <= 4; bits++) {
            size_t tq_per_vec = tq_neon_bytes_per_cell(D, bits);
            size_t tq_total = (size_t)2 * layers * kv_heads * ctx_len * tq_per_vec;
            printf("  %6dK ctx, %d-bit TQ: %6.1f MB  (FP16: %6.1f MB, %.1fx savings)\n",
                   ctx_len / 1024, bits,
                   tq_total / (1024.0 * 1024.0),
                   fp16_total / (1024.0 * 1024.0),
                   (float)fp16_total / (float)tq_total);
        }
    }

    free(kv_f16); free(kv_f32);
    return 0;
}
