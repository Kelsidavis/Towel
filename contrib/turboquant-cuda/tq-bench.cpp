// TurboQuant KV cache compression benchmark
//
// Tests compression quality and speed on random F16 vectors.
// Usage: tq-bench [n_vectors] [head_dim] [kv_bits] [qjl_ratio]
//
// Example: tq-bench 262144 256 3 0.5
//   → simulates 262K context, 256-dim heads, 3-bit quant, 50% JL ratio

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml-turboquant.h"

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                     \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count() / 1000.0;
}

int main(int argc, char ** argv) {
    int   n_vectors = argc > 1 ? atoi(argv[1]) : 4096;
    int   D         = argc > 2 ? atoi(argv[2]) : 256;
    int   kv_bits   = argc > 3 ? atoi(argv[3]) : 3;
    float qjl_r     = argc > 4 ? (float)atof(argv[4]) : 0.5f;
    int   seed      = 42;

    printf("TurboQuant KV Cache Benchmark\n");
    printf("  vectors:    %d\n", n_vectors);
    printf("  head_dim:   %d\n", D);
    printf("  kv_bits:    %d\n", kv_bits);
    printf("  qjl_ratio:  %.2f\n", qjl_r);
    printf("\n");

    int nwq = tq_calc_n_words_quant(D, kv_bits);
    int nws = tq_calc_n_words_signs(D, qjl_r);
    size_t bytes_tq  = tq_calc_bytes_per_cell(D, kv_bits, qjl_r);
    size_t bytes_f16 = (size_t)D * sizeof(half);
    float ratio = (float)bytes_f16 / (float)bytes_tq;

    printf("  packed words/vec:  %d  (%zu bytes)\n", nwq, (size_t)nwq * 4);
    printf("  sign words/vec:    %d  (%zu bytes)\n", nws, (size_t)nws * 4);
    printf("  radii:             2 bytes\n");
    printf("  total TQ/vec:      %zu bytes\n", bytes_tq);
    printf("  total F16/vec:     %zu bytes\n", bytes_f16);
    printf("  compression:       %.2fx\n\n", ratio);

    // -- Init context --
    printf("Initializing TQ context ... ");
    fflush(stdout);
    tq_context * ctx = tq_context_create(D, qjl_r, kv_bits, seed);
    printf("done.\n");

    // -- Allocate GPU buffers --
    size_t kv_bytes     = (size_t)n_vectors * D * sizeof(half);
    size_t radii_bytes  = (size_t)n_vectors * sizeof(half);
    size_t packed_bytes = (size_t)n_vectors * nwq * sizeof(uint32_t);
    size_t signs_bytes  = (size_t)n_vectors * nws * sizeof(uint32_t);

    half     * d_kv, * d_radii, * d_out;
    uint32_t * d_packed, * d_signs;

    CHECK_CUDA(cudaMalloc(&d_kv,     kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_radii,  radii_bytes));
    CHECK_CUDA(cudaMalloc(&d_packed, packed_bytes));
    CHECK_CUDA(cudaMalloc(&d_signs,  signs_bytes));
    CHECK_CUDA(cudaMalloc(&d_out,    kv_bytes));

    // Fill random F16 data
    {
        std::vector<half> h_kv((size_t)n_vectors * D);
        uint64_t rng = 12345;
        for (auto & v : h_kv) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            float f = ((rng >> 32) & 0xFFFFFFFF) / 4294967296.0f * 2.0f - 1.0f;
            v = __float2half_rn(f);
        }
        CHECK_CUDA(cudaMemcpy(d_kv, h_kv.data(), kv_bytes, cudaMemcpyHostToDevice));
    }

    // -- Warmup --
    printf("Warmup ... ");
    fflush(stdout);
    tq_do_compress(ctx, d_kv, n_vectors, d_radii, d_packed, d_signs, nullptr);
    tq_do_decompress(ctx, d_radii, d_packed, d_signs, n_vectors, d_out, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("done.\n");

    // -- Benchmark --
    int n_iter = 10;

    CHECK_CUDA(cudaDeviceSynchronize());
    double t0 = now_ms();
    for (int i = 0; i < n_iter; i++)
        tq_do_compress(ctx, d_kv, n_vectors, d_radii, d_packed, d_signs, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_compress = (now_ms() - t0) / n_iter;

    CHECK_CUDA(cudaDeviceSynchronize());
    t0 = now_ms();
    for (int i = 0; i < n_iter; i++)
        tq_do_decompress(ctx, d_radii, d_packed, d_signs, n_vectors, d_out, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_decompress = (now_ms() - t0) / n_iter;

    printf("\nTiming (avg over %d iters):\n", n_iter);
    printf("  compress:    %7.2f ms  (%6.1f M vec/s)\n",
           t_compress, n_vectors / t_compress / 1000.0);
    printf("  decompress:  %7.2f ms  (%6.1f M vec/s)\n",
           t_decompress, n_vectors / t_decompress / 1000.0);

    // -- Quality --
    {
        std::vector<half> h_kv((size_t)n_vectors * D);
        std::vector<half> h_out((size_t)n_vectors * D);
        CHECK_CUDA(cudaMemcpy(h_kv.data(),  d_kv,  kv_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, kv_bytes, cudaMemcpyDeviceToHost));

        double sum_cos = 0.0, sum_mse = 0.0;
        int n_sample = n_vectors < 10000 ? n_vectors : 10000;
        for (int i = 0; i < n_sample; i++) {
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (int d = 0; d < D; d++) {
                float a = __half2float(h_kv [(size_t)i * D + d]);
                float b = __half2float(h_out[(size_t)i * D + d]);
                dot += (double)a * b;
                na  += (double)a * a;
                nb  += (double)b * b;
                sum_mse += (double)(a - b) * (a - b);
            }
            if (na > 0 && nb > 0)
                sum_cos += dot / (sqrt(na) * sqrt(nb));
        }

        printf("\nQuality (%d vectors):\n", n_sample);
        printf("  cosine similarity:  %.6f\n", sum_cos / n_sample);
        printf("  RMSE:               %.6e\n", sqrt(sum_mse / (n_sample * D)));
    }

    // -- Memory --
    size_t total_tq = radii_bytes + packed_bytes + signs_bytes;
    printf("\nMemory:\n");
    printf("  F16 cache:   %8.2f MiB\n", (double)kv_bytes / (1024 * 1024));
    printf("  TQ cache:    %8.2f MiB\n", (double)total_tq / (1024 * 1024));
    printf("  ratio:       %.2fx\n", (double)kv_bytes / total_tq);

    // ======================================================================
    // Phase 2: Fused Flash Attention + TQ decompression benchmark
    // ======================================================================
    printf("\n========================================\n");
    printf("Phase 2: Fused Flash Attention + TQ\n");
    printf("========================================\n\n");

    int n_q  = 1;   // single token generation (most common inference case)
    int n_kv = n_vectors;

    // Allocate separate K and V (reuse d_kv as K, allocate new V)
    half * d_K = d_kv; // reuse existing random data as K
    half * d_V;
    CHECK_CUDA(cudaMalloc(&d_V, kv_bytes));
    {
        std::vector<half> h_v((size_t)n_kv * D);
        uint64_t rng = 67890;
        for (auto & v : h_v) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            float f = ((rng >> 32) & 0xFFFFFFFF) / 4294967296.0f * 2.0f - 1.0f;
            v = __float2half_rn(f);
        }
        CHECK_CUDA(cudaMemcpy(d_V, h_v.data(), kv_bytes, cudaMemcpyHostToDevice));
    }

    // Allocate Q
    size_t q_bytes = (size_t)n_q * D * sizeof(half);
    half * d_Q;
    CHECK_CUDA(cudaMalloc(&d_Q, q_bytes));
    {
        std::vector<half> h_q((size_t)n_q * D);
        uint64_t rng = 11111;
        for (auto & v : h_q) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            float f = ((rng >> 32) & 0xFFFFFFFF) / 4294967296.0f * 2.0f - 1.0f;
            v = __float2half_rn(f);
        }
        CHECK_CUDA(cudaMemcpy(d_Q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
    }

    // Compress K and V separately
    half     * d_K_radii, * d_V_radii;
    uint32_t * d_K_packed, * d_K_signs, * d_V_packed, * d_V_signs;
    CHECK_CUDA(cudaMalloc(&d_K_radii,  radii_bytes));
    CHECK_CUDA(cudaMalloc(&d_K_packed, packed_bytes));
    CHECK_CUDA(cudaMalloc(&d_K_signs,  signs_bytes));
    CHECK_CUDA(cudaMalloc(&d_V_radii,  radii_bytes));
    CHECK_CUDA(cudaMalloc(&d_V_packed, packed_bytes));
    CHECK_CUDA(cudaMalloc(&d_V_signs,  signs_bytes));

    tq_do_compress(ctx, d_K, n_kv, d_K_radii, d_K_packed, d_K_signs, nullptr);
    tq_do_compress(ctx, d_V, n_kv, d_V_radii, d_V_packed, d_V_signs, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocate output buffers
    half * d_out_ref, * d_out_fused;
    CHECK_CUDA(cudaMalloc(&d_out_ref,   q_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_fused, q_bytes));

    float scale = 1.0f / sqrtf((float)D);
    printf("  n_q:         %d\n", n_q);
    printf("  n_kv:        %d\n", n_kv);
    printf("  scale:       %.6f\n\n", scale);

    // -- Warmup --
    printf("Warmup ... ");
    fflush(stdout);
    tq_do_reference_attention(ctx, d_Q, d_K, d_V, n_q, n_kv, scale, d_out_ref, nullptr);
    tq_do_fused_attention(ctx, d_Q, d_K_radii, d_K_packed, d_K_signs,
                          d_V_radii, d_V_packed, d_V_signs,
                          n_q, n_kv, scale, 1, d_out_fused, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("done.\n");

    // -- Benchmark: Reference F16 attention --
    CHECK_CUDA(cudaDeviceSynchronize());
    t0 = now_ms();
    for (int i = 0; i < n_iter; i++)
        tq_do_reference_attention(ctx, d_Q, d_K, d_V, n_q, n_kv, scale, d_out_ref, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_ref = (now_ms() - t0) / n_iter;

    // -- Benchmark: Fused TQ attention (with QJL) --
    CHECK_CUDA(cudaDeviceSynchronize());
    t0 = now_ms();
    for (int i = 0; i < n_iter; i++)
        tq_do_fused_attention(ctx, d_Q, d_K_radii, d_K_packed, d_K_signs,
                              d_V_radii, d_V_packed, d_V_signs,
                              n_q, n_kv, scale, 1, d_out_fused, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_fused_qjl = (now_ms() - t0) / n_iter;

    // -- Benchmark: Fused TQ attention (PolarQuant only, no QJL) --
    half * d_out_fused_nq;
    CHECK_CUDA(cudaMalloc(&d_out_fused_nq, q_bytes));
    CHECK_CUDA(cudaDeviceSynchronize());
    t0 = now_ms();
    for (int i = 0; i < n_iter; i++)
        tq_do_fused_attention(ctx, d_Q, d_K_radii, d_K_packed, d_K_signs,
                              d_V_radii, d_V_packed, d_V_signs,
                              n_q, n_kv, scale, 0, d_out_fused_nq, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_fused_noqjl = (now_ms() - t0) / n_iter;

    printf("\nAttention timing (avg over %d iters, n_kv=%d):\n", n_iter, n_kv);
    printf("  reference (F16):         %7.2f ms\n", t_ref);
    printf("  fused TQ (with QJL):     %7.2f ms  (%.1fx vs ref)\n",
           t_fused_qjl, t_fused_qjl / t_ref);
    printf("  fused TQ (no QJL):       %7.2f ms  (%.1fx vs ref)\n",
           t_fused_noqjl, t_fused_noqjl / t_ref);

    // -- Quality: compare fused output vs reference --
    {
        std::vector<half> h_ref((size_t)n_q * D);
        std::vector<half> h_fused((size_t)n_q * D);
        std::vector<half> h_fused_nq((size_t)n_q * D);
        CHECK_CUDA(cudaMemcpy(h_ref.data(),      d_out_ref,      q_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_fused.data(),    d_out_fused,    q_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_fused_nq.data(), d_out_fused_nq, q_bytes, cudaMemcpyDeviceToHost));

        auto cosine_sim = [&](const std::vector<half> & a, const std::vector<half> & b) {
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (size_t i = 0; i < a.size(); i++) {
                float fa = __half2float(a[i]);
                float fb = __half2float(b[i]);
                dot += (double)fa * fb;
                na  += (double)fa * fa;
                nb  += (double)fb * fb;
            }
            return (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
        };

        auto max_abs_err = [&](const std::vector<half> & a, const std::vector<half> & b) {
            float mx = 0.0f;
            for (size_t i = 0; i < a.size(); i++) {
                float diff = fabsf(__half2float(a[i]) - __half2float(b[i]));
                if (diff > mx) mx = diff;
            }
            return mx;
        };

        printf("\nAttention output quality (vs F16 reference):\n");
        printf("  fused TQ (with QJL):  cosine=%.6f  max_err=%.6f\n",
               cosine_sim(h_ref, h_fused), max_abs_err(h_ref, h_fused));
        printf("  fused TQ (no QJL):    cosine=%.6f  max_err=%.6f\n",
               cosine_sim(h_ref, h_fused_nq), max_abs_err(h_ref, h_fused_nq));
    }

    // -- VRAM comparison --
    size_t vram_f16 = (size_t)n_kv * D * sizeof(half) * 2; // K + V in F16
    size_t vram_tq  = (radii_bytes + packed_bytes + signs_bytes) * 2; // K + V compressed
    printf("\nVRAM for KV cache (%d tokens):\n", n_kv);
    printf("  F16 (K+V):   %8.2f MiB\n", (double)vram_f16 / (1024 * 1024));
    printf("  TQ  (K+V):   %8.2f MiB\n", (double)vram_tq / (1024 * 1024));
    printf("  savings:     %.2fx  (%.1f%% reduction)\n",
           (double)vram_f16 / vram_tq,
           (1.0 - (double)vram_tq / vram_f16) * 100.0);
    printf("  F16 decompression workspace: 0 bytes (eliminated by Phase 2)\n");

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_K_radii);
    cudaFree(d_K_packed);
    cudaFree(d_K_signs);
    cudaFree(d_V_radii);
    cudaFree(d_V_packed);
    cudaFree(d_V_signs);
    cudaFree(d_out_ref);
    cudaFree(d_out_fused);
    cudaFree(d_out_fused_nq);
    cudaFree(d_kv);
    cudaFree(d_radii);
    cudaFree(d_packed);
    cudaFree(d_signs);
    cudaFree(d_out);
    tq_context_free(ctx);

    return 0;
}
