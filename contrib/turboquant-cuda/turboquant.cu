// TurboQuant KV cache compression — CUDA implementation
//
// Uses Randomized Walsh-Hadamard Transform (RWHT) instead of a full random
// orthogonal matrix, giving O(D log D) rotation that fits in shared memory.
//
// Reference: TurboQuant paper (Google Research, ICLR 2026) + FWHT adaptation

#include "turboquant.cuh"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

// ---------------------------------------------------------------------------
// Shared-memory Fast Walsh-Hadamard Transform (in-place, D must be power of 2)
// Normalized: result has unit L2 norm if input is unit-norm
// Each thread block handles ONE vector; blockDim.x == D.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void fwht_inplace(volatile float * sh, int D) {
    const int tid = threadIdx.x;
    for (int stride = 1; stride < D; stride <<= 1) {
        int group = tid / (stride * 2);
        int pos   = tid % (stride * 2);
        if (pos < stride) {
            int a_idx = group * stride * 2 + pos;
            int b_idx = a_idx + stride;
            float a = sh[a_idx];
            float b = sh[b_idx];
            sh[a_idx] = a + b;
            sh[b_idx] = a - b;
        }
        __syncthreads();
    }
    // normalize by 1/sqrt(D) for orthogonality
    sh[tid] *= rsqrtf((float)D);
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Warp-reduce sum helper
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

// ---------------------------------------------------------------------------
// Compress kernel
// One thread block per vector. blockDim.x == D (e.g. 256).
// ---------------------------------------------------------------------------

template<int D>
__global__ void tq_compress_kernel(
    const half     * __restrict__ kv,       // [n_total, D]
    const float    * __restrict__ rand_signs,// [D]  ±1
    const float    * __restrict__ J,         // [D, m]  JL matrix
    half           * __restrict__ radii,    // [n_total]
    uint32_t       * __restrict__ packed,   // [n_total, n_words_q]
    uint32_t       * __restrict__ signs_out,// [n_total, n_words_s]
    int n_total, int kv_bits, float quant_range,
    int n_words_q, int m, int n_words_s)
{
    __shared__ float sh[D];

    const int vid = blockIdx.x;
    if (vid >= n_total) return;
    const int tid = threadIdx.x;

    // ---- Stage 1a: load + apply random signs ----
    sh[tid] = __half2float(kv[vid * D + tid]) * rand_signs[tid];
    __syncthreads();

    // ---- Stage 1b: FWHT ----
    fwht_inplace(sh, D);
    // sh[tid] now holds the rotated vector

    // ---- Stage 1c: compute L2 norm via block reduction ----
    float v = sh[tid];
    float sq = v * v;
    __shared__ float sh_norm[32]; // one partial sum per warp
    float wsum = warp_reduce_sum(sq);
    if ((tid & 31) == 0) sh_norm[tid >> 5] = wsum;
    __syncthreads();
    // Final reduction + broadcast via sh_norm[0]
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < D / 32; i++) total += sh_norm[i];
        sh_norm[0] = sqrtf(fmaxf(total, 1e-16f));
    }
    __syncthreads();
    float norm = sh_norm[0]; // all threads see the same norm

    if (tid == 0) radii[vid] = __float2half(norm);

    // ---- Stage 1d: uniform quantize unit vector ----
    float unit = v / norm;
    int n_levels = (1 << kv_bits) - 1;
    float scaled = fminf(fmaxf((unit + quant_range) / (2.0f * quant_range), 0.0f), 1.0f);
    uint32_t qval = (uint32_t) rintf(scaled * (float)n_levels);

    // ---- Stage 1e: pack bits ----
    // vals_per_word = 32 / kv_bits
    const int vpw = 32 / kv_bits;
    // Each thread tid owns value qval. Determine which word and bit offset.
    int word_idx = tid / vpw;
    int bit_off  = (tid % vpw) * kv_bits;
    uint32_t contrib = (qval & ((1u << kv_bits) - 1u)) << bit_off;

    // Use shared memory as packed word accumulator (zero first)
    __shared__ uint32_t sh_packed[D / 1 + 4]; // over-allocate; we need n_words_q words
    if (tid < n_words_q) sh_packed[tid] = 0;
    __syncthreads();

    atomicOr(&sh_packed[word_idx], contrib);
    __syncthreads();

    if (tid < n_words_q)
        packed[vid * n_words_q + tid] = sh_packed[tid];
    __syncthreads();

    // ---- Stage 2: JL residual ----
    // dequant to reconstruct unit vector, then inverse FWHT to get x_recon
    float y_deq = (float)((sh_packed[word_idx] >> bit_off) & ((1u << kv_bits) - 1u));
    float x_unit_recon = (y_deq / (float)n_levels) * (2.0f * quant_range) - quant_range;
    // x_recon in rotated space (scaled by radius):
    float y_recon = x_unit_recon * norm;

    // residual in rotated space
    float residual = v - y_recon; // v = sh[tid] which is the rotated original

    // JL projection: residual [D] × J [D, m] → projected [m]
    // Each thread handles multiple j outputs (m may be < D)
    // We reuse sh[] for projected values
    if (m > 0) {
        __shared__ float sh_proj[D];
        // Zero projection buffer
        for (int j = tid; j < m; j += D) {
            sh_proj[j] = 0.0f;
        }
        __syncthreads();
        // For j in [0, m): proj[j] += residual[tid] * J[tid*m + j]
        // Thread tid processes its row of J
        for (int j = 0; j < m; j++) {
            atomicAdd(&sh_proj[j], residual * J[tid * m + j]);
        }
        __syncthreads();

        // Pack sign bits of sh_proj[0..m-1]
        __shared__ uint32_t sh_signs[D / 32 + 1];
        if (tid < n_words_s) sh_signs[tid] = 0;
        __syncthreads();

        if (tid < m) {
            int sw = tid >> 5;
            int sb = tid & 31;
            if (sh_proj[tid] >= 0.0f)
                atomicOr(&sh_signs[sw], (1u << sb));
        }
        __syncthreads();

        if (tid < n_words_s)
            signs_out[vid * n_words_s + tid] = sh_signs[tid];
    }
}

// ---------------------------------------------------------------------------
// Decompress kernel
// One thread block per vector. blockDim.x == D.
// ---------------------------------------------------------------------------

template<int D>
__global__ void tq_decompress_kernel(
    const float    * __restrict__ rand_signs,// [D]  ±1
    const float    * __restrict__ J,         // [D, m]
    const half     * __restrict__ radii,    // [n_total]
    const uint32_t * __restrict__ packed,   // [n_total, n_words_q]
    const uint32_t * __restrict__ signs_in, // [n_total, n_words_s]
    half           * __restrict__ out,       // [n_total, D]
    int n_total, int kv_bits, float quant_range,
    int n_words_q, int m, int n_words_s)
{
    __shared__ float sh[D];

    const int vid = blockIdx.x;
    if (vid >= n_total) return;
    const int tid = threadIdx.x;

    float norm = __half2float(radii[vid]);

    // ---- Unpack + dequantize ----
    const int vpw = 32 / kv_bits;
    int word_idx = tid / vpw;
    int bit_off  = (tid % vpw) * kv_bits;
    uint32_t qval = (packed[vid * n_words_q + word_idx] >> bit_off)
                    & ((1u << kv_bits) - 1u);

    int n_levels = (1 << kv_bits) - 1;
    float y_unit = ((float)qval / (float)n_levels) * (2.0f * quant_range) - quant_range;
    // scaled by radius
    sh[tid] = y_unit * norm;
    __syncthreads();

    // ---- JL residual correction (in rotated space) ----
    if (m > 0 && n_words_s > 0) {
        // residual_approx[tid] = sum_j(signs[j] * J[tid, j]) * scale
        float scale = quant_range / (float)n_levels;
        float corr = 0.0f;
        for (int j = 0; j < m; j++) {
            float sj = ((signs_in[vid * n_words_s + (j >> 5)] >> (j & 31)) & 1u) ? 1.0f : -1.0f;
            corr += sj * J[tid * m + j];
        }
        sh[tid] += corr * scale;
        __syncthreads();
    }

    // ---- Inverse FWHT (same op, but we scale differently) ----
    // Forward FWHT normalized by 1/sqrt(D).
    // Inverse: apply FWHT again (H is self-inverse up to factor D), then multiply by
    // rand_signs and scale by 1/sqrt(D).
    // Net: x = rand_signs * H(y) / sqrt(D)
    // Since fwht_inplace already applies 1/sqrt(D), we call it and then multiply signs.
    fwht_inplace(sh, D);
    sh[tid] *= rand_signs[tid];
    __syncthreads();

    // ---- Write F16 output ----
    out[vid * D + tid] = __float2half(sh[tid]);
}

// ---------------------------------------------------------------------------
// Host: generate random sign vector and JL matrix on GPU
// ---------------------------------------------------------------------------

static void fill_rand_signs_cpu(float * h, int D, int seed) {
    // Deterministic ±1 signs from seed
    uint64_t state = (uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL;
    for (int i = 0; i < D; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        h[i] = (state >> 63) ? 1.0f : -1.0f;
    }
}

static void fill_jl_matrix_cpu(float * h, int D, int m, int seed) {
    // Random Gaussian scaled by 1/sqrt(m)
    float scale = 1.0f / sqrtf((float)m);
    // Simple Box-Muller / LCG for determinism
    uint64_t state = (uint64_t)(seed + 95) * 6364136223846793005ULL + 1;
    for (int i = 0; i < D * m; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        // Map uint64 to [0,1)
        float u1 = ((state >> 32) & 0xFFFFFFFF) / 4294967296.0f + 1e-10f;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u2 = ((state >> 32) & 0xFFFFFFFF) / 4294967296.0f;
        // Box-Muller
        float n = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        h[i] = n * scale;
    }
}

TQMatrices * tq_matrices_create(int D, float qjl_ratio, int kv_bits, int seed) {
    TQMatrices * m = new TQMatrices;
    m->D        = D;
    m->kv_bits  = kv_bits;
    m->m        = (int)(D * qjl_ratio);
    m->quant_range = 4.0f / sqrtf((float)D);
    m->n_words_q   = tq_n_words_quant(D, kv_bits);
    m->n_words_s   = tq_n_words_signs(m->m);

    // rand_signs
    float * h_signs = new float[D];
    fill_rand_signs_cpu(h_signs, D, seed);
    CUDA_CHECK(cudaMalloc(&m->rand_signs, D * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(m->rand_signs, h_signs, D * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_signs;

    // JL matrix [D, m]
    if (m->m > 0) {
        float * h_J = new float[(size_t)D * m->m];
        fill_jl_matrix_cpu(h_J, D, m->m, seed);
        CUDA_CHECK(cudaMalloc(&m->J, (size_t)D * m->m * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(m->J, h_J, (size_t)D * m->m * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_J;
    } else {
        m->J = nullptr;
    }

    return m;
}

void tq_matrices_free(TQMatrices * m) {
    if (!m) return;
    if (m->rand_signs) cudaFree(m->rand_signs);
    if (m->J)          cudaFree(m->J);
    delete m;
}

// ---------------------------------------------------------------------------
// Launch wrappers
// ---------------------------------------------------------------------------

void tq_compress(
    const TQMatrices * mats,
    const half       * kv,
    int                n_total,
    half             * radii,
    uint32_t         * packed,
    uint32_t         * signs,
    cudaStream_t       stream)
{
    if (n_total == 0) return;
    const int D = mats->D;
    GGML_ASSERT(D == 256 && "tq_compress: only D=256 supported in Phase 1");

    dim3 grid(n_total);
    dim3 block(D);

    tq_compress_kernel<256><<<grid, block, 0, stream>>>(
        kv, mats->rand_signs, mats->J,
        radii, packed, signs,
        n_total, mats->kv_bits, mats->quant_range,
        mats->n_words_q, mats->m, mats->n_words_s);
    CUDA_CHECK(cudaGetLastError());
}

void tq_decompress(
    const TQMatrices * mats,
    const half       * radii,
    const uint32_t   * packed,
    const uint32_t   * signs,
    int                n_total,
    half             * out,
    cudaStream_t       stream)
{
    if (n_total == 0) return;
    const int D = mats->D;
    GGML_ASSERT(D == 256 && "tq_decompress: only D=256 supported in Phase 1");

    dim3 grid(n_total);
    dim3 block(D);

    tq_decompress_kernel<256><<<grid, block, 0, stream>>>(
        mats->rand_signs, mats->J,
        radii, packed, signs,
        out,
        n_total, mats->kv_bits, mats->quant_range,
        mats->n_words_q, mats->m, mats->n_words_s);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Phase 2: Fused Flash Attention with in-kernel TQ decompression
//
// Decompresses K/V vectors via FWHT in shared memory during attention,
// eliminating the pre-allocated F16 decompression workspace.
//
// One block per Q vector. blockDim.x == D == 256.
// Uses online softmax (O(1) extra memory per output element).
// ---------------------------------------------------------------------------

// Reference attention kernel: standard F16 K/V, online softmax.
// Used as correctness baseline for the fused kernel.
template<int D>
__global__ void tq_reference_attention_kernel(
    const half  * __restrict__ Q,     // [n_q, D]
    const half  * __restrict__ K,     // [n_kv, D]
    const half  * __restrict__ V,     // [n_kv, D]
    half        * __restrict__ out,   // [n_q, D]
    int n_kv, float scale)
{
    const int q_idx = blockIdx.x;
    const int tid   = threadIdx.x;

    __shared__ float sh_reduce[D / 32];

    float q_val = __half2float(Q[q_idx * D + tid]) * scale;

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float o_acc  = 0.0f;

    for (int kv_idx = 0; kv_idx < n_kv; kv_idx++) {
        float k_val = __half2float(K[kv_idx * D + tid]);

        // Dot product Q @ K  (full block reduction)
        float dot = q_val * k_val;
        dot = warp_reduce_sum(dot);
        if ((tid & 31) == 0) sh_reduce[tid >> 5] = dot;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < D / 32; w++) total += sh_reduce[w];
            sh_reduce[0] = total;
        }
        __syncthreads();
        float score = sh_reduce[0];

        // Online softmax
        float new_max = fmaxf(m_prev, score);
        float rescale = expf(m_prev - new_max);
        float attn_w  = expf(score - new_max);
        o_acc  = o_acc * rescale + attn_w * __half2float(V[kv_idx * D + tid]);
        l_prev = l_prev * rescale + attn_w;
        m_prev = new_max;
    }

    out[q_idx * D + tid] = __float2half(o_acc / fmaxf(l_prev, 1e-16f));
}

// Fused Flash Attention + TQ decompression kernel.
// Reads compressed K/V directly from global memory, decompresses each vector
// via FWHT in shared memory, computes attention inline.
// USE_QJL: whether to apply Johnson-Lindenstrauss residual correction.
template<int D, bool USE_QJL>
__global__ void tq_fused_attention_kernel(
    const half     * __restrict__ Q,           // [n_q, D]   F16
    const half     * __restrict__ K_radii,     // [n_kv]     F16
    const uint32_t * __restrict__ K_packed,    // [n_kv, n_words_q] u32
    const uint32_t * __restrict__ K_signs,     // [n_kv, n_words_s] u32
    const half     * __restrict__ V_radii,     // [n_kv]     F16
    const uint32_t * __restrict__ V_packed,    // [n_kv, n_words_q] u32
    const uint32_t * __restrict__ V_signs,     // [n_kv, n_words_s] u32
    const float    * __restrict__ rand_signs,  // [D]  ±1
    const float    * __restrict__ J,           // [D, m] JL matrix (nullptr if !USE_QJL)
    half           * __restrict__ out,         // [n_q, D]   F16
    int n_kv, int kv_bits, float quant_range,
    int n_words_q, int m, int n_words_s,
    float scale)
{
    const int q_idx = blockIdx.x;
    const int tid   = threadIdx.x;

    __shared__ float sh[D];           // FWHT workspace (1 KiB)
    __shared__ float sh_reduce[D/32]; // cross-warp dot product reduction

    // Load Q and random sign into registers (persistent across all KV)
    const float q_val   = __half2float(Q[q_idx * D + tid]) * scale;
    const float my_sign = rand_signs[tid];

    // Pre-compute bit-packing constants
    const int      vpw          = 32 / kv_bits;
    const int      n_levels     = (1 << kv_bits) - 1;
    const int      my_word      = tid / vpw;
    const int      my_bit       = (tid % vpw) * kv_bits;
    const uint32_t bit_mask     = (1u << kv_bits) - 1u;
    const float    inv_n_levels = 1.0f / (float)n_levels;
    const float    qr2          = 2.0f * quant_range;
    const float    jl_scale     = USE_QJL ? (quant_range * inv_n_levels) : 0.0f;

    // Online softmax state (per-thread, one output dimension each)
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float o_acc  = 0.0f;

    for (int kv_idx = 0; kv_idx < n_kv; kv_idx++) {
        // ============ Decompress K[kv_idx] ============
        {
            float norm_k   = __half2float(K_radii[kv_idx]);
            uint32_t qval  = (K_packed[kv_idx * n_words_q + my_word] >> my_bit) & bit_mask;
            float y_unit   = (float)qval * inv_n_levels * qr2 - quant_range;
            sh[tid] = y_unit * norm_k;
            __syncthreads();

            if constexpr (USE_QJL) {
                float corr = 0.0f;
                for (int j = 0; j < m; j++) {
                    float sj = ((K_signs[kv_idx * n_words_s + (j >> 5)] >> (j & 31)) & 1u)
                               ? 1.0f : -1.0f;
                    corr += sj * J[tid * m + j];
                }
                sh[tid] += corr * jl_scale;
                __syncthreads();
            }

            // Inverse FWHT (self-inverse up to normalization)
            fwht_inplace(sh, D);
        }
        float k_val = sh[tid] * my_sign; // decompressed K value for this dim

        // ============ Dot product Q @ K ============
        float dot = q_val * k_val;
        dot = warp_reduce_sum(dot);
        if ((tid & 31) == 0) sh_reduce[tid >> 5] = dot;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < D / 32; w++) total += sh_reduce[w];
            sh_reduce[0] = total;
        }
        __syncthreads();
        float score = sh_reduce[0]; // all threads see the same score

        // ============ Online softmax update ============
        float new_max = fmaxf(m_prev, score);
        float rescale = expf(m_prev - new_max);
        float attn_w  = expf(score - new_max);
        o_acc  *= rescale;
        l_prev  = l_prev * rescale + attn_w;

        // ============ Decompress V[kv_idx] ============
        // sh[] is free: k_val is in a register, sh_reduce is separate
        {
            float norm_v   = __half2float(V_radii[kv_idx]);
            uint32_t qval  = (V_packed[kv_idx * n_words_q + my_word] >> my_bit) & bit_mask;
            float y_unit   = (float)qval * inv_n_levels * qr2 - quant_range;
            sh[tid] = y_unit * norm_v;
            __syncthreads();

            if constexpr (USE_QJL) {
                float corr = 0.0f;
                for (int j = 0; j < m; j++) {
                    float sj = ((V_signs[kv_idx * n_words_s + (j >> 5)] >> (j & 31)) & 1u)
                               ? 1.0f : -1.0f;
                    corr += sj * J[tid * m + j];
                }
                sh[tid] += corr * jl_scale;
                __syncthreads();
            }

            fwht_inplace(sh, D);
        }
        float v_val = sh[tid] * my_sign; // decompressed V value for this dim

        // ============ Accumulate weighted V ============
        o_acc += attn_w * v_val;
        m_prev = new_max;
    }

    // Normalize and write output
    out[q_idx * D + tid] = __float2half(o_acc / fmaxf(l_prev, 1e-16f));
}

// ---------------------------------------------------------------------------
// Phase 2 launch wrappers
// ---------------------------------------------------------------------------

void tq_reference_attention(
    const TQMatrices * mats,
    const half * Q, const half * K, const half * V,
    int n_q, int n_kv, float scale,
    half * out, cudaStream_t stream)
{
    if (n_q == 0 || n_kv == 0) return;
    GGML_ASSERT(mats->D == 256 && "tq_reference_attention: only D=256 supported");
    tq_reference_attention_kernel<256><<<n_q, 256, 0, stream>>>(
        Q, K, V, out, n_kv, scale);
    CUDA_CHECK(cudaGetLastError());
}

void tq_fused_attention(
    const TQMatrices * mats,
    const half * Q,
    const half * K_radii, const uint32_t * K_packed, const uint32_t * K_signs,
    const half * V_radii, const uint32_t * V_packed, const uint32_t * V_signs,
    int n_q, int n_kv, float scale, bool use_qjl,
    half * out, cudaStream_t stream)
{
    if (n_q == 0 || n_kv == 0) return;
    GGML_ASSERT(mats->D == 256 && "tq_fused_attention: only D=256 supported");

    if (use_qjl) {
        tq_fused_attention_kernel<256, true><<<n_q, 256, 0, stream>>>(
            Q, K_radii, K_packed, K_signs,
            V_radii, V_packed, V_signs,
            mats->rand_signs, mats->J,
            out, n_kv, mats->kv_bits, mats->quant_range,
            mats->n_words_q, mats->m, mats->n_words_s, scale);
    } else {
        tq_fused_attention_kernel<256, false><<<n_q, 256, 0, stream>>>(
            Q, K_radii, K_packed, nullptr,
            V_radii, V_packed, nullptr,
            mats->rand_signs, nullptr,
            out, n_kv, mats->kv_bits, mats->quant_range,
            mats->n_words_q, 0, 0, scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// C API (exported via ggml-turboquant.h)
// ---------------------------------------------------------------------------

#include "ggml-turboquant.h"

struct tq_context {
    TQMatrices * mats;
};

extern "C" {

int tq_calc_n_words_quant(int head_dim, int kv_bits) {
    return tq_n_words_quant(head_dim, kv_bits);
}

int tq_calc_n_words_signs(int head_dim, float qjl_ratio) {
    return tq_n_words_signs(tq_jl_dim(head_dim, qjl_ratio));
}

size_t tq_calc_bytes_per_cell(int head_dim, int kv_bits, float qjl_ratio) {
    return tq_bytes_per_cell(head_dim, kv_bits, qjl_ratio);
}

tq_context * tq_context_create(int head_dim, float qjl_ratio, int kv_bits, int seed) {
    tq_context * ctx = new tq_context;
    ctx->mats = tq_matrices_create(head_dim, qjl_ratio, kv_bits, seed);
    return ctx;
}

void tq_context_free(tq_context * ctx) {
    if (!ctx) return;
    tq_matrices_free(ctx->mats);
    delete ctx;
}

void tq_do_compress(
    const tq_context * ctx,
    const void       * kv,
    int                n_total,
    void             * radii,
    void             * packed,
    void             * signs,
    void             * stream)
{
    tq_compress(ctx->mats,
                static_cast<const half *>(kv), n_total,
                static_cast<half *>(radii),
                static_cast<uint32_t *>(packed),
                static_cast<uint32_t *>(signs),
                static_cast<cudaStream_t>(stream));
}

void tq_do_decompress(
    const tq_context * ctx,
    const void       * radii,
    const void       * packed,
    const void       * signs,
    int                n_total,
    void             * out,
    void             * stream)
{
    tq_decompress(ctx->mats,
                  static_cast<const half *>(radii),
                  static_cast<const uint32_t *>(packed),
                  static_cast<const uint32_t *>(signs),
                  n_total,
                  static_cast<half *>(out),
                  static_cast<cudaStream_t>(stream));
}

// Phase 2: reference attention on F16 K/V
void tq_do_reference_attention(
    const tq_context * ctx,
    const void       * Q,
    const void       * K,
    const void       * V,
    int                n_q,
    int                n_kv,
    float              scale,
    void             * out,
    void             * stream)
{
    tq_reference_attention(ctx->mats,
                           static_cast<const half *>(Q),
                           static_cast<const half *>(K),
                           static_cast<const half *>(V),
                           n_q, n_kv, scale,
                           static_cast<half *>(out),
                           static_cast<cudaStream_t>(stream));
}

// Phase 2: fused attention reading compressed K/V directly
void tq_do_fused_attention(
    const tq_context * ctx,
    const void       * Q,
    const void       * K_radii,
    const void       * K_packed,
    const void       * K_signs,
    const void       * V_radii,
    const void       * V_packed,
    const void       * V_signs,
    int                n_q,
    int                n_kv,
    float              scale,
    int                use_qjl,
    void             * out,
    void             * stream)
{
    tq_fused_attention(ctx->mats,
                       static_cast<const half *>(Q),
                       static_cast<const half *>(K_radii),
                       static_cast<const uint32_t *>(K_packed),
                       static_cast<const uint32_t *>(K_signs),
                       static_cast<const half *>(V_radii),
                       static_cast<const uint32_t *>(V_packed),
                       static_cast<const uint32_t *>(V_signs),
                       n_q, n_kv, scale, use_qjl != 0,
                       static_cast<half *>(out),
                       static_cast<cudaStream_t>(stream));
}

} // extern "C"
