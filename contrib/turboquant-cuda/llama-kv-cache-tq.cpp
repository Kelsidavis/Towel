#include "llama-kv-cache-tq.h"
#include "ggml-turboquant.h"

llama_kv_cache_tq::llama_kv_cache_tq(const llama_kv_cache_tq_params & params)
    : params_(params) {
    ctx_ = tq_context_create(params_.head_dim, params_.qjl_ratio,
                              params_.kv_bits, params_.seed);
    if (!ctx_) {
        throw std::runtime_error("llama_kv_cache_tq: failed to create TQ context (requires CUDA)");
    }
}

llama_kv_cache_tq::~llama_kv_cache_tq() {
    tq_context_free(ctx_);
}

void llama_kv_cache_tq::compress(
    const void * kv, int n_total,
    void * radii, void * packed, void * signs,
    void * stream) const
{
    tq_do_compress(ctx_, kv, n_total, radii, packed, signs, stream);
}

void llama_kv_cache_tq::decompress(
    const void * radii, const void * packed, const void * signs,
    int n_total, void * out,
    void * stream) const
{
    tq_do_decompress(ctx_, radii, packed, signs, n_total, out, stream);
}

int llama_kv_cache_tq::n_words_quant() const {
    return tq_calc_n_words_quant(params_.head_dim, params_.kv_bits);
}

int llama_kv_cache_tq::n_words_signs() const {
    return tq_calc_n_words_signs(params_.head_dim, params_.qjl_ratio);
}

size_t llama_kv_cache_tq::compressed_bytes() const {
    return tq_calc_bytes_per_cell(params_.head_dim, params_.kv_bits, params_.qjl_ratio);
}

float llama_kv_cache_tq::compression_ratio() const {
    size_t f16_bytes = (size_t)params_.head_dim * 2; // sizeof(half)
    size_t tq_bytes  = compressed_bytes();
    return tq_bytes > 0 ? (float)f16_bytes / (float)tq_bytes : 0.0f;
}
