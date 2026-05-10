#pragma once

// TurboQuant KV cache compression utilities
//
// Phase 1: standalone compress/decompress API backed by CUDA kernels.
//          Intended for compressed state serialization and as foundation
//          for Phase 2 in-kernel Flash Attention integration.

#include <cstddef>
#include <cstdint>
#include <stdexcept>

struct tq_context; // opaque, defined in ggml-turboquant.h

struct llama_kv_cache_tq_params {
    int   head_dim   = 256;
    int   kv_bits    = 3;
    float qjl_ratio  = 0.5f;
    int   seed       = 42;
};

class llama_kv_cache_tq {
public:
    explicit llama_kv_cache_tq(const llama_kv_cache_tq_params & params);
    ~llama_kv_cache_tq();

    llama_kv_cache_tq(const llama_kv_cache_tq &) = delete;
    llama_kv_cache_tq & operator=(const llama_kv_cache_tq &) = delete;

    // Compress n_total KV vectors: F16 → TurboQuant (all device ptrs)
    void compress(const void * kv, int n_total,
                  void * radii, void * packed, void * signs,
                  void * stream = nullptr) const;

    // Decompress n_total KV vectors: TurboQuant → F16 (all device ptrs)
    void decompress(const void * radii, const void * packed, const void * signs,
                    int n_total, void * out,
                    void * stream = nullptr) const;

    // Layout
    int    head_dim()          const { return params_.head_dim; }
    int    kv_bits()           const { return params_.kv_bits; }
    int    n_words_quant()     const;
    int    n_words_signs()     const;
    size_t compressed_bytes()  const;  // per vector
    float  compression_ratio() const;  // F16 / TQ

private:
    llama_kv_cache_tq_params params_;
    tq_context * ctx_ = nullptr;
};
