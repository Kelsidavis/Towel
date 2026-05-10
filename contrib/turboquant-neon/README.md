# TurboQuant NEON — KV cache compression for ARM

Port of Towel's TurboQuant to ARM NEON (AArch64) for Raspberry Pi 5 and
other Cortex-A76+ targets.

## Algorithm

PolarQuant-only (no QJL — see CUDA README for rationale):

1. Random sign flip (deterministic, matches CUDA seed)
2. Fast Walsh-Hadamard Transform (NEON-vectorized butterfly)
3. L2 norm extraction (radius)
4. Uniform quantization to kv_bits (3 or 4)
5. Bit-packing into uint32 words

QJL is omitted because the J matrix reads dominate bandwidth on both GPU
and CPU. PolarQuant alone gives ~4.2x compression at 3-bit.

## Build (standalone benchmark)

```bash
cmake -B build && cmake --build build
./build/tq_bench_neon
```

## Integration with llama.cpp

To use as a KV cache backend in llama.cpp on ARM:

1. Copy `turboquant_neon.h`, `turboquant_neon.c` → `ggml/src/`
2. Hook `tq_neon_compress` into `llama_kv_cache_update` after K/V projection
3. Replace KV reads in attention with `tq_neon_fused_attention` or
   decompress-on-read via `tq_neon_decompress`

## Expected memory savings (Qwen3-9B, D=128, 40 layers, 8 KV heads)

| Context | FP16 KV | 3-bit TQ | 4-bit TQ |
|---------|---------|----------|----------|
| 4K      | 320 MB  | ~76 MB   | ~91 MB   |
| 32K     | 2.5 GB  | ~610 MB  | ~729 MB  |
| 64K     | 5.0 GB  | ~1.2 GB  | ~1.5 GB  |
| 262K    | 20.5 GB | ~4.9 GB  | ~5.8 GB  |

With 3-bit TQ + IQ3_M weights (4.5 GB), a Pi 5 with 8 GB RAM could
theoretically run ~32K context. 262K remains out of reach (4.9 GB KV +
4.5 GB weights > 8 GB before OS overhead).

## Files

| File | Purpose |
|------|---------|
| `turboquant_neon.h` | Public C API |
| `turboquant_neon.c` | ARM NEON implementation (compress, decompress, fused attention) |
| `tq_bench_neon.c` | Accuracy + throughput benchmark |
| `CMakeLists.txt` | Build |
