# TurboQuant CUDA — KV cache compression for llama.cpp

Port of Towel's MLX-based TurboQuant to NVIDIA GPUs via CUDA.

## Algorithm

Two-stage quantization using Randomized Walsh-Hadamard Transform (RWHT):

1. **PolarQuant**: RWHT rotation → radius extraction → uniform quantize to kv_bits
2. **QJL**: Johnson-Lindenstrauss projection of residual → 1-bit sign storage

Uses FWHT (O(D log D)) instead of full random orthogonal matrix (O(D²)),
making it efficient in GPU shared memory without loading a large rotation matrix.

## Phase 1 Results (RTX 5080, D=256, standalone compress/decompress)

| Setting     | Ratio  | Cosine Sim | RMSE  | Decompress |
|-------------|--------|------------|-------|------------|
| 3-bit + QJL | 4.20x  | 0.9575     | 0.174 | 32.6 ms    |
| 4-bit + QJL | 3.51x  | 0.9903     | 0.081 | 32.6 ms    |

## Phase 2 Results (RTX 5080, D=256, fused Flash Attention + TQ)

In-kernel TQ decompression during attention — K/V decompressed via FWHT
in shared memory, **no F16 workspace buffer needed**.

| Context | Setting | VRAM (K+V) | Savings | Attn Cosine | Attn Time | vs Ref |
|---------|---------|-----------|---------|-------------|-----------|--------|
| 16K     | 4-bit   | 4.56 MiB  | 3.51x   | 0.9831      | 17.6 ms   | 6.0x   |
| 65K     | 3-bit   | 15.25 MiB | 4.20x   | 0.9397      | 81.4 ms   | 4.8x   |

**Key**: QJL correction is too bandwidth-heavy for in-kernel use (J matrix
reads dominate). PolarQuant-only (no QJL) is the practical path: 4.2x VRAM
savings at ~5x attention overhead. Since attention is ~10-30% of total
inference, the overall slowdown is moderate while enabling much longer contexts.

## Files

| File | Purpose |
|------|---------|
| `turboquant.cuh` | CUDA kernel declarations + layout helpers |
| `turboquant.cu` | CUDA kernels (FWHT, compress, decompress, fused FA) + C API |
| `ggml-turboquant.h` | Public C API header (Phase 1 + Phase 2) |
| `llama-kv-cache-tq.h/cpp` | C++ wrapper class |
| `tq-bench.cpp` | Standalone benchmark (Phase 1 compress/decompress + Phase 2 fused attention) |

## Build (in llama.cpp tree)

1. Copy `turboquant.cuh`, `turboquant.cu` → `ggml/src/ggml-cuda/`
2. Copy `ggml-turboquant.h` → `ggml/include/`
3. Copy `llama-kv-cache-tq.h/cpp` → `src/`
4. Add `llama-kv-cache-tq.cpp` to `src/CMakeLists.txt`
5. Copy `tq-bench.cpp`, `CMakeLists.txt` → `tools/tq-bench/`
6. Build: `cmake -B build -DGGML_CUDA=ON && cmake --build build --target tq-bench`

## Architecture

### Phase 1: Standalone compress/decompress

Standalone CUDA kernels for KV cache serialization (4.2x smaller saves).
One thread block per vector, `blockDim.x = D = 256`.

### Phase 2: Fused Flash Attention + TQ decompression

Custom Flash Attention kernel that reads compressed K/V directly from VRAM
and decompresses each vector inline via FWHT in shared memory. Uses online
softmax (O(1) extra memory per output element).

**Shared memory footprint**: 1 KiB FWHT workspace + 32 bytes reduction buffer.
This is ~30x smaller than the standard FA tile kernel's shared memory usage.

**Algorithm per Q vector** (one CUDA block):
```
for each KV vector:
  1. Decompress K via FWHT in shared memory (1 KiB)
  2. Compute dot(Q, K_decompressed) via warp + cross-warp reduction
  3. Online softmax update (running max + sum in registers)
  4. Decompress V via FWHT (reuses same shared memory)
  5. Accumulate attention_weight × V_decompressed in registers
Normalize output by running sum
```

The FWHT approach was chosen specifically for Phase 2: the transform fits
entirely in shared memory (D floats = 1 KiB for D=256), unlike the full
rotation matrix (256 KiB) which wouldn't fit.
