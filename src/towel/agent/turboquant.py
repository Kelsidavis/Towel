"""TurboQuant KV cache — two-stage quantization with PolarQuant + QJL.

Implements the TurboQuant framework for extreme KV cache compression:
  Stage 1: Random rotation + PolarQuant (uniform quantization, no scale factors)
  Stage 2: QJL (1-bit sign quantization of residuals via Johnson-Lindenstrauss)

Reference: "TurboQuant: Redefining AI Efficiency with Extreme Compression"
           Google Research, ICLR 2026 / AISTATS 2026

Memory per vector (d=128, 3-bit, qjl_ratio=0.5):
  TurboQuant: ~56 bytes vs float16: 256 bytes → ~4.6x compression
  vs MLX built-in 4-bit QuantizedKVCache: ~72 bytes → ~3.6x
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import _BaseCache, create_attention_mask

log = logging.getLogger("towel.turboquant")


# ---------------------------------------------------------------------------
# Random matrix generation (one-time, on CPU via numpy)
# ---------------------------------------------------------------------------


def _random_orthogonal(d: int, seed: int) -> mx.array:
    """Random orthogonal matrix via QR decomposition of Gaussian matrix."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(d, d).astype(np.float32))
    return mx.array(Q)


def _random_jl_matrix(d: int, m: int, seed: int) -> mx.array:
    """Random Johnson-Lindenstrauss projection matrix, scaled by 1/sqrt(m)."""
    rng = np.random.RandomState(seed)
    return mx.array(rng.randn(d, m).astype(np.float32) / math.sqrt(m))


# ---------------------------------------------------------------------------
# Bit packing utilities
# ---------------------------------------------------------------------------


def _pack_bits(values: mx.array, bits: int) -> mx.array:
    """Pack low-bit integers into uint32 words.

    Args:
        values: integer values in [0, 2^bits - 1], shape (..., D)
        bits: bits per value (1–4)

    Returns:
        uint32 array, shape (..., ceil(D * bits / 32))
    """
    *batch, D = values.shape
    vals_per_word = 32 // bits
    pad = (-D) % vals_per_word
    if pad:
        values = mx.concatenate(
            [values, mx.zeros((*batch, pad), dtype=values.dtype)], axis=-1
        )

    D_padded = D + pad
    n_words = D_padded // vals_per_word
    values = values.reshape(*batch, n_words, vals_per_word).astype(mx.uint32)

    shifts = mx.arange(vals_per_word, dtype=mx.uint32) * bits  # [vals_per_word]
    packed = (values << shifts).sum(axis=-1)  # no overlap ⇒ sum == bitwise OR
    return packed


def _unpack_bits(packed: mx.array, bits: int, D: int) -> mx.array:
    """Unpack uint32 words to low-bit integer values.

    Args:
        packed: uint32 array, shape (..., n_words)
        bits: bits per value
        D: original (un-padded) dimension

    Returns:
        uint32 array, shape (..., D)
    """
    *batch, n_words = packed.shape
    vals_per_word = 32 // bits
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)

    shifts = mx.arange(vals_per_word, dtype=mx.uint32) * bits  # [vals_per_word]
    expanded = (mx.expand_dims(packed, -1) >> shifts) & mask
    return expanded.reshape(*batch, n_words * vals_per_word)[..., :D]


# ---------------------------------------------------------------------------
# TurboQuant KV Cache
# ---------------------------------------------------------------------------


class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant two-stage compression.

    Drop-in replacement for ``mlx_lm.models.cache.KVCache``.  Keys and values
    are compressed on write and decompressed on read so the attention layer
    sees ordinary float16 tensors.

    Compression pipeline (per vector):
      1. Rotate by a fixed random orthogonal matrix R.
      2. Extract radius (L2 norm) and unit direction.
      3. Uniform-quantize the rotated components to ``kv_bits`` without any
         per-group scale/bias (the rotation makes the distribution uniform).
      4. Compute the reconstruction residual.
      5. Project the residual with a random JL matrix and store only the
         sign bits (QJL — zero overhead for quantization constants).
    """

    step = 256  # pre-allocation granularity, same as upstream KVCache

    def __init__(
        self,
        kv_bits: int = 3,
        qjl_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.kv_bits = kv_bits
        self.qjl_ratio = qjl_ratio
        self.seed = seed

        # Lazily initialised on first update_and_fetch
        self._head_dim: int = 0
        self._qjl_dim: int = 0
        self._R: Optional[mx.array] = None
        self._J: Optional[mx.array] = None
        self._quant_range: float = 0.0

        # Compressed storage (None until first write)
        self._key_radii: Optional[mx.array] = None
        self._key_packed: Optional[mx.array] = None
        self._key_signs: Optional[mx.array] = None
        self._val_radii: Optional[mx.array] = None
        self._val_packed: Optional[mx.array] = None
        self._val_signs: Optional[mx.array] = None

        self.offset = 0

    # -- lazy init -----------------------------------------------------------

    def _init_matrices(self, d: int) -> None:
        self._head_dim = d
        self._qjl_dim = max(int(d * self.qjl_ratio), 8)
        self._R = _random_orthogonal(d, seed=self.seed)
        self._J = _random_jl_matrix(d, self._qjl_dim, seed=self.seed + 95)
        # After orthogonal rotation a unit-vector's components are ≈ N(0, 1/d).
        # 4/√d covers >99.99 % of the mass.
        self._quant_range = 4.0 / math.sqrt(d)

    # -- compress / decompress -----------------------------------------------

    def _compress(self, x: mx.array):
        """Compress [B, H, S, D] → (radii, packed_quant, packed_signs)."""
        d = self._head_dim
        n_levels = (1 << self.kv_bits) - 1  # 7 for 3-bit, 15 for 4-bit
        qr = self._quant_range

        # Stage 1 — PolarQuant
        x_rot = x @ self._R
        radii = mx.sqrt((x_rot * x_rot).sum(axis=-1, keepdims=True))
        radii = mx.maximum(radii, 1e-8)
        x_unit = x_rot / radii

        # Uniform quantization (no per-group scale — the whole point)
        x_scaled = mx.clip((x_unit + qr) / (2 * qr), 0, 1)
        x_quant = mx.round(x_scaled * n_levels).astype(mx.uint32)
        packed_quant = _pack_bits(x_quant, self.kv_bits)

        # Dequantize for residual
        x_deq = (x_quant.astype(mx.float32) / n_levels) * (2 * qr) - qr
        x_recon = (x_deq * radii) @ self._R.T

        # Stage 2 — QJL on residual
        residual = x - x_recon
        projected = residual @ self._J
        signs = (projected >= 0).astype(mx.uint32)
        packed_signs = _pack_bits(signs, bits=1)

        return radii.astype(mx.float16), packed_quant, packed_signs

    def _decompress(self, radii, packed_quant, packed_signs) -> mx.array:
        """Decompress → [B, H, S, D] float16."""
        d = self._head_dim
        n_levels = (1 << self.kv_bits) - 1
        qr = self._quant_range

        # PolarQuant inverse
        x_quant = _unpack_bits(packed_quant, self.kv_bits, d)
        x_deq = (x_quant.astype(mx.float32) / n_levels) * (2 * qr) - qr
        x_hat = (x_deq * radii.astype(mx.float32)) @ self._R.T

        # QJL residual correction
        signs_f = _unpack_bits(packed_signs, 1, self._qjl_dim).astype(mx.float32) * 2 - 1
        residual_approx = signs_f @ self._J.T
        residual_approx = residual_approx * (qr / n_levels)

        return (x_hat + residual_approx).astype(mx.float16)

    # -- cache interface (matches KVCache) -----------------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, n_kv_heads, num_steps, head_dim = keys.shape

        if self._R is None:
            self._init_matrices(head_dim)

        prev = self.offset

        # Compress incoming tokens
        k_rad, k_pack, k_sign = self._compress(keys.astype(mx.float32))
        v_rad, v_pack, v_sign = self._compress(values.astype(mx.float32))

        # Grow buffers when needed
        need_alloc = self._key_radii is None or (prev + num_steps) > self._key_radii.shape[2]
        if need_alloc:
            n_alloc = ((self.step + num_steps - 1) // self.step) * self.step

            def _grow(existing: Optional[mx.array], template: mx.array, last_dim: int):
                buf = mx.zeros((B, n_kv_heads, n_alloc, last_dim), dtype=template.dtype)
                if existing is not None:
                    trimmed = existing[..., :prev, :] if prev % self.step != 0 else existing
                    return mx.concatenate([trimmed, buf], axis=2)
                return buf

            pack_d = k_pack.shape[-1]
            sign_d = k_sign.shape[-1]

            self._key_radii = _grow(self._key_radii, k_rad, 1)
            self._key_packed = _grow(self._key_packed, k_pack, pack_d)
            self._key_signs = _grow(self._key_signs, k_sign, sign_d)
            self._val_radii = _grow(self._val_radii, v_rad, 1)
            self._val_packed = _grow(self._val_packed, v_pack, pack_d)
            self._val_signs = _grow(self._val_signs, v_sign, sign_d)

        self.offset += num_steps

        # Write compressed data into buffer
        self._key_radii[..., prev : self.offset, :] = k_rad
        self._key_packed[..., prev : self.offset, :] = k_pack
        self._key_signs[..., prev : self.offset, :] = k_sign
        self._val_radii[..., prev : self.offset, :] = v_rad
        self._val_packed[..., prev : self.offset, :] = v_pack
        self._val_signs[..., prev : self.offset, :] = v_sign

        # Decompress full cache for attention
        full_keys = self._decompress(
            self._key_radii[..., : self.offset, :],
            self._key_packed[..., : self.offset, :],
            self._key_signs[..., : self.offset, :],
        )
        full_values = self._decompress(
            self._val_radii[..., : self.offset, :],
            self._val_packed[..., : self.offset, :],
            self._val_signs[..., : self.offset, :],
        )
        return full_keys, full_values

    # -- bookkeeping ---------------------------------------------------------

    def size(self):
        return self.offset

    @property
    def state(self):
        if self._key_radii is None:
            return []
        return [
            self._key_radii[..., : self.offset, :],
            self._key_packed[..., : self.offset, :],
            self._key_signs[..., : self.offset, :],
            self._val_radii[..., : self.offset, :],
            self._val_packed[..., : self.offset, :],
            self._val_signs[..., : self.offset, :],
        ]

    @state.setter
    def state(self, v):
        if not v:
            return
        (
            self._key_radii, self._key_packed, self._key_signs,
            self._val_radii, self._val_packed, self._val_signs,
        ) = v
        self.offset = self._key_radii.shape[2]

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.offset, self.kv_bits, self.qjl_ratio, self.seed,
                      self._head_dim, self._qjl_dim))
        )

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])
        self.kv_bits = int(v[1])
        self.qjl_ratio = float(v[2])
        self.seed = int(v[3])
        hd, qd = int(v[4]), int(v[5])
        self._head_dim = hd
        self._qjl_dim = qd
        if hd > 0:
            self._init_matrices(hd)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self._key_radii is None

    @property
    def nbytes(self):
        if self._key_radii is None:
            return 0
        return sum(
            a[..., : self.offset, :].nbytes
            for a in (
                self._key_radii, self._key_packed, self._key_signs,
                self._val_radii, self._val_packed, self._val_signs,
            )
        )

    @property
    def uncompressed_nbytes(self):
        if not self._head_dim:
            return 0
        B, H = self._key_radii.shape[:2]
        return 2 * B * H * self.offset * self._head_dim * 2

    @property
    def compression_ratio(self) -> float:
        nb = self.nbytes
        return self.uncompressed_nbytes / nb if nb else 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_turboquant_cache(
    model: Any,
    kv_bits: int = 3,
    qjl_ratio: float = 0.5,
) -> list:
    """Create a prompt cache, replacing standard KV caches with TurboQuant.

    Respects the model's ``make_cache()`` when present — only swaps out
    ``KVCache`` instances, leaving other cache types (e.g. ``ArraysCache``
    for linear/SSM layers) untouched.
    """
    from mlx_lm.models.cache import KVCache as _KVCache

    if hasattr(model, "make_cache"):
        base_cache = model.make_cache()
    else:
        base_cache = [_KVCache() for _ in range(len(model.layers))]

    replaced = 0
    result = []
    for i, c in enumerate(base_cache):
        if isinstance(c, _KVCache):
            result.append(
                TurboQuantKVCache(kv_bits=kv_bits, qjl_ratio=qjl_ratio, seed=42 + i)
            )
            replaced += 1
        else:
            result.append(c)

    log.info(
        "TurboQuant KV cache: %d/%d layers replaced, %d-bit PolarQuant, QJL ratio %.1f",
        replaced, len(result), kv_bits, qjl_ratio,
    )
    return result
