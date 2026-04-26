"""
DCT-based image compression (JPEG-style baseline).

Implements the 2D DCT-II from first principles, JPEG quantization with a
quality parameter, inverse pipeline, and reconstruction quality metrics.

Author: Ved
Course: IC2306 Signal and Image Processing
"""

from __future__ import annotations

import numpy as np


BLOCK = 8


JPEG_LUMA_Q50 = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float64)


def _alpha(k: int) -> float:
    """DCT-II normalization: a(0) = 1/sqrt(2), a(k>0) = 1."""
    return 1.0 / np.sqrt(2.0) if k == 0 else 1.0


def build_dct_matrix(n: int = BLOCK) -> np.ndarray:
    """
    Orthonormal DCT-II matrix T such that F = T @ f @ T.T computes the 2D DCT.

    T[k, i] = a(k) * sqrt(2/n) * cos( (2i+1) k pi / (2n) )
    """
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    T = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    T *= np.sqrt(2.0 / n)
    T[0, :] *= 1.0 / np.sqrt(2.0)
    return T


T = build_dct_matrix(BLOCK)


def dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II of an 8x8 block via the separable matrix form."""
    return T @ block @ T.T


def idct2(coeffs: np.ndarray) -> np.ndarray:
    """2D inverse DCT-II (orthonormal, so inverse = transpose)."""
    return T.T @ coeffs @ T


def quality_to_matrix(quality: int) -> np.ndarray:
    """
    Scale the JPEG Q50 matrix to a target quality factor (1..100).
    Standard IJG scaling: lower quality -> coarser quantization.
    """
    quality = int(np.clip(quality, 1, 100))
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    Q = np.floor((JPEG_LUMA_Q50 * scale + 50.0) / 100.0)
    Q = np.clip(Q, 1, 255)
    return Q.astype(np.float64)


def _pad_to_block(img: np.ndarray, n: int = BLOCK) -> np.ndarray:
    """Pad with edge replication so H and W are multiples of n."""
    h, w = img.shape
    ph = (-h) % n
    pw = (-w) % n
    if ph == 0 and pw == 0:
        return img
    return np.pad(img, ((0, ph), (0, pw)), mode="edge")


def compress(img: np.ndarray, quality: int = 50) -> dict:
    """
    Run the full JPEG-baseline pipeline on a grayscale image.

    Steps
    -----
    1. Shift pixel range from [0, 255] to [-128, 127].
    2. Split into 8x8 blocks.
    3. DCT each block.
    4. Divide by the quality-scaled quantization matrix and round.
    5. Dequantize and IDCT.
    6. Un-shift and clip to [0, 255].

    Returns a dict with the reconstructed image, all intermediate tensors,
    and diagnostic metrics.
    """
    img = img.astype(np.float64)
    original_shape = img.shape
    padded = _pad_to_block(img, BLOCK)
    h, w = padded.shape

    shifted = padded - 128.0
    Q = quality_to_matrix(quality)

    dct_blocks = np.zeros_like(shifted)
    quantized = np.zeros_like(shifted)
    dequantized = np.zeros_like(shifted)
    reconstructed_shifted = np.zeros_like(shifted)

    for by in range(0, h, BLOCK):
        for bx in range(0, w, BLOCK):
            block = shifted[by:by + BLOCK, bx:bx + BLOCK]
            F = dct2(block)
            Fq = np.round(F / Q)
            Fdq = Fq * Q
            f_hat = idct2(Fdq)

            dct_blocks[by:by + BLOCK, bx:bx + BLOCK] = F
            quantized[by:by + BLOCK, bx:bx + BLOCK] = Fq
            dequantized[by:by + BLOCK, bx:bx + BLOCK] = Fdq
            reconstructed_shifted[by:by + BLOCK, bx:bx + BLOCK] = f_hat

    reconstructed = np.clip(reconstructed_shifted + 128.0, 0, 255)
    reconstructed = reconstructed[:original_shape[0], :original_shape[1]]

    nonzero = int(np.count_nonzero(quantized))
    total = quantized.size

    return {
        "original": img[:original_shape[0], :original_shape[1]],
        "reconstructed": reconstructed,
        "dct_blocks": dct_blocks,
        "quantized": quantized,
        "dequantized": dequantized,
        "quantization_matrix": Q,
        "quality": quality,
        "nonzero_coeffs": nonzero,
        "total_coeffs": total,
        "sparsity": 1.0 - nonzero / total,
    }


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    e = mse(a, b)
    if e == 0:
        return float("inf")
    return float(20.0 * np.log10(peak / np.sqrt(e)))


def estimate_bpp(quantized: np.ndarray) -> float:
    """
    Rough entropy-coded bits-per-pixel estimate.

    Uses the first-order Shannon entropy of the quantized coefficients as a
    lower bound on what Huffman / arithmetic coding would need per symbol.
    This mirrors what JPEG's entropy stage achieves in practice to within a
    small overhead.
    """
    values, counts = np.unique(quantized.astype(np.int32), return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return entropy


def compression_ratio(quantized: np.ndarray, original_bits_per_pixel: float = 8.0) -> float:
    return original_bits_per_pixel / max(estimate_bpp(quantized), 1e-6)
