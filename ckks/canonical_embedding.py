"""
Canonical embedding for CKKS using HEAAN's special FFT with cyclotomic roots.

Replaces rfft/irfft with the correct embedding for R = Z[X]/(X^N + 1).
Uses the 5-power subgroup of Z_{2N}^* for root indexing.
"""

import numpy as np
from typing import Tuple


def compute_rot_group(M: int, Nh: int) -> np.ndarray:
    """
    Compute rotGroup: rotGroup[j] = 5^j mod M for j = 0..Nh-1.

    Args:
        M: 2N (ring parameter)
        Nh: N/2 (number of slots)

    Returns:
        Array of length Nh
    """
    rot_group = np.zeros(Nh, dtype=np.int64)
    five_pows = 1
    for j in range(Nh):
        rot_group[j] = five_pows
        five_pows = (five_pows * 5) % M
    return rot_group


def compute_ksi_pows(M: int) -> np.ndarray:
    """
    Compute ksiPows: ksiPows[j] = exp(2πi j/M) for j = 0..M.

    Args:
        M: 2N (ring parameter)

    Returns:
        Complex array of length M+1 (index M used as ksiPows[M] = ksiPows[0])
    """
    j_vals = np.arange(M + 1, dtype=np.float64)
    return np.exp(2j * np.pi * j_vals / M)


def _bit_reverse(vals: np.ndarray, size: int) -> None:
    """
    In-place Cooley-Tukey bit reversal permutation.

    Args:
        vals: Complex array (modified in place)
        size: Size of the FFT (power of 2)
    """
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
        if i < j:
            vals[i], vals[j] = vals[j].copy(), vals[i].copy()


def fft_special(vals: np.ndarray, rot_group: np.ndarray, ksi_pows: np.ndarray, M: int) -> None:
    """
    Forward special FFT (in-place). Used for decode (polynomial coeffs -> slot values).

    Args:
        vals: Complex array of length Nh (size), modified in place
        rot_group: From compute_rot_group(M, Nh)
        ksi_pows: From compute_ksi_pows(M)
        M: 2N
    """
    size = len(vals)
    _bit_reverse(vals, size)

    len_val = 2
    while len_val <= size:
        lenh = len_val >> 1
        lenq = len_val << 2
        for i in range(0, size, len_val):
            for j in range(lenh):
                idx = int((rot_group[j] % lenq) * M / lenq)
                u = vals[i + j].copy()
                v = vals[i + j + lenh] * ksi_pows[idx]
                vals[i + j] = u + v
                vals[i + j + lenh] = u - v
        len_val <<= 1


def fft_special_inv_lazy(
    vals: np.ndarray, rot_group: np.ndarray, ksi_pows: np.ndarray, M: int
) -> None:
    """
    Inverse special FFT without normalization (in-place).
    fft_special(fft_special_inv_lazy(vals)) / size == vals.

    Args:
        vals: Complex array of length Nh, modified in place
        rot_group: From compute_rot_group(M, Nh)
        ksi_pows: From compute_ksi_pows(M)
        M: 2N
    """
    size = len(vals)
    len_val = size
    while len_val >= 2:
        lenh = len_val >> 1
        lenq = len_val << 2
        for i in range(0, size, len_val):
            for j in range(lenh):
                idx = int((lenq - (rot_group[j] % lenq)) * M / lenq)
                u = vals[i + j] + vals[i + j + lenh]
                v = (vals[i + j] - vals[i + j + lenh]) * ksi_pows[idx]
                vals[i + j] = u
                vals[i + j + lenh] = v
        len_val >>= 1
    _bit_reverse(vals, size)


def fft_special_inv(
    vals: np.ndarray, rot_group: np.ndarray, ksi_pows: np.ndarray, M: int
) -> None:
    """
    Inverse special FFT with normalization by 1/size (in-place).
    fft_special(fft_special_inv(vals)) == vals.

    Args:
        vals: Complex array of length Nh, modified in place
        rot_group: From compute_rot_group(M, Nh)
        ksi_pows: From compute_ksi_pows(M)
        M: 2N
    """
    fft_special_inv_lazy(vals, rot_group, ksi_pows, M)
    vals /= len(vals)


def get_fft_tables(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get rot_group and ksi_pows for given polynomial degree N.

    Args:
        N: Polynomial degree (power of 2)

    Returns:
        (rot_group, ksi_pows)
    """
    M = N << 1
    Nh = N >> 1
    rot_group = compute_rot_group(M, Nh)
    ksi_pows = compute_ksi_pows(M)
    return rot_group, ksi_pows
