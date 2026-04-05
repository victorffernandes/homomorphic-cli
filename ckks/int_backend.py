import numpy as np


"""
Backend helpers for fixed-width integer arithmetic in CKKS.

Design goals:
- Prefer int128 where available for fixed-width arrays.
- Never introduce new int64-based fast paths; fall back to Python ints instead.
"""


try:
    INT128_DTYPE = np.dtype("int128")
    HAS_INT128 = True
except TypeError:
    INT128_DTYPE = None
    HAS_INT128 = False


# Conceptual word size for design and documentation.
WORD_BITS = 128 if HAS_INT128 else 0


def make_int_backend_array(values, copy: bool = True):
    """
    Create an array backed by int128 when available, otherwise Python ints.
    """
    if HAS_INT128:
        return np.array(values, dtype=INT128_DTYPE, copy=copy)
    return np.array([int(v) for v in values], dtype=object, copy=copy)


def cast_array_to_backend(arr: np.ndarray):
    """
    Cast an existing array to the backend integer representation.
    """
    if HAS_INT128:
        return arr.astype(INT128_DTYPE)
    return np.array([int(v) for v in arr], dtype=object)


def rand_int_mod_backend(size, modulus: int):
    """
    Generate random integers in [0, modulus) using the backend representation.
    """
    q_int = int(modulus)
    if q_int <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")

    if HAS_INT128:
        return np.random.randint(0, q_int, size=size, dtype=INT128_DTYPE)

    # Fallback: Python bigints
    import math
    import random

    k = max(1, int(math.ceil(math.log2(q_int))))
    vals = [random.getrandbits(k) % q_int for _ in range(size)]
    return np.array(vals, dtype=object)


def round_to_backend(values, scale: float):
    """
    Multiply values by scale, round to nearest integer, and store using backend dtype.
    """
    vals = np.array(values, dtype=np.float64)
    rounded = np.round(vals * scale)
    return cast_array_to_backend(rounded)

