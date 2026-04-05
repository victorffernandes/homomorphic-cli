import math
import random
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
from .int_backend import make_int_backend_array, rand_int_mod_backend


def mod_centered(value: Any, modulus: int):
    """
    Versão independente de mod_centered que funciona para módulos grandes.
    Reduz para o intervalo (-modulus/2, modulus/2].
    """
    reduced = np.mod(value, modulus)
    half_modulus = modulus // 2

    if np.isscalar(reduced):
        if reduced > half_modulus:
            return reduced - modulus
        return reduced

    result = reduced.copy()
    mask = result > half_modulus
    result[mask] = result[mask] - modulus
    return result


def poly_coeffs_mod_q(p_numpy: Polynomial, q_coeff: int) -> Polynomial:
    """
    Aplica operação modular centrada aos coeficientes de um polinômio.

    Retorna coeficientes em ℤ_q = (-q/2, q/2], usando ints Python para módulos grandes.
    """
    raw_coeffs = p_numpy.coef.copy()
    coeffs = np.array([int(c) for c in raw_coeffs], dtype=object)

    coeffs = mod_centered(coeffs, int(q_coeff))

    # Always return using the backend integer representation (int128 when available).
    return Polynomial(make_int_backend_array(coeffs))


def poly_ring_mod(p_numpy, ring_poly_mod: Polynomial, q_coeff: int) -> Polynomial:
    """
    Redução modular no anel R_q = ℤ_q[X]/(X^N + 1) usando aritmética de precisão arbitrária.
    """
    degree = len(ring_poly_mod.coef) - 1

    raw = p_numpy.coef if hasattr(p_numpy, "coef") else p_numpy
    coeffs = np.array([int(c) for c in raw], dtype=object)

    if len(coeffs) < 2 * degree:
        pad = np.zeros(2 * degree - len(coeffs), dtype=object)
        coeffs = np.concatenate([coeffs, pad])
    elif len(coeffs) > 2 * degree:
        coeffs = coeffs[: 2 * degree]

    q_int = int(q_coeff)
    pp_low = np.array([int(c) % q_int for c in coeffs[:degree]], dtype=object)
    pp_high = np.array([int(c) % q_int for c in coeffs[degree: 2 * degree]], dtype=object)

    result_coeffs = np.array(
        [(int(lo) - int(hi)) % q_int for lo, hi in zip(pp_low, pp_high)],
        dtype=object,
    )

    # Use backend representation for coefficients (int128 when available).
    coeff_array = make_int_backend_array(result_coeffs)

    result_poly = Polynomial(coeff_array)
    return poly_coeffs_mod_q(result_poly, q_coeff)


def poly_mul_mod(p1: Polynomial, p2: Polynomial, q: int, ring_poly_mod: Polynomial) -> Polynomial:
    """
    Multiplicação de polinômios com redução modular no anel R_q, usando ints Python.
    """
    p1_int = Polynomial(np.array([int(c) for c in p1.coef], dtype=object))
    p2_int = Polynomial(np.array([int(c) for c in p2.coef], dtype=object))
    full_poly = p1_int * p2_int
    return poly_ring_mod(full_poly, ring_poly_mod, q)


def generate_uniform_random_poly(degree_n: int, q_bound: int) -> Polynomial:
    """
    Gera um polinômio com coeficientes uniformemente aleatórios em [0, q_bound).
    Suporta q_bound >> 2^63 usando ints Python.
    """
    q_int = int(q_bound)
    if q_int <= 0:
        raise ValueError(f"q_bound deve ser positivo, recebido: {q_bound}")

    coeffs = rand_int_mod_backend(degree_n, q_int)
    return Polynomial(coeffs)

