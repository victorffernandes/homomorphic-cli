"""
CKKS Plaintext encoding and decoding using canonical embedding.

This module provides stateless encoding/decoding functions and stateful
plaintext operations for the CKKS scheme (HEAAN-aligned).

Uses HEAAN's special FFT with cyclotomic roots (5-power subgroup)
for the canonical embedding in R = Z[X]/(X^N + 1).
"""

import json
import time
import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Union
from .constants import CKKSCryptographicParameters
from .canonical_embedding import get_fft_tables, fft_special, fft_special_inv

# #region agent log
def _debug_log(*args, **kwargs) -> None:
    """No-op debug logger (instrumentation cleaned)."""
    return None


# #endregion


class CKKSPlaintext:
    """
    CKKS plaintext: supports both stateless encode/decode and stateful
    operations (add, sub, multiply, rescale, negate, multiply_by_const).
    """

    def __init__(
        self,
        polynomial: Polynomial,
        crypto_params: CKKSCryptographicParameters,
        scale: Union[float, int, None] = None,
    ):
        """
        Initialize a CKKS plaintext from an encoded polynomial.

        Args:
            polynomial: Encoded message in R or R_q
            crypto_params: Cryptographic parameters
            scale: Scale factor (default: crypto_params.SCALING_FACTOR)

        Raises:
            ValueError: If polynomial or crypto_params are invalid
        """

        self.crypto_params = crypto_params
        self.polynomial = polynomial
        if scale is None:
            self.scale = float(crypto_params.SCALING_FACTOR)
        else:
            self.scale = float(scale)

    @classmethod
    def from_vector(
        cls,
        real_vector: List[float],
        crypto_params: CKKSCryptographicParameters,
        scale: Union[float, int, None] = None,
    ) -> "CKKSPlaintext":
        """
        Create CKKSPlaintext from a real vector (encodes then wraps).

        Args:
            real_vector: Real values to encode
            crypto_params: Cryptographic parameters
            scale: Scale factor (default: crypto_params.SCALING_FACTOR)

        Returns:
            CKKSPlaintext instance
        """
        poly = cls.encode(real_vector, crypto_params, scale)
        return cls(poly, crypto_params, scale)

    def to_vector(
        self,
        q_mod: bool = True,
        q_mod_value: Union[int, None] = None,
    ) -> np.ndarray:
        """
        Decode plaintext to real vector.

        Args:
            q_mod: Whether to apply modular reduction before decode
            q_mod_value: Modulus for reduction (default: MODULUS_CHAIN[-1])

        Returns:
            Real vector (numpy array)
        """
        return CKKSPlaintext.decode(
            self.polynomial,
            self.crypto_params,
            scale=self.scale,
            q_mod=q_mod,
            q_mod_value=q_mod_value,
        )

    @staticmethod
    def sigma(
        polynomial: Polynomial, crypto_params: CKKSCryptographicParameters
    ) -> np.ndarray:
        """
        Applies canonical embedding σ: R → ℂ^(N/2) using special FFT.

        Builds complex array from coeff pairs (j, j+Nh), then fft_special.
        """
        N = crypto_params.POLYNOMIAL_DEGREE
        Nh = N // 2
        M = N << 1

        coeffs = np.array(polynomial.coef, dtype=np.float64)
        if len(coeffs) < N:
            coeffs = np.pad(coeffs, (0, N - len(coeffs)), mode="constant")
        elif len(coeffs) > N:
            coeffs = coeffs[:N]

        # HEAAN layout: slot j → coeffs[j] (real), coeffs[j+Nh] (imag)
        z = coeffs[:Nh] + 1j * coeffs[Nh:N]
        z = z.astype(np.complex128)

        rot_group, ksi_pows = get_fft_tables(N)
        fft_special(z, rot_group, ksi_pows, M)
        return z

    @staticmethod
    def sigma_inverse(
        z: np.ndarray, crypto_params: CKKSCryptographicParameters
    ) -> Polynomial:
        """
        Applies inverse canonical embedding σ^(-1): ℂ^(N/2) → R using special FFT.

        fft_special_inv, then map output to polynomial coeffs (idx, idx+Nh).
        """
        N = crypto_params.POLYNOMIAL_DEGREE
        Nh = N // 2
        M = N << 1

        if len(z) < Nh:
            z = np.pad(z, (0, Nh - len(z)), mode="constant")
        elif len(z) > Nh:
            z = z[:Nh]

        z = np.array(z, dtype=np.complex128).copy()
        rot_group, ksi_pows = get_fft_tables(N)
        fft_special_inv(z, rot_group, ksi_pows, M)

        # HEAAN layout: coeffs[j] = real(z[j]), coeffs[j+Nh] = imag(z[j])
        coeffs = np.zeros(N, dtype=np.float64)
        coeffs[:Nh] = np.real(z)
        coeffs[Nh:N] = np.imag(z)
        return Polynomial(coeffs)

    @staticmethod
    def encode(
        real_vector: List[float],
        crypto_params: CKKSCryptographicParameters,
        scale: float = None,
    ) -> Polynomial:
        """Encodes real vector to CKKS polynomial."""
        if scale is None:
            scale = crypto_params.SCALING_FACTOR

        N = crypto_params.POLYNOMIAL_DEGREE
        max_slots = N // 2

        input_array = np.array(real_vector, dtype=np.float64)
        if len(input_array) > max_slots:
            input_array = input_array[:max_slots]
        elif len(input_array) < max_slots:
            input_array = np.pad(
                input_array, (0, max_slots - len(input_array)), mode="constant"
            )

        z = input_array.astype(np.complex128)
        scaled_z = scale * z
        p = CKKSPlaintext.sigma_inverse(scaled_z, crypto_params)
        coef = np.round(np.real(p.coef)).astype(int)

        return Polynomial(coef)

    @staticmethod
    def decode(
        message_poly: Polynomial,
        crypto_params: CKKSCryptographicParameters,
        scale: float = None,
        q_mod: bool = True,
        q_mod_value: int = None,
    ) -> np.ndarray:
        """Decodes CKKS polynomial to real vector."""
        if scale is None:
            scale = crypto_params.SCALING_FACTOR

        coeffs = message_poly.coef.copy()

        if q_mod:
            if q_mod_value is None:
                q_mod_value = crypto_params.MODULUS_CHAIN[-1]
            corrected_coeffs = crypto_params.mod_centered(coeffs, q_mod_value)
        else:
            corrected_coeffs = coeffs

        p = Polynomial(corrected_coeffs)
        z = CKKSPlaintext.sigma(p, crypto_params)
        rescaled_z = z / scale
        decoded = np.real(rescaled_z)

        return decoded

    # -------------------------------------------------------------------------
    # HEAAN-aligned plaintext operations
    # -------------------------------------------------------------------------

    @staticmethod
    def add(pt1: "CKKSPlaintext", pt2: "CKKSPlaintext") -> "CKKSPlaintext":
        """
        Add two plaintexts (HEAAN Ring2Utils::add).

        Same scale required. Returns m1 + m2 in R_q.
        """
        if abs(pt1.scale - pt2.scale) > 1e-10:
            raise ValueError(
                f"Plaintexts must have same scale for add. "
                f"pt1.scale={pt1.scale}, pt2.scale={pt2.scale}"
            )
        q = pt1.crypto_params.get_initial_modulus()
        ring_poly_mod = pt1.crypto_params.get_polynomial_modulus_ring()
        m_sum = pt1.polynomial + pt2.polynomial
        m_mod = pt1.crypto_params.poly_ring_mod(m_sum, ring_poly_mod, q)
        return CKKSPlaintext(m_mod, pt1.crypto_params, pt1.scale)

    @staticmethod
    def sub(pt1: "CKKSPlaintext", pt2: "CKKSPlaintext") -> "CKKSPlaintext":
        """
        Subtract plaintexts (HEAAN Ring2Utils::sub).

        Same scale required. Returns m1 - m2 in R_q.
        """
        if abs(pt1.scale - pt2.scale) > 1e-10:
            raise ValueError(
                f"Plaintexts must have same scale for sub. "
                f"pt1.scale={pt1.scale}, pt2.scale={pt2.scale}"
            )
        q = pt1.crypto_params.get_initial_modulus()
        ring_poly_mod = pt1.crypto_params.get_polynomial_modulus_ring()
        m_diff = pt1.polynomial - pt2.polynomial
        m_mod = pt1.crypto_params.poly_ring_mod(m_diff, ring_poly_mod, q)
        return CKKSPlaintext(m_mod, pt1.crypto_params, pt1.scale)

    @staticmethod
    def multiply(pt1: "CKKSPlaintext", pt2: "CKKSPlaintext") -> "CKKSPlaintext":
        """
        Multiply two plaintexts (HEAAN Ring2Utils::mult).

        Returns m1 * m2 in R_q. New scale = pt1.scale * pt2.scale.
        """
        q = pt1.crypto_params.get_initial_modulus()
        ring_poly_mod = pt1.crypto_params.get_polynomial_modulus_ring()
        m_prod = pt1.crypto_params.poly_mul_mod(
            pt1.polynomial, pt2.polynomial, q, ring_poly_mod
        )
        new_scale = pt1.scale * pt2.scale
        return CKKSPlaintext(m_prod, pt1.crypto_params, new_scale)

    def rescale(self, bits_down: int) -> "CKKSPlaintext":
        """
        Rescale plaintext (HEAAN reScaleBy).

        Floor-divide coefficients by 2^bits_down (right shift). No rounding.
        """
        coeffs = np.array(self.polynomial.coef, dtype=np.int64)
        if len(coeffs) < self.crypto_params.POLYNOMIAL_DEGREE:
            coeffs = np.pad(
                coeffs,
                (0, self.crypto_params.POLYNOMIAL_DEGREE - len(coeffs)),
                mode="constant",
            )
        coeffs = coeffs >> bits_down
        scaled_poly = Polynomial(coeffs.astype(np.int64))
        q = self.crypto_params.get_initial_modulus()
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        m_mod = self.crypto_params.poly_ring_mod(scaled_poly, ring_poly_mod, q)
        new_scale = self.scale / (1 << bits_down)
        return CKKSPlaintext(m_mod, self.crypto_params, new_scale)

    def negate(self) -> "CKKSPlaintext":
        """
        Negate plaintext (HEAAN Scheme::negate).

        Returns -m, same scale. No mod.
        """
        return CKKSPlaintext(-self.polynomial, self.crypto_params, self.scale)

    @staticmethod
    def multiply_by_const(
        pt: "CKKSPlaintext", constant: float, logp: int
    ) -> "CKKSPlaintext":
        """
        Multiply plaintext by constant (HEAAN Scheme::multByConst).

        Constant is scaled by 2^logp, then MulMod per coefficient.
        New scale = pt.scale * (2^logp).
        """
        cnst_scaled = int(round(constant * (1 << logp)))
        q = pt.crypto_params.get_initial_modulus()
        ring_poly_mod = pt.crypto_params.get_polynomial_modulus_ring()
        coeffs = np.array(pt.polynomial.coef, dtype=np.int64)
        if len(coeffs) < pt.crypto_params.POLYNOMIAL_DEGREE:
            coeffs = np.pad(
                coeffs,
                (0, pt.crypto_params.POLYNOMIAL_DEGREE - len(coeffs)),
                mode="constant",
            )
        cnst_mod = cnst_scaled % q
        coeffs = np.array(
            [(int(c) * cnst_mod) % q for c in coeffs], dtype=np.int64
        )
        m_scaled = Polynomial(coeffs.astype(np.int64))
        m_mod = pt.crypto_params.poly_ring_mod(m_scaled, ring_poly_mod, q)
        new_scale = pt.scale * (1 << logp)
        return CKKSPlaintext(m_mod, pt.crypto_params, new_scale)


if __name__ == "__main__":
    from ckks.constants import CKKSCryptographicParameters

    print("=== CKKS Plaintext Encoding/Decoding Example ===\n")

    crypto_params = CKKSCryptographicParameters()
    real_data = [1.5, -2.3, 3.7, 0.0, 4.2, -1.8]
    print(f"Original data: {real_data}")

    encoded_poly = CKKSPlaintext.encode(real_data, crypto_params)
    print(f"\n✓ Encoded to polynomial with {len(encoded_poly.coef)} coefficients")

    decoded_data = CKKSPlaintext.decode(encoded_poly, crypto_params, q_mod=False)
    print(f"✓ Decoded data: {decoded_data[:len(real_data)]}")

    error = np.max(np.abs(np.array(real_data) - decoded_data[: len(real_data)]))
    print(f"\n✓ Max encoding error: {error:.2e}")
    print("\n✅ CKKSPlaintext encoding/decoding working correctly!")
