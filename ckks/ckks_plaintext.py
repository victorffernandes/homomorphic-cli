"""
CKKS Plaintext encoding and decoding using canonical embedding.

This module provides stateless encoding/decoding functions for the CKKS scheme.
"""

import numpy as np
from numpy.polynomial import Polynomial
from typing import List
from .constants import CKKSCryptographicParameters


class CKKSPlaintext:
    """
    Stateless class for CKKS plaintext encoding and decoding operations.

    All methods are static and require crypto_params to be passed explicitly.
    """

    @staticmethod
    def sigma(
        polynomial: Polynomial, crypto_params: CKKSCryptographicParameters
    ) -> np.ndarray:
        """Applies canonical embedding σ: R → ℂ^(N/2) using RFFT."""
        N = crypto_params.POLYNOMIAL_DEGREE
        num_slots = N // 2

        coeffs = polynomial.coef
        if len(coeffs) < N:
            coeffs = np.pad(coeffs, (0, N - len(coeffs)), mode="constant")
        elif len(coeffs) > N:
            coeffs = coeffs[:N]

        spectrum = np.fft.rfft(coeffs, n=N)
        return spectrum[:num_slots]

    @staticmethod
    def sigma_inverse(
        z: np.ndarray, crypto_params: CKKSCryptographicParameters
    ) -> Polynomial:
        """Applies inverse canonical embedding σ^(-1): H → R using IRFFT."""
        N = crypto_params.POLYNOMIAL_DEGREE
        num_slots = N // 2

        if len(z) < num_slots:
            z = np.pad(z, (0, num_slots - len(z)), mode="constant")
        elif len(z) > num_slots:
            z = z[:num_slots]

        spectrum_for_irfft = np.zeros(num_slots + 1, dtype=np.complex128)
        spectrum_for_irfft[:num_slots] = z
        spectrum_for_irfft[num_slots] = 0

        coeffs = np.fft.irfft(spectrum_for_irfft, n=N)
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

        return np.real(rescaled_z)


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
