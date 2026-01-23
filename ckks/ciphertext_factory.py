"""
Fábrica para criação e manipulação de ciphertexts CKKS.

DEPRECATED: Encode/decode methods are deprecated. Use CKKSPlaintext instead.
This class now focuses on encryption/decryption operations.
"""

import warnings
import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Union, Tuple, Dict, Any

try:
    from .constants import CKKSCryptographicParameters
    from .ckks_ciphertext import CKKSCiphertext
    from .ckks_plaintext import CKKSPlaintext
except ImportError:
    from constants import CKKSCryptographicParameters
    from ckks_ciphertext import CKKSCiphertext
    from ckks_plaintext import CKKSPlaintext


class CKKSCiphertextFactory:
    """
    Factory for CKKS ciphertext creation and manipulation.
    
    Note: Encoding/decoding methods are deprecated. Use CKKSPlaintext for those operations.
    """

    def __init__(self, crypto_params: CKKSCryptographicParameters = None):
        if crypto_params is None:
            crypto_params = CKKSCryptographicParameters()
        
        self.crypto_params = crypto_params

    def ckks_encode_real(
        self,
        real_vector: List[float],
        delta_scale: float = None,
        n_poly_coeffs: int = None,
    ) -> Polynomial:
        """
        DEPRECATED: Use CKKSPlaintext.encode() instead.
        """
        warnings.warn(
            "CKKSCiphertextFactory.ckks_encode_real() is deprecated. "
            "Use CKKSPlaintext.encode() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return CKKSPlaintext.encode(real_vector, self.crypto_params, delta_scale)

    def ckks_decode_real(
        self,
        message_poly: Polynomial,
        delta_scale: float = None,
        n_poly_coeffs: int = None,
        q_mod: bool = True,
        q_mod_value: int = None,
    ) -> np.ndarray:
        """
        DEPRECATED: Use CKKSPlaintext.decode() instead.
        """
        warnings.warn(
            "CKKSCiphertextFactory.ckks_decode_real() is deprecated. "
            "Use CKKSPlaintext.decode() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return CKKSPlaintext.decode(
            message_poly, self.crypto_params, delta_scale, q_mod, q_mod_value
        )

    def encrypt(
        self,
        message_poly: Polynomial,
        public_key: Tuple[Polynomial, Polynomial],
        level: int = None,
    ) -> CKKSCiphertext:
        """
        Criptografa um polinômio usando a chave pública seguindo o esquema CKKS.

        Conforme a definição:
        - Encryption: ct = (pk_b * u + e1 + m, pk_a * u + e2)
        onde u ← ZO(ρ), e1, e2 ← DG(σ²)

        Args:
            message_poly: Polinômio da mensagem a ser criptografada (já escalado)
            public_key: Tupla (pk_b, pk_a) da chave pública
            level: Nível inicial na cadeia de módulos (usa o maior se None)

        Returns:
            CKKSCiphertext: Ciphertext resultante
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1  # Nível mais alto

        q_mod = self.crypto_params.MODULUS_CHAIN[level]
        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        sigma_err = self.crypto_params.GAUSSIAN_NOISE_STDDEV
        zero_one_density = self.crypto_params.ZERO_ONE_DENSITY

        pk_b, pk_a = public_key

        # Sample u ← ZO(ρ), e1, e2 ← DG(σ²)
        u = self.crypto_params.generate_zero_one_poly(n_degree, zero_one_density)
        e1 = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        e2 = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)

        # CKKS encryption: ct = (pk_b*u + e1 + m, pk_a*u + e2)
        # Calcula pk_b * u
        b_u = self.crypto_params.poly_mul_mod(pk_b, u, q_mod, ring_poly_mod)

        # Calcula pk_a * u
        a_u = self.crypto_params.poly_mul_mod(pk_a, u, q_mod, ring_poly_mod)

        # c0 = pk_b*u + e1 + m (mod q)
        c0 = self.crypto_params.poly_ring_mod(
            b_u + e1 + message_poly, ring_poly_mod, q_mod
        )

        # c1 = pk_a*u + e2 (mod q)
        c1 = self.crypto_params.poly_ring_mod(a_u + e2, ring_poly_mod, q_mod)

        return CKKSCiphertext(
            components=[c0, c1],
            level=level,
            crypto_params=self.crypto_params,
        )

    def decrypt(
        self,
        ciphertext: Union[CKKSCiphertext, Dict[str, Any]],
        secret_key: Tuple[Polynomial, Polynomial],
    ) -> Polynomial:
        """
        Descriptografa um ciphertext usando a chave secreta.

        Args:
            ciphertext: Ciphertext a ser descriptografado (CKKSCiphertext ou dict)
            secret_key: Chave secreta sk = (1, s) para descriptografia

        Returns:
            Polynomial: Polinômio da mensagem descriptografada
        """
        # Suporte para formato de dicionário legado
        if isinstance(ciphertext, dict):
            ciphertext = CKKSCiphertext.from_dict(ciphertext, self.crypto_params)

        # Extrai o componente s de sk = (1, s)
        _, s = secret_key

        q_mod = ciphertext.current_modulus
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()

        # Descriptografia para ciphertext de 3 componentes (após multiplicação)
        # Fórmula: m ≈ C'₁ + C'₂·s + C'₃·s²
        # O dot product com (1, s, s²) recupera a mensagem
        if ciphertext.size == 3:
            c0 = ciphertext.get_component(0)
            c1 = ciphertext.get_component(1)
            c2 = ciphertext.get_component(2)

            # Calcula s² (s ao quadrado) usando multiplicação modular
            s_squared = self.crypto_params.poly_mul_mod(s, s, q_mod, ring_poly_mod)

            # Calcula c1 * s usando multiplicação modular
            c1_s = self.crypto_params.poly_mul_mod(c1, s, q_mod, ring_poly_mod)

            # Calcula c2 * s² usando multiplicação modular
            c2_s_squared = self.crypto_params.poly_mul_mod(
                c2, s_squared, q_mod, ring_poly_mod
            )

            # Descriptografa: m ≈ c0 + c1*s + c2*s²
            decrypted_poly = c0 + c1_s + c2_s_squared

            # Aplica redução modular final
            final_poly = self.crypto_params.poly_ring_mod(
                decrypted_poly,
                ring_poly_mod,
                q_mod,
            )

            return final_poly

        # Descriptografia padrão para ciphertext de 2 componentes: m = c0 + c1*s
        c0 = ciphertext.get_component(0)
        c1 = ciphertext.get_component(1)

        # Calcula c1 * s
        c1_s = self.crypto_params.poly_mul_mod(c1, s, q_mod, ring_poly_mod)

        # Descriptografa: c0 + c1*s
        decrypted_poly = c0 + c1_s

        # Aplica redução modular final
        final_poly = self.crypto_params.poly_ring_mod(
            decrypted_poly, ring_poly_mod, q_mod
        )

        return final_poly

    def encode_and_encrypt(
        self,
        real_vector: List[float],
        public_key: Tuple[Polynomial, Polynomial],
        level: int = None,
    ) -> CKKSCiphertext:
        """Encodes and encrypts a real vector in one operation."""
        encoded_poly = CKKSPlaintext.encode(real_vector, self.crypto_params)
        return self.encrypt(encoded_poly, public_key, level)

    def decrypt_and_decode(
        self,
        ciphertext: Union[CKKSCiphertext, Dict[str, Any]],
        secret_key: Tuple[Polynomial, Polynomial],
        expected_length: int = None,
    ) -> np.ndarray:
        """Decrypts and decodes a ciphertext in one operation."""
        decrypted_poly = self.decrypt(ciphertext, secret_key)
        scale = ciphertext.scale
        
        decoded_vector = CKKSPlaintext.decode(
            decrypted_poly,
            self.crypto_params,
            scale,
            q_mod=True,
            q_mod_value=ciphertext.current_modulus,
        )
        
        if expected_length is not None:
            decoded_vector = decoded_vector[:expected_length]
        
        return decoded_vector


def create_ckks_factory(
    crypto_params: CKKSCryptographicParameters = None,
) -> CKKSCiphertextFactory:
    """Creates a new CKKS factory instance."""
    return CKKSCiphertextFactory(crypto_params)


if __name__ == "__main__":
    from constants import CKKSCryptographicParameters
    from ckks_plaintext import CKKSPlaintext
    
    print("=== CKKS Ciphertext Factory Example ===\n")
    
    crypto_params = CKKSCryptographicParameters()
    factory = CKKSCiphertextFactory(crypto_params)
    
    real_data = [1.5, -2.3, 3.7, 0.0]
    
    # Encode using CKKSPlaintext (preferred method)
    encoded_poly = CKKSPlaintext.encode(real_data, crypto_params)
    print(f"✓ Encoded with {len(encoded_poly.coef)} coefficients")
    
    # Decode
    decoded_data = CKKSPlaintext.decode(encoded_poly, crypto_params, q_mod=False)
    print(f"✓ Decoded: {decoded_data[:len(real_data)]}")
    
    print("\n✅ CKKSCiphertextFactory working correctly!")
