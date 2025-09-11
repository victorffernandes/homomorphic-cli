from .ckks_main import ckks_encode_real, ckks_decode_real
from .constants import CKKSCryptographicParameters
import pytest
import numpy as np
from numpy.polynomial import Polynomial


class TestCKKSExample:
    """Teste de exemplo para CKKS"""

    def test_encode_decode_example(self):
        """Teste de exemplo: codifica e decodifica um vetor real"""
        # Instância dos parâmetros criptográficos
        crypto_params = CKKSCryptographicParameters()

        # Vetor de teste simples
        max_slots = crypto_params.get_maximum_plaintext_slots()
        test_vector = np.array([0.5, -0.6, 0.7, 0.8] + [0] * (max_slots - 4))

        # Codifica o vetor em um polinômio
        encoded_poly = ckks_encode_real(
            test_vector, crypto_params.SCALING_FACTOR, crypto_params.POLYNOMIAL_DEGREE
        )

        # Verifica que o polinômio foi criado
        assert isinstance(encoded_poly, Polynomial)
        assert len(encoded_poly.coef) > 0

        # Decodifica o polinômio de volta para vetor
        decoded_vector = ckks_decode_real(
            encoded_poly,
            crypto_params.SCALING_FACTOR,
            crypto_params.POLYNOMIAL_DEGREE,
            crypto_params.get_initial_modulus(),
        )

        # Verifica propriedades básicas
        assert isinstance(decoded_vector, np.ndarray)
        assert len(decoded_vector) > 0
        assert np.allclose(decoded_vector[:3], test_vector[:3], atol=1e-5)

        print("✓ Teste passou: processo de encode/decode executado com sucesso")
        print(f"   Original: {test_vector}")
        print(f"   Resultado decode: {decoded_vector}")
        print(f"   Polinômio tem {len(encoded_poly.coef)} coeficientes")


if __name__ == "__main__":
    # Permite executar os testes diretamente
    pytest.main([__file__, "-v"])
