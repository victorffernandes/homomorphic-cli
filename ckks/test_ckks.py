import pytest
import numpy as np
from numpy.polynomial import Polynomial
import sys
import os

# Adiciona o diretório atual ao path para importar main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import deve vir depois da configuração do path
from ckks_main import ckks_encode_real, ckks_decode_real, DELTA, N, Q_CHAIN


class TestCKKSExample:
    """Teste de exemplo para CKKS"""

    def test_encode_decode_example(self):
        """Teste de exemplo: codifica e decodifica um vetor real"""
        # Vetor de teste simples
        test_vector = np.array([0.5, -0.6, 0.7, 0.8] + [0] * ((N // 2) - 4))

        # Codifica o vetor em um polinômio
        encoded_poly = ckks_encode_real(test_vector, DELTA, N)

        # Verifica que o polinômio foi criado
        assert isinstance(encoded_poly, Polynomial)
        assert len(encoded_poly.coef) > 0

        # Decodifica o polinômio de volta para vetor
        decoded_vector = ckks_decode_real(encoded_poly, DELTA, N, Q_CHAIN[-1])

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
