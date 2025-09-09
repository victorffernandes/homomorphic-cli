from tfhe_main import k, N, alpha, secret
import pytest
import numpy as np
import sys
import os

# Adiciona o diretório atual ao path para importar main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import deve vir depois da configuração do path


class TestTFHEMain:
    """Testes básicos para o módulo TFHE principal"""

    def test_parameters(self):
        """Teste para verificar se os parâmetros estão definidos corretamente"""
        assert k == 1
        assert N == 4
        assert alpha == 0.005
        assert len(secret) == N
        assert all(s in [0, 1] for s in secret)

    def test_secret_key_binary(self):
        """Teste para verificar se a chave secreta é binária"""
        assert isinstance(secret, np.ndarray)
        assert secret.dtype in [np.int32, np.int64, int]
        for bit in secret:
            assert bit in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
