from .constants import TFHECryptographicParameters
import pytest
import numpy as np
import sys
import os

# Adiciona o diretório atual ao path para importar main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import deve vir depois da configuração do path
try:
    from .tfhe_main import trivial_tlwe, phase_s, secret
except ImportError as e:
    print(f"Erro ao importar módulos TFHE: {e}")
    raise


class TestTFHEExample:
    """Teste de exemplo para TFHE"""

    def test_trivial_tlwe_example(self):
        """Teste de exemplo: cria TLWE trivial e verifica a fase"""
        # Instância dos parâmetros criptográficos
        crypto_params = TFHECryptographicParameters()

        # Mensagem de teste
        message = 0.5

        # Cria TLWE trivial
        tlwe_sample = trivial_tlwe(message)

        # Verifica estrutura básica
        expected_shape = (
            crypto_params.LWE_DIMENSION + 1,
            crypto_params.POLYNOMIAL_DEGREE,
        )
        assert tlwe_sample.shape == expected_shape

        # Computa a fase (descriptografia)
        phase = phase_s(tlwe_sample, secret)

        # Para TLWE trivial, a fase deve ser aproximadamente a mensagem original
        assert np.isclose(phase[0], message, atol=1e-10)

        print(f"✓ Teste passou: mensagem {message} foi recuperada como {phase[0]}")


if __name__ == "__main__":
    # Permite executar os testes diretamente
    pytest.main([__file__, "-v"])
