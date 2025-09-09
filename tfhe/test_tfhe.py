import pytest
import numpy as np
import sys
import os

# Adiciona o diretório atual ao path para importar main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import deve vir depois da configuração do path
from tfhe_main import trivial_tlwe, phase_s, k, N, secret


class TestTFHEExample:
    """Teste de exemplo para TFHE"""

    def test_trivial_tlwe_example(self):
        """Teste de exemplo: cria TLWE trivial e verifica a fase"""
        # Mensagem de teste
        message = 0.5

        # Cria TLWE trivial
        tlwe_sample = trivial_tlwe(message)

        # Verifica estrutura básica
        assert tlwe_sample.shape == (k + 1, N)

        # Computa a fase (descriptografia)
        phase = phase_s(tlwe_sample, secret)

        # Para TLWE trivial, a fase deve ser aproximadamente a mensagem original
        assert np.isclose(phase[0], message, atol=1e-10)

        print(f"✓ Teste passou: mensagem {message} foi recuperada como {phase[0]}")


if __name__ == "__main__":
    # Permite executar os testes diretamente
    pytest.main([__file__, "-v"])
