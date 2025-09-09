from ckks_main import N, Q_CHAIN, DELTA, SIGMA
import pytest
import sys
import os

# Adiciona o diretório atual ao path para importar main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import deve vir depois da configuração do path


class TestCKKSMain:
    """Testes básicos para o módulo CKKS principal"""

    def test_parameters(self):
        """Teste para verificar se os parâmetros estão definidos corretamente"""
        assert N == 2048
        assert isinstance(Q_CHAIN, list)
        assert len(Q_CHAIN) > 0
        assert DELTA > 0
        assert SIGMA > 0

    def test_q_chain_values(self):
        """Teste para verificar os valores da cadeia de módulos"""
        expected_q_chain = [1099511922689, 1099512004609, 1099512037377]
        assert Q_CHAIN == expected_q_chain

    def test_delta_value(self):
        """Teste para verificar o valor de DELTA"""
        assert DELTA == 1099511922688


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
