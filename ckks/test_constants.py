from .constants import CKKSCryptographicParameters
import pytest


class TestCKKSMain:
    """Testes básicos para o módulo CKKS principal"""

    def test_parameters(self):
        """Teste para verificar se os parâmetros estão definidos corretamente"""
        crypto_params = CKKSCryptographicParameters()

        assert isinstance(crypto_params.MODULUS_CHAIN, list)
        assert len(crypto_params.MODULUS_CHAIN) > 0
        assert crypto_params.SCALING_FACTOR > 0
        assert crypto_params.GAUSSIAN_NOISE_STDDEV > 0

    def test_parameter_validation(self):
        """Teste para verificar a validação dos parâmetros"""
        # Deve passar sem exceções
        crypto_params = CKKSCryptographicParameters()
        crypto_params.validate_parameters()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
