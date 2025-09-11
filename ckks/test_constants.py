from .constants import CKKSCryptographicParameters
import pytest


class TestCKKSMain:
    """Testes básicos para o módulo CKKS principal"""

    def test_parameters(self):
        """Teste para verificar se os parâmetros estão definidos corretamente"""
        crypto_params = CKKSCryptographicParameters()

        assert crypto_params.POLYNOMIAL_DEGREE == 2048
        assert isinstance(crypto_params.MODULUS_CHAIN, list)
        assert len(crypto_params.MODULUS_CHAIN) > 0
        assert crypto_params.SCALING_FACTOR > 0
        assert crypto_params.GAUSSIAN_NOISE_STDDEV > 0

    def test_q_chain_values(self):
        """Teste para verificar os valores da cadeia de módulos"""
        crypto_params = CKKSCryptographicParameters()
        expected_q_chain = [1099511922689, 1099512004609, 1099512037377]
        assert crypto_params.MODULUS_CHAIN == expected_q_chain

    def test_delta_value(self):
        """Teste para verificar o valor de DELTA"""
        crypto_params = CKKSCryptographicParameters()
        assert crypto_params.SCALING_FACTOR == 1099511922688

    def test_parameter_validation(self):
        """Teste para verificar a validação dos parâmetros"""
        # Deve passar sem exceções
        CKKSCryptographicParameters.validate_parameters()

    def test_helper_methods(self):
        """Teste para verificar os métodos auxiliares"""
        crypto_params = CKKSCryptographicParameters()

        # Testa métodos de acesso
        assert crypto_params.get_initial_modulus() == 1099512037377
        assert crypto_params.get_maximum_plaintext_slots() == 1024
        assert crypto_params.get_scaling_factor_squared() == 1099511922688**2

        # Testa criação do anel polinomial
        poly_ring = crypto_params.get_polynomial_modulus_ring()
        assert poly_ring is not None
        assert len(poly_ring.coef) == 2049  # X^2048 + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
