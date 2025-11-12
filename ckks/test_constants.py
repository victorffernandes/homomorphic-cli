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

    def test_parameter_validation(self):
        """Teste para verificar a validação dos parâmetros"""
        # Deve passar sem exceções
        crypto_params = CKKSCryptographicParameters()
        crypto_params.validate_parameters()


class TestModCentered:
    """Testes para a função mod_centered que implementa ℤ_a = (-a/2, a/2]"""

    def test_mod_centered_inside_interval(self):
        """Testa valores dentro do intervalo (-a/2, a/2]"""
        crypto_params = CKKSCryptographicParameters()

        # Para modulus=10, o intervalo é (-5, 5]
        assert crypto_params.mod_centered(3, 10) == 3
        assert crypto_params.mod_centered(-3, 10) == -3
        assert crypto_params.mod_centered(5, 10) == 5  # Limite superior incluído
        assert crypto_params.mod_centered(0, 10) == 0

    def test_mod_centered_greater_than_modulus(self):
        """Testa valores maiores que o módulo que precisam ser reduzidos"""
        crypto_params = CKKSCryptographicParameters()

        # Valores > a/2 devem ser reduzidos para o intervalo (-a/2, a/2]
        assert crypto_params.mod_centered(7, 10) == -3  # 7 - 10 = -3
        assert crypto_params.mod_centered(23, 10) == 3  # 23 mod 10 = 3
        assert (
            crypto_params.mod_centered(28, 10) == -2
        )  # 28 mod 10 = 8, então 8 - 10 = -2

    def test_mod_centered_smaller_than_negative_half(self):
        """Testa valores menores que -a/2 que precisam ser ajustados"""
        crypto_params = CKKSCryptographicParameters()

        # Valores <= -a/2 devem ser ajustados para o intervalo (-a/2, a/2]
        assert crypto_params.mod_centered(-6, 10) == 4  # -6 + 10 = 4
        assert (
            crypto_params.mod_centered(-23, 10) == -3
        )  # -23 mod 10 = 7, então 7 - 10 = -3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
