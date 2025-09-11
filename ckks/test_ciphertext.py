from .ckks import CKKSCiphertext
from .constants import CKKSCryptographicParameters
import pytest
from numpy.polynomial import Polynomial


class TestCKKSCiphertext:
    """Testes para a classe CKKSCiphertext"""

    def setup_method(self):
        """Configuração executada antes de cada teste"""
        self.crypto_params = CKKSCryptographicParameters()
        self.c0 = self.crypto_params.generate_uniform_random_poly()
        self.c1 = self.crypto_params.generate_uniform_random_poly()

    def test_basic_creation(self):
        """Teste de criação básica de ciphertext"""
        ct = CKKSCiphertext(
            components=[self.c0, self.c1],
            level=2,
            scale=self.crypto_params.SCALING_FACTOR,
        )

        assert ct.size == 2
        assert ct.level == 2
        assert ct.scale == self.crypto_params.SCALING_FACTOR
        assert ct.current_modulus == self.crypto_params.MODULUS_CHAIN[2]
        assert ct.is_fresh()

    def test_invalid_creation(self):
        """Teste de criação com parâmetros inválidos"""
        # Lista vazia de componentes
        with pytest.raises(
            ValueError, match="Lista de componentes não pode estar vazia"
        ):
            CKKSCiphertext([], 1, 1000.0)

        # Nível inválido
        with pytest.raises(ValueError, match="Nível deve estar entre"):
            CKKSCiphertext([self.c0], -1, 1000.0)

        with pytest.raises(ValueError, match="Nível deve estar entre"):
            CKKSCiphertext([self.c0], 10, 1000.0)

        # Escala inválida
        with pytest.raises(ValueError, match="Escala deve ser positiva"):
            CKKSCiphertext([self.c0], 1, -100.0)

    def test_from_dict_conversion(self):
        """Teste de conversão de/para dicionário"""
        original_dict = {"c0": self.c0, "c1": self.c1, "level": 1, "scale": 1000.0}

        # Criar a partir de dicionário
        ct = CKKSCiphertext.from_dict(original_dict)

        assert ct.size == 2
        assert ct.level == 1
        assert ct.scale == 1000.0

        # Converter de volta para dicionário
        result_dict = ct.to_dict()

        assert result_dict["level"] == 1
        assert result_dict["scale"] == 1000.0
        assert "c0" in result_dict
        assert "c1" in result_dict

    def test_compatibility_checks(self):
        """Teste de verificações de compatibilidade"""
        ct1 = CKKSCiphertext([self.c0, self.c1], 1, 1000.0)
        ct2 = CKKSCiphertext([self.c0, self.c1], 1, 1000.0)
        ct3 = CKKSCiphertext([self.c0, self.c1], 2, 1000.0)  # Nível diferente
        ct4 = CKKSCiphertext([self.c0, self.c1], 1, 2000.0)  # Escala diferente

        # Devem ser compatíveis para adição
        assert ct1.can_add_with(ct2)
        assert not ct1.can_add_with(ct3)  # Nível diferente
        assert not ct1.can_add_with(ct4)  # Escala diferente

        # Devem ser compatíveis para multiplicação
        assert ct1.can_multiply_with(ct2)
        assert not ct1.can_multiply_with(ct3)  # Nível diferente

        # Ciphertext no nível 0 não pode multiplicar (precisa de rescale)
        ct0_level = CKKSCiphertext([self.c0, self.c1], 0, 1000.0)
        assert not ct0_level.can_multiply_with(ct1)

    def test_component_access(self):
        """Teste de acesso aos componentes"""
        ct = CKKSCiphertext([self.c0, self.c1], 1, 1000.0)

        # Acesso válido
        comp0 = ct.get_component(0)
        comp1 = ct.get_component(1)

        assert isinstance(comp0, Polynomial)
        assert isinstance(comp1, Polynomial)

        # Acesso inválido
        with pytest.raises(IndexError):
            ct.get_component(2)

        with pytest.raises(IndexError):
            ct.get_component(-1)

    def test_copy_functionality(self):
        """Teste de funcionalidade de cópia"""
        original = CKKSCiphertext([self.c0, self.c1], 1, 1000.0)
        copy_ct = original.copy()

        # Deve ser independente
        assert copy_ct.level == original.level
        assert copy_ct.scale == original.scale
        assert copy_ct.size == original.size

        # Modificar a cópia não deve afetar o original
        copy_ct.level = 0
        assert original.level == 1

    def test_rescale_update(self):
        """Teste de atualização após rescale"""
        ct = CKKSCiphertext([self.c0, self.c1], 2, 1000.0)

        # Rescale válido
        ct.update_after_rescale(1, 500.0)
        assert ct.level == 1
        assert ct.scale == 500.0

        # Rescale inválido
        with pytest.raises(ValueError):
            ct.update_after_rescale(2, 250.0)  # Nível maior que atual

        with pytest.raises(ValueError):
            ct.update_after_rescale(-1, 250.0)  # Nível negativo

    def test_properties(self):
        """Teste de propriedades calculadas"""
        ct = CKKSCiphertext([self.c0, self.c1], 1, 1000.0)

        # Propriedades básicas
        assert ct.current_modulus == self.crypto_params.MODULUS_CHAIN[1]
        assert ct.noise_budget == 1
        assert ct.size == 2
        assert not ct.is_fresh()  # Não é fresh porque escala != SCALING_FACTOR

        # Ciphertext fresh
        fresh_ct = CKKSCiphertext(
            [self.c0, self.c1], 2, self.crypto_params.SCALING_FACTOR
        )
        assert fresh_ct.is_fresh()

    def test_string_representations(self):
        """Teste das representações string"""
        ct = CKKSCiphertext([self.c0, self.c1], 1, 1000.0)

        str_repr = str(ct)
        assert "CKKSCiphertext" in str_repr
        assert "size=2" in str_repr
        assert "level=1" in str_repr

        repr_str = repr(ct)
        assert "CKKSCiphertext" in repr_str
        assert "level=1/2" in repr_str

    def test_three_component_ciphertext(self):
        """Teste com ciphertext de 3 componentes (após multiplicação)"""
        c2 = self.crypto_params.generate_uniform_random_poly()

        ct = CKKSCiphertext([self.c0, self.c1, c2], 1, 1000.0)

        assert ct.size == 3

        # Conversão para dicionário deve incluir c2
        dict_format = ct.to_dict()
        assert "c2" in dict_format

        # Recriação a partir do dicionário
        ct_recreated = CKKSCiphertext.from_dict(dict_format)
        assert ct_recreated.size == 3

    def test_add_homomorphic_basic(self):
        """Teste de adição homomórfica básica"""
        import numpy as np
        from .factories import CKKSKeyFactory, CKKSCiphertextFactory

        # Setup das factories
        key_factory = CKKSKeyFactory(self.crypto_params)
        ciphertext_factory = CKKSCiphertextFactory(self.crypto_params)

        # Gera chaves
        keyset = key_factory.generate_full_keyset()
        sk = keyset["secret_key"]
        pk = keyset["public_key"]

        # Vetores de teste
        m1 = np.array([1.0, 2.0, 3.0, 4.0] + [0.0] * 508)
        m2 = np.array([0.5, 1.5, 2.5, 3.5] + [0.0] * 508)

        # Criptografa os vetores
        ct1 = ciphertext_factory.encode_and_encrypt(m1, pk)
        ct2 = ciphertext_factory.encode_and_encrypt(m2, pk)

        # Realiza adição homomórfica
        ct_sum = CKKSCiphertext.add_homomorphic(ct1, ct2)

        # Descriptografa e decodifica o resultado
        result = ciphertext_factory.decrypt_and_decode(ct_sum, sk, len(m1))

        # Verifica o resultado
        expected = m1 + m2
        np.testing.assert_allclose(result[:4], expected[:4], rtol=1e-3)

        # Verifica propriedades do ciphertext resultado
        assert ct_sum.level == ct1.level == ct2.level
        assert abs(ct_sum.scale - ct1.scale) < 1e-10
        assert ct_sum.size == ct1.size == ct2.size

    def test_add_homomorphic_incompatible_level(self):
        """Teste de erro ao tentar somar ciphertexts com níveis diferentes"""
        import numpy as np
        from .factories import CKKSKeyFactory, CKKSCiphertextFactory

        key_factory = CKKSKeyFactory(self.crypto_params)
        ciphertext_factory = CKKSCiphertextFactory(self.crypto_params)

        keyset = key_factory.generate_full_keyset()
        pk = keyset["public_key"]

        m1 = np.array([1.0, 2.0] + [0.0] * 510)
        m2 = np.array([3.0, 4.0] + [0.0] * 510)

        ct1 = ciphertext_factory.encode_and_encrypt(m1, pk)
        ct2 = ciphertext_factory.encode_and_encrypt(m2, pk)

        # Modifica artificialmente o nível de um dos ciphertexts
        ct2.level = ct1.level - 1

        with pytest.raises(ValueError, match="não são compatíveis para adição"):
            CKKSCiphertext.add_homomorphic(ct1, ct2)

    def test_add_homomorphic_incompatible_scale(self):
        """Teste de erro ao tentar somar ciphertexts com escalas diferentes"""
        import numpy as np
        from .factories import CKKSKeyFactory, CKKSCiphertextFactory

        key_factory = CKKSKeyFactory(self.crypto_params)
        ciphertext_factory = CKKSCiphertextFactory(self.crypto_params)

        keyset = key_factory.generate_full_keyset()
        pk = keyset["public_key"]

        m1 = np.array([1.0, 2.0] + [0.0] * 510)
        m2 = np.array([3.0, 4.0] + [0.0] * 510)

        ct1 = ciphertext_factory.encode_and_encrypt(m1, pk)
        ct2 = ciphertext_factory.encode_and_encrypt(m2, pk)

        # Modifica artificialmente a escala de um dos ciphertexts
        ct2.scale = ct1.scale * 2

        with pytest.raises(ValueError, match="não são compatíveis para adição"):
            CKKSCiphertext.add_homomorphic(ct1, ct2)

    def test_add_homomorphic_preserves_properties(self):
        """Teste se a adição preserva propriedades importantes do ciphertext"""
        import numpy as np
        from .factories import CKKSKeyFactory, CKKSCiphertextFactory

        key_factory = CKKSKeyFactory(self.crypto_params)
        ciphertext_factory = CKKSCiphertextFactory(self.crypto_params)

        keyset = key_factory.generate_full_keyset()
        pk = keyset["public_key"]

        m1 = np.array([1.0, 2.0, 3.0] + [0.0] * 509)
        m2 = np.array([0.1, 0.2, 0.3] + [0.0] * 509)

        ct1 = ciphertext_factory.encode_and_encrypt(m1, pk)
        ct2 = ciphertext_factory.encode_and_encrypt(m2, pk)

        original_level = ct1.level
        original_scale = ct1.scale
        original_size = ct1.size

        ct_sum = CKKSCiphertext.add_homomorphic(ct1, ct2)

        # Verifica que as propriedades foram preservadas
        assert ct_sum.level == original_level
        assert abs(ct_sum.scale - original_scale) < 1e-10
        assert ct_sum.size == original_size
        assert ct_sum.crypto_params == ct1.crypto_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
