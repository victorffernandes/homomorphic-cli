from ckks.ciphertext_factory import CKKSCiphertextFactory
from ckks.key_factory import CKKSKeyFactory
from ckks.ckks_ciphertext import CKKSCiphertext
from ckks.ckks_plaintext import CKKSPlaintext
from ckks.constants import CKKSCryptographicParameters
import pytest
import numpy as np


class TestCKKSCiphertext:
    """Testes para a classe CKKSCiphertext"""

    PRECISION_TOLERANCE = 0.001  # adição, ciclo encrypt/decrypt
    MULTIPLY_PRECISION_TOLERANCE = 0.01  # alvo para multiplicação (usa precision_multiply_config no teste dedicado)
    SCALE_TOLERANCE = 1e-10

    def setup_method(self):
        """Configuração executada antes de cada teste"""
        self.crypto_params = CKKSCryptographicParameters()
        self.key_factory = CKKSKeyFactory(self.crypto_params)
        self.ciphertext_factory = CKKSCiphertextFactory(self.crypto_params)
        self.c0 = self.crypto_params.generate_uniform_random_poly()
        self.c1 = self.crypto_params.generate_uniform_random_poly()

    def test_add_homomorphic_basic(self):
        """Teste de adição homomórfica básica"""
        import numpy as np
        from ckks.key_factory import CKKSKeyFactory
        from ckks.ciphertext_factory import CKKSCiphertextFactory

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
        np.testing.assert_allclose(result[:4], expected[:4], rtol=self.PRECISION_TOLERANCE)

        # Verifica propriedades do ciphertext resultado
        assert ct_sum.level == ct1.level == ct2.level
        assert abs(ct_sum.scale - ct1.scale) < self.SCALE_TOLERANCE
        assert ct_sum.size == ct1.size == ct2.size

    def test_complete_encrypt_decrypt_cycle(self):
        """Testa ciclo completo de criptografia e descriptografia (sem operações homomórficas)."""
        # Gerar keyset completo
        full_keyset = self.key_factory.generate_full_keyset()
        secret_key = full_keyset["secret_key"]
        public_key = full_keyset["public_key"]

        # Criar mensagens de teste
        m1 = np.array([1.5, 2.5] + [0.0] * 510)
        m2 = np.array([3.0, 4.0] + [0.0] * 510)

        print("\n=== TESTE COMPLETO ENCRYPT/DECRYPT (SEM MULTIPLICAÇÃO) ===")
        print(f"Mensagem 1: {m1[:4]}")
        print(f"Mensagem 2: {m2[:4]}")

        # Criptografar mensagens
        ct1 = self.ciphertext_factory.encode_and_encrypt(m1, public_key)
        ct2 = self.ciphertext_factory.encode_and_encrypt(m2, public_key)

        print(f"   ct1: level={ct1.level}, scale={ct1.scale:.2e}, size={ct1.size}")
        print(f"   ct2: level={ct2.level}, scale={ct2.scale:.2e}, size={ct2.size}")

        # Descriptografar e validar precisão
        dec1 = self.ciphertext_factory.decrypt_and_decode(ct1, secret_key, 4)
        dec2 = self.ciphertext_factory.decrypt_and_decode(ct2, secret_key, 4)

        print(f"   ct1 descriptografado: {dec1}")
        print(f"   ct2 descriptografado: {dec2}")

        error1 = np.max(np.abs(dec1 - m1[:4]))
        error2 = np.max(np.abs(dec2 - m2[:4]))

        print(f"   Erro ct1: {error1:.6f}")
        print(f"   Erro ct2: {error2:.6f}")

        assert error1 < self.PRECISION_TOLERANCE, (
            f"Precisão fora da tolerância para ct1: erro={error1:.6f} >= {self.PRECISION_TOLERANCE}."
        )
        assert error2 < self.PRECISION_TOLERANCE, (
            f"Precisão fora da tolerância para ct2: erro={error2:.6f} >= {self.PRECISION_TOLERANCE}."
        )

        print("✅ Ciclo completo encrypt/decrypt dentro da tolerância de precisão!")

    def test_multiply_homomorphic_precision_validation(self):
        """Validação de precisão em uma multiplicação homomórfica com precision_multiply_config (alvo 0.01)."""

        # Parâmetros orientados a precisão (headroom, menor ruído) com margem generosa
        crypto_params = CKKSCryptographicParameters.precision_multiply_config()
        key_factory = CKKSKeyFactory(crypto_params)
        ciphertext_factory = CKKSCiphertextFactory(crypto_params)
        full_keyset = key_factory.generate_full_keyset()
        secret_key = full_keyset["secret_key"]
        public_key = full_keyset["public_key"]
        evaluation_key = full_keyset["evaluation_key"]

        max_slots = crypto_params.POLYNOMIAL_DEGREE // 2

        # Único caso de teste de precisão
        m1_vals = [1.0, 2.0, 3.0, 4.0]
        m2_vals = [0.5, 1.5, 2.5, 3.5]

        num_test_values = len(m1_vals)
        padding_size = max_slots - num_test_values

        m1 = np.array(m1_vals + [0.0] * padding_size)
        m2 = np.array(m2_vals + [0.0] * padding_size)
        expected_product = np.array(m1_vals) * np.array(m2_vals)

        # Criptografar mensagens
        ct1 = ciphertext_factory.encode_and_encrypt(m1, public_key)
        ct2 = ciphertext_factory.encode_and_encrypt(m2, public_key)

        # Multiplicação homomórfica com rescale
        ct_mult = CKKSCiphertext.multiply_homomorphic(ct1, ct2, evaluation_key)

        # Descriptografar e decodificar resultado
        result = ciphertext_factory.decrypt_and_decode(
            ct_mult, secret_key, num_test_values
        )

        # Calcular erro de precisão apenas nos valores de teste
        precision_error = np.max(
            np.abs(result[:num_test_values] - expected_product)
        )

        # Assert direto na precisão, mantendo a tolerância configurada na classe
        assert precision_error < self.MULTIPLY_PRECISION_TOLERANCE, (
            f"Erro de precisão na multiplicação homomórfica fora da tolerância: "
            f"{precision_error:.6f} >= {self.MULTIPLY_PRECISION_TOLERANCE}"
        )

    def test_rescale_properties(self):
        """Testa propriedades específicas do rescale."""
        # Gerar keyset
        full_keyset = self.key_factory.generate_full_keyset()
        public_key = full_keyset["public_key"]

        # Criar mensagem de teste
        m = np.array([5.0, 7.0] + [0.0] * 510)
        ct = self.ciphertext_factory.encode_and_encrypt(m, public_key)

        initial_level = ct.level
        initial_scale = ct.scale

        # Aplicar rescale
        ct_rescaled = CKKSCiphertext.rescale(ct)

        # Verificar propriedades
        assert ct_rescaled.level == initial_level - 1, "Nível deve diminuir em 1"
        assert ct_rescaled.size == ct.size, "Número de componentes deve ser mantido"
        assert ct_rescaled.scale != initial_scale, "Escala deve ser ajustada"

        # Verificar que não pode rescalar no nível 0
        if ct_rescaled.level > 0:
            # Continue rescaling until level 0
            ct_level_0 = ct_rescaled
            while ct_level_0.level > 0:
                ct_level_0 = CKKSCiphertext.rescale(ct_level_0)

            # Now try to rescale at level 0 - should fail
            try:
                CKKSCiphertext.rescale(ct_level_0)
                assert False, "Deveria ter falhado ao rescalar no nível 0"
            except ValueError as e:
                assert "Não há mais níveis" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
