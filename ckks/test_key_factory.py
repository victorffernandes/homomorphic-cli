"""
Testes para a CKKSKeyFactory.

Este módulo contém testes para validar a funcionalidade
da fábrica de chaves CKKS.
"""

import numpy as np
from numpy.polynomial import Polynomial


from .factories import CKKSKeyFactory, CKKSCiphertextFactory, create_key_factory
from .constants import CKKSCryptographicParameters
from .ckks_main import Polynomial as CKKSPolynomial


class TestCKKSKeyFactory:
    """Testes para a fábrica de chaves CKKS."""

    def setup_method(self):
        """Configuração para cada teste."""
        self.crypto_params = CKKSCryptographicParameters()
        self.key_factory = CKKSKeyFactory(self.crypto_params)
        self.ciphertext_factory = CKKSCiphertextFactory(self.crypto_params)

    def test_factory_initialization(self):
        """Testa a inicialização da fábrica de chaves."""
        # Teste com parâmetros personalizados
        factory = CKKSKeyFactory(self.crypto_params)
        assert factory.crypto_params == self.crypto_params

        # Teste com parâmetros padrão
        factory_default = CKKSKeyFactory()
        assert factory_default.crypto_params is not None
        assert isinstance(factory_default.crypto_params, CKKSCryptographicParameters)

    def test_create_key_factory_function(self):
        """Testa a função de conveniência para criar fábrica."""
        factory = create_key_factory(self.crypto_params)
        assert isinstance(factory, CKKSKeyFactory)
        assert factory.crypto_params == self.crypto_params

        # Teste com parâmetros padrão
        factory_default = create_key_factory()
        assert isinstance(factory_default, CKKSKeyFactory)
        assert factory_default.crypto_params is not None

    def test_generate_secret_key(self):
        """Testa a geração de chave secreta."""
        secret_key = self.key_factory.generate_secret_key()

        # Verifica que é um polinômio
        assert isinstance(secret_key, (Polynomial, CKKSPolynomial))

        # Verifica o grau correto
        expected_degree = self.crypto_params.POLYNOMIAL_DEGREE - 1
        if hasattr(secret_key, "degree"):
            assert secret_key.degree() <= expected_degree
        elif hasattr(secret_key, "coef"):
            assert len(secret_key.coef) <= self.crypto_params.POLYNOMIAL_DEGREE

        # Verifica que não é zero
        if hasattr(secret_key, "coef"):
            assert not np.allclose(secret_key.coef, 0)

    def test_generate_public_key(self):
        """Testa a geração de chave pública."""
        secret_key = self.key_factory.generate_secret_key()
        public_key = self.key_factory.generate_public_key(secret_key)

        # Verifica que retorna uma tupla de dois elementos
        assert isinstance(public_key, tuple)
        assert len(public_key) == 2

        pk_b, pk_a = public_key

        # Verifica que ambos são polinômios
        assert isinstance(pk_b, (Polynomial, CKKSPolynomial))
        assert isinstance(pk_a, (Polynomial, CKKSPolynomial))

        # Verifica que não são zero
        if hasattr(pk_b, "coef") and hasattr(pk_a, "coef"):
            assert not np.allclose(pk_b.coef, 0)
            assert not np.allclose(pk_a.coef, 0)

    def test_generate_public_key_with_level(self):
        """Testa a geração de chave pública com nível específico."""
        secret_key = self.key_factory.generate_secret_key()
        level = 0
        public_key = self.key_factory.generate_public_key(secret_key, level)

        assert isinstance(public_key, tuple)
        assert len(public_key) == 2

        pk_b, pk_a = public_key
        assert isinstance(pk_b, (Polynomial, CKKSPolynomial))
        assert isinstance(pk_a, (Polynomial, CKKSPolynomial))

    def test_generate_relinearization_key(self):
        """Testa a geração de chave de relinearização."""
        secret_key = self.key_factory.generate_secret_key()
        relin_key = self.key_factory.generate_relinearization_key(secret_key)

        # Verifica que retorna uma tupla de dois elementos
        assert isinstance(relin_key, tuple)
        assert len(relin_key) == 2

        rlk_b, rlk_a = relin_key

        # Verifica que ambos são polinômios
        assert isinstance(rlk_b, (Polynomial, CKKSPolynomial))
        assert isinstance(rlk_a, (Polynomial, CKKSPolynomial))

        # Verifica que não são zero
        if hasattr(rlk_b, "coef") and hasattr(rlk_a, "coef"):
            assert not np.allclose(rlk_b.coef, 0)
            assert not np.allclose(rlk_a.coef, 0)

    def test_generate_relinearization_key_with_level(self):
        """Testa a geração de chave de relinearização com nível específico."""
        secret_key = self.key_factory.generate_secret_key()
        level = 0
        relin_key = self.key_factory.generate_relinearization_key(secret_key, level)

        assert isinstance(relin_key, tuple)
        assert len(relin_key) == 2

        rlk_b, rlk_a = relin_key
        assert isinstance(rlk_b, (Polynomial, CKKSPolynomial))
        assert isinstance(rlk_a, (Polynomial, CKKSPolynomial))

    def test_generate_keypair(self):
        """Testa a geração de par de chaves."""
        secret_key, public_key = self.key_factory.generate_keypair()

        # Verifica chave secreta
        assert isinstance(secret_key, (Polynomial, CKKSPolynomial))

        # Verifica chave pública
        assert isinstance(public_key, tuple)
        assert len(public_key) == 2

        pk_b, pk_a = public_key
        assert isinstance(pk_b, (Polynomial, CKKSPolynomial))
        assert isinstance(pk_a, (Polynomial, CKKSPolynomial))

        # Verifica que as chaves não são zero
        if hasattr(secret_key, "coef"):
            assert not np.allclose(secret_key.coef, 0)
        if hasattr(pk_b, "coef") and hasattr(pk_a, "coef"):
            assert not np.allclose(pk_b.coef, 0)
            assert not np.allclose(pk_a.coef, 0)

    def test_generate_full_keyset(self):
        """Testa a geração de conjunto completo de chaves."""
        keyset = self.key_factory.generate_full_keyset()

        # Verifica que retorna um dicionário com as chaves esperadas
        assert isinstance(keyset, dict)
        expected_keys = {"secret_key", "public_key", "relinearization_key"}
        assert set(keyset.keys()) == expected_keys

        # Verifica chave secreta
        secret_key = keyset["secret_key"]
        assert isinstance(secret_key, (Polynomial, CKKSPolynomial))

        # Verifica chave pública
        public_key = keyset["public_key"]
        assert isinstance(public_key, tuple)
        assert len(public_key) == 2

        # Verifica chave de relinearização
        relin_key = keyset["relinearization_key"]
        assert isinstance(relin_key, tuple)
        assert len(relin_key) == 2

        # Verifica que todas as chaves não são zero
        if hasattr(secret_key, "coef"):
            assert not np.allclose(secret_key.coef, 0)

    def test_validate_keypair_valid(self):
        """Testa a validação de par de chaves válido através de encrypt/decrypt."""
        secret_key, public_key = self.key_factory.generate_keypair()

        # Dados de teste para validação
        test_vector = [1.0, 2.0, 3.0, 4.0]

        try:
            # Codifica e criptografa usando a chave pública
            ciphertext = self.ciphertext_factory.encode_and_encrypt(
                test_vector, public_key
            )

            # Verifica que o ciphertext foi criado
            assert ciphertext is not None
            assert hasattr(ciphertext, "components")
            assert len(ciphertext.components) >= 2  # Deve ter pelo menos c0 e c1

            # Descriptografa e decodifica usando a chave secreta
            decrypted_vector = self.ciphertext_factory.decrypt_and_decode(
                ciphertext, secret_key, len(test_vector)
            )

            # Verifica que conseguiu recuperar os dados originais (com tolerância para ruído)
            assert len(decrypted_vector) >= len(test_vector)
            for i in range(len(test_vector)):
                assert (
                    abs(decrypted_vector[i] - test_vector[i]) < 0.1
                ), f"Valor {i}: esperado {test_vector[i]}, obtido {decrypted_vector[i]}"

            # Se chegou até aqui, as chaves são válidas
            validation_success = True

        except Exception as e:
            # Se houve erro na criptografia/descriptografia, as chaves são inválidas
            print(f"Erro durante validação: {e}")
            validation_success = False

        # Assert final da validação
        assert (
            validation_success
        ), "Validação das chaves falhou - encrypt/decrypt não funcionou"

        # O importante é que conseguimos executar o processo
        # A diferença matemática pode variar dependendo da implementação

    def test_multiple_key_generation_uniqueness(self):
        """Testa que múltiplas gerações produzem chaves diferentes."""
        # Gera múltiplas chaves secretas
        secret_keys = [self.key_factory.generate_secret_key() for _ in range(3)]

        # Verifica que são diferentes (probabilisticamente)
        for i in range(len(secret_keys)):
            for j in range(i + 1, len(secret_keys)):
                sk1, sk2 = secret_keys[i], secret_keys[j]
                if hasattr(sk1, "coef") and hasattr(sk2, "coef"):
                    # Chaves devem ser diferentes
                    assert not np.allclose(sk1.coef, sk2.coef)

    def test_keyset_consistency(self):
        """Testa a consistência de um conjunto completo de chaves."""
        keyset = self.key_factory.generate_full_keyset()

        secret_key = keyset["secret_key"]
        public_key = keyset["public_key"]
        relin_key = keyset["relinearization_key"]

        # Verifica que as chaves foram geradas corretamente
        assert secret_key is not None
        assert public_key is not None
        assert len(public_key) == 2

        # Testa a consistência das chaves através de encrypt/decrypt
        test_vector = [5.0, 10.0, 15.0]

        try:
            # Criptografa com chave pública
            ciphertext = self.ciphertext_factory.encode_and_encrypt(
                test_vector, public_key
            )

            # Descriptografa com chave secreta
            decrypted_vector = self.ciphertext_factory.decrypt_and_decode(
                ciphertext, secret_key, len(test_vector)
            )

            # Verifica precisão da recuperação
            keys_consistent = True
            for i in range(len(test_vector)):
                if abs(decrypted_vector[i] - test_vector[i]) > 0.1:
                    keys_consistent = False
                    break

        except Exception:
            keys_consistent = False

        assert keys_consistent, "Chaves do keyset não são consistentes entre si"

        # Verifica que a chave de relinearização é bem formada
        assert isinstance(relin_key, tuple)
        assert len(relin_key) == 2

        rlk_b, rlk_a = relin_key
        assert isinstance(rlk_b, (Polynomial, CKKSPolynomial))
        assert isinstance(rlk_a, (Polynomial, CKKSPolynomial))


if __name__ == "__main__":
    # Execução direta para desenvolvimento
    test_instance = TestCKKSKeyFactory()
    test_instance.setup_method()

    print("Executando testes da CKKSKeyFactory...")

    try:
        test_instance.test_factory_initialization()
        print("✓ test_factory_initialization")

        test_instance.test_create_key_factory_function()
        print("✓ test_create_key_factory_function")

        test_instance.test_generate_secret_key()
        print("✓ test_generate_secret_key")

        test_instance.test_generate_public_key()
        print("✓ test_generate_public_key")

        test_instance.test_generate_relinearization_key()
        print("✓ test_generate_relinearization_key")

        test_instance.test_generate_keypair()
        print("✓ test_generate_keypair")

        test_instance.test_generate_full_keyset()
        print("✓ test_generate_full_keyset")

        test_instance.test_validate_keypair_valid()
        print("✓ test_validate_keypair_valid")

        test_instance.test_validate_keypair_invalid()
        print("✓ test_validate_keypair_invalid")

        test_instance.test_multiple_key_generation_uniqueness()
        print("✓ test_multiple_key_generation_uniqueness")

        test_instance.test_keyset_consistency()
        print("✓ test_keyset_consistency")

        print("\nTodos os testes passaram! ✓")

    except Exception as e:
        print(f"\nErro durante execução dos testes: {e}")
        import traceback

        traceback.print_exc()
