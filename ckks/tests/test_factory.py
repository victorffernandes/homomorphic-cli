"""
Testes para a classe CKKSCiphertextFactory.

Este módulo contém testes unitários para verificar o funcionamento
correto da fábrica de ciphertexts CKKS.
"""

import numpy as np

from ckks.constants import CKKSCryptographicParameters
from ckks.ciphertext_factory import CKKSCiphertextFactory, create_ckks_factory
from ckks.key_factory import create_key_factory


class TestCKKSCiphertextFactory:
    """Testes para a classe CKKSCiphertextFactory."""

    PRECISION_TOLERANCE = 1e-1

    def test_factory_creation(self):
        """Testa criação da fábrica."""
        factory = create_ckks_factory()
        assert isinstance(factory, CKKSCiphertextFactory)
        assert isinstance(factory.crypto_params, CKKSCryptographicParameters)


    def test_encrypt_decrypt_cycle(self):
        """Testa ciclo completo de criptografia e descriptografia."""
        factory = create_ckks_factory()

        key_factory = create_key_factory()

        key_set = key_factory.generate_full_keyset()

        # Mensagem de teste
        test_data = [1.5, -2.3, 3.7]

        # Codifica e criptografa
        ciphertext = factory.encode_and_encrypt(test_data, key_set["public_key"])

        # Verifica se o ciphertext foi criado corretamente
        assert hasattr(ciphertext, "components")
        assert len(ciphertext.components) == 2
        assert ciphertext.level == len(factory.crypto_params.MODULUS_CHAIN) - 1

        # Descriptografa e decodifica
        decoded_data = factory.decrypt_and_decode(
            ciphertext, key_set["secret_key"], len(test_data)
        )

        # Verifica se conseguimos recuperar os dados originais
        np.testing.assert_allclose(
            decoded_data,
            test_data,
            rtol=self.PRECISION_TOLERANCE,
        )

    def test_decrypt_and_decode(self):
        """Testa a função combinada de descriptografia e decodificação."""
        # Cria as fábricas
        factory = create_ckks_factory()
        key_factory = create_key_factory()

        # Gera o conjunto de chaves usando a fábrica
        keyset = key_factory.generate_full_keyset()
        secret_key = keyset["secret_key"]
        public_key = keyset["public_key"]

        # Dados de teste
        original_data = [3.14, -2.71, 1.41]

        # Processo completo: codifica e criptografa
        ciphertext = factory.encode_and_encrypt(original_data, public_key)

        # Verifica se a criptografia funcionou
        assert hasattr(ciphertext, "components")
        assert len(ciphertext.components) == 2
        assert ciphertext.level >= 0
        assert ciphertext.scale > 0

        # Descriptografa e decodifica em uma operação
        recovered_data = factory.decrypt_and_decode(
            ciphertext, secret_key, expected_length=len(original_data)
        )

        # Verifica o resultado da descriptografia/decodificação
        assert isinstance(recovered_data, np.ndarray)
        assert len(recovered_data) == len(original_data)

        # Verifica se recuperamos os dados originais
        np.testing.assert_allclose(recovered_data, original_data, rtol=self.PRECISION_TOLERANCE)

    def test_encryption_with_custom_level(self):
        """Testa criptografia com nível personalizado."""
        factory = create_ckks_factory()
        key_factory = create_key_factory()

        # Testa com nível personalizado
        custom_level = len(factory.crypto_params.MODULUS_CHAIN) - 2  # Não o mais alto

        # Verifica se o nível personalizado é válido
        assert custom_level >= 0
        assert custom_level < len(factory.crypto_params.MODULUS_CHAIN)

        # Gera chaves usando o key_factory
        keyset = key_factory.generate_full_keyset(level=custom_level)
        secret_key = keyset["secret_key"]
        public_key = keyset["public_key"]

        test_data = [5.5, -3.3]

        # Criptografa com nível personalizado
        ciphertext = factory.encode_and_encrypt(
            test_data, public_key, level=custom_level
        )

        # Verifica se o ciphertext foi criado corretamente
        assert hasattr(ciphertext, "components")
        assert len(ciphertext.components) == 2
        assert hasattr(ciphertext, "level")
        assert hasattr(ciphertext, "scale")

        # Verifica se o nível foi configurado corretamente
        assert ciphertext.level == custom_level
        assert ciphertext.scale > 0

        # Verifica se ainda conseguimos descriptografar
        recovered_data = factory.decrypt_and_decode(
            ciphertext, secret_key, expected_length=len(test_data)
        )

        # Verifica o resultado da descriptografia
        assert isinstance(recovered_data, np.ndarray)
        assert len(recovered_data) == len(test_data)

        # Verifica precisão (pode ser menor em níveis mais baixos)
        np.testing.assert_allclose(
            recovered_data,
            test_data,
            rtol=self.PRECISION_TOLERANCE,
        )
