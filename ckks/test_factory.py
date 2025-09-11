"""
Testes para a classe CKKSCiphertextFactory.

Este módulo contém testes unitários para verificar o funcionamento
correto da fábrica de ciphertexts CKKS.
"""

import numpy as np
from numpy.polynomial import Polynomial

from .constants import CKKSCryptographicParameters
from .factories import CKKSCiphertextFactory, create_ckks_factory


class TestCKKSCiphertextFactory:
    """Testes para a classe CKKSCiphertextFactory."""

    def test_factory_creation(self):
        """Testa criação da fábrica."""
        factory = create_ckks_factory()
        assert isinstance(factory, CKKSCiphertextFactory)
        assert isinstance(factory.crypto_params, CKKSCryptographicParameters)

    def test_encode_decode_cycle(self):
        """Testa ciclo completo de codificação e decodificação."""
        factory = create_ckks_factory()

        # Dados de teste
        original_data = [1.5, -2.3, 3.7, 0.0, 1.1]

        # Codifica
        encoded_poly = factory.ckks_encode_real(original_data)
        assert isinstance(encoded_poly, Polynomial)

        # Decodifica
        decoded_data = factory.ckks_decode_real(encoded_poly)

        # Verifica precisão
        np.testing.assert_allclose(
            decoded_data[: len(original_data)],
            original_data,
            rtol=1e-2,
            atol=1e-10,  # Tolerância relativa de 1% e absoluta pequena
        )

    def test_encode_with_custom_params(self):
        """Testa codificação com parâmetros personalizados."""
        factory = create_ckks_factory()

        original_data = [2.5, -1.8]
        custom_scale = 1000.0

        # Verifica se os dados de entrada são válidos
        assert len(original_data) > 0
        assert custom_scale > 0

        # Codifica com escala personalizada
        encoded_poly = factory.ckks_encode_real(original_data, delta_scale=custom_scale)

        # Verifica se a codificação funcionou
        assert isinstance(encoded_poly, Polynomial)
        assert len(encoded_poly.coef) > 0

        # Decodifica com a mesma escala
        decoded_data = factory.ckks_decode_real(encoded_poly, delta_scale=custom_scale)

        # Verifica se a decodificação funcionou
        assert isinstance(decoded_data, np.ndarray)
        assert len(decoded_data) >= len(original_data)

        # Verifica precisão
        np.testing.assert_allclose(
            decoded_data[: len(original_data)],
            original_data,
            rtol=5e-2,  # Tolerância de 5% para escalas personalizadas
        )

    def test_factory_with_custom_params(self):
        """Testa criação da fábrica com parâmetros personalizados."""
        custom_params = CKKSCryptographicParameters()
        factory = CKKSCiphertextFactory(custom_params)

        assert factory.crypto_params is custom_params

    def test_empty_vector_encoding(self):
        """Testa codificação de vetor vazio."""
        factory = create_ckks_factory()

        # Vetor vazio deve funcionar
        empty_data = []
        encoded_poly = factory.ckks_encode_real(empty_data)
        assert isinstance(encoded_poly, Polynomial)

        # Decodificação deve retornar valores próximos de zero
        decoded_data = factory.ckks_decode_real(encoded_poly)
        assert len(decoded_data) > 0
        # Os primeiros elementos devem ser próximos de zero
        np.testing.assert_allclose(decoded_data[:5], [0, 0, 0, 0, 0], atol=1e-10)

    def test_single_element_encoding(self):
        """Testa codificação de um único elemento."""
        factory = create_ckks_factory()

        single_value = [42.7]
        encoded_poly = factory.ckks_encode_real(single_value)
        decoded_data = factory.ckks_decode_real(encoded_poly)

        # Verifica se o primeiro elemento foi recuperado corretamente
        np.testing.assert_allclose([decoded_data[0]], single_value, rtol=1e-3)

    def test_large_vector_encoding(self):
        """Testa codificação de vetor grande."""
        factory = create_ckks_factory()

        # Cria vetor com metade do tamanho máximo permitido
        max_elements = factory.crypto_params.POLYNOMIAL_DEGREE // 2
        large_data = np.random.uniform(-10, 10, max_elements // 2).tolist()

        # Verifica que os dados foram gerados corretamente
        assert len(large_data) > 0
        assert all(isinstance(x, float) for x in large_data)

        encoded_poly = factory.ckks_encode_real(large_data)
        assert isinstance(encoded_poly, Polynomial)
        assert len(encoded_poly.coef) > 0

        decoded_data = factory.ckks_decode_real(encoded_poly)
        assert isinstance(decoded_data, np.ndarray)
        assert len(decoded_data) >= len(large_data)

        # Verifica precisão para os elementos originais
        np.testing.assert_allclose(
            decoded_data[: len(large_data)],
            large_data,
            rtol=1e-2,  # Tolerância um pouco maior para vetores grandes
        )

    def test_encrypt_decrypt_cycle(self):
        """Testa ciclo completo de criptografia e descriptografia."""
        factory = create_ckks_factory()

        # Gera chaves para teste
        n_degree = factory.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = factory.crypto_params.get_polynomial_modulus_ring()
        q_mod = factory.crypto_params.MODULUS_CHAIN[-1]
        sigma_err = factory.crypto_params.GAUSSIAN_NOISE_STDDEV

        # Gera chave secreta e pública usando as funções auxiliares da fábrica
        secret_key = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        pk_a = factory.crypto_params.generate_uniform_random_poly(n_degree, q_mod)
        e_err = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        neg_a_s = -factory.crypto_params.poly_mul_mod(
            pk_a, secret_key, q_mod, ring_poly_mod
        )
        pk_b = (neg_a_s + e_err) % ring_poly_mod
        pk_b_final = factory.crypto_params.poly_ring_mod(pk_b, ring_poly_mod, q_mod)
        public_key = (pk_b_final, pk_a)

        # Mensagem de teste
        test_data = [1.5, -2.3, 3.7]

        # Codifica mensagem
        encoded_message = factory.ckks_encode_real(test_data)

        # Criptografa
        ciphertext = factory.encrypt(encoded_message, public_key)

        # Verifica se o ciphertext foi criado corretamente
        assert hasattr(ciphertext, "components")
        assert len(ciphertext.components) == 2
        assert ciphertext.level == len(factory.crypto_params.MODULUS_CHAIN) - 1

        # Descriptografa
        decrypted_poly = factory.decrypt(ciphertext, secret_key)

        # Decodifica
        decoded_data = factory.ckks_decode_real(
            decrypted_poly,
            ciphertext.scale,
            factory.crypto_params.POLYNOMIAL_DEGREE,
            factory.crypto_params.MODULUS_CHAIN[ciphertext.level],
        )

        # Verifica se conseguimos recuperar os dados originais
        np.testing.assert_allclose(
            decoded_data[: len(test_data)],
            test_data,
            rtol=1e-1,  # Tolerância maior devido ao ruído da criptografia
        )

    def test_encode_and_encrypt(self):
        """Testa a função combinada de codificação e criptografia."""
        factory = create_ckks_factory()

        # Gera chaves
        n_degree = factory.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = factory.crypto_params.get_polynomial_modulus_ring()
        q_mod = factory.crypto_params.MODULUS_CHAIN[-1]
        sigma_err = factory.crypto_params.GAUSSIAN_NOISE_STDDEV

        secret_key = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        pk_a = factory.crypto_params.generate_uniform_random_poly(n_degree, q_mod)
        e_err = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        neg_a_s = -factory.crypto_params.poly_mul_mod(
            pk_a, secret_key, q_mod, ring_poly_mod
        )
        pk_b = (neg_a_s + e_err) % ring_poly_mod
        pk_b_final = factory.crypto_params.poly_ring_mod(pk_b, ring_poly_mod, q_mod)
        public_key = (pk_b_final, pk_a)

        # Dados de teste
        test_vector = [2.5, -1.8, 4.2]

        # Codifica e criptografa em uma operação
        ciphertext = factory.encode_and_encrypt(test_vector, public_key)

        # Verifica o resultado
        assert hasattr(ciphertext, "components")
        assert len(ciphertext.components) == 2
        assert hasattr(ciphertext, "level")
        assert hasattr(ciphertext, "scale")
        assert ciphertext.level >= 0
        assert ciphertext.scale > 0

        # Descriptografa para verificar
        decrypted_poly = factory.decrypt(ciphertext, secret_key)
        assert isinstance(decrypted_poly, Polynomial)
        assert len(decrypted_poly.coef) > 0

        decoded_data = factory.ckks_decode_real(
            decrypted_poly,
            ciphertext.scale,
            factory.crypto_params.POLYNOMIAL_DEGREE,
            factory.crypto_params.MODULUS_CHAIN[ciphertext.level],
        )
        assert isinstance(decoded_data, np.ndarray)
        assert len(decoded_data) >= len(test_vector)

        # Verifica precisão
        np.testing.assert_allclose(
            decoded_data[: len(test_vector)], test_vector, rtol=1e-1
        )

    def test_decrypt_and_decode(self):
        """Testa a função combinada de descriptografia e decodificação."""
        factory = create_ckks_factory()

        # Gera chaves
        n_degree = factory.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = factory.crypto_params.get_polynomial_modulus_ring()
        q_mod = factory.crypto_params.MODULUS_CHAIN[-1]
        sigma_err = factory.crypto_params.GAUSSIAN_NOISE_STDDEV

        secret_key = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        pk_a = factory.crypto_params.generate_uniform_random_poly(n_degree, q_mod)
        e_err = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        neg_a_s = -factory.crypto_params.poly_mul_mod(
            pk_a, secret_key, q_mod, ring_poly_mod
        )
        pk_b = (neg_a_s + e_err) % ring_poly_mod
        pk_b_final = factory.crypto_params.poly_ring_mod(pk_b, ring_poly_mod, q_mod)
        public_key = (pk_b_final, pk_a)

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
        np.testing.assert_allclose(recovered_data, original_data, rtol=1e-1)

    def test_decrypt_with_dict_format(self):
        """Testa descriptografia com formato de dicionário (compatibilidade)."""
        factory = create_ckks_factory()

        # Gera chaves
        n_degree = factory.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = factory.crypto_params.get_polynomial_modulus_ring()
        q_mod = factory.crypto_params.MODULUS_CHAIN[-1]
        sigma_err = factory.crypto_params.GAUSSIAN_NOISE_STDDEV

        secret_key = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        pk_a = factory.crypto_params.generate_uniform_random_poly(n_degree, q_mod)
        e_err = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        neg_a_s = -factory.crypto_params.poly_mul_mod(
            pk_a, secret_key, q_mod, ring_poly_mod
        )
        pk_b = (neg_a_s + e_err) % ring_poly_mod
        pk_b_final = factory.crypto_params.poly_ring_mod(pk_b, ring_poly_mod, q_mod)
        public_key = (pk_b_final, pk_a)

        # Criptografa usando a fábrica
        test_data = [1.0, 2.0]
        ciphertext = factory.encode_and_encrypt(test_data, public_key)

        # Verifica se a criptografia funcionou
        assert hasattr(ciphertext, "components")
        assert len(ciphertext.components) == 2

        # Converte para formato de dicionário
        ciphertext_dict = ciphertext.to_dict()

        # Verifica se a conversão funcionou
        assert isinstance(ciphertext_dict, dict)
        assert "c0" in ciphertext_dict
        assert "c1" in ciphertext_dict
        assert "level" in ciphertext_dict
        assert "scale" in ciphertext_dict

        # Testa descriptografia com dicionário
        decrypted_poly = factory.decrypt(ciphertext_dict, secret_key)

        # Verifica se a descriptografia funciona
        assert hasattr(decrypted_poly, "coef")
        assert isinstance(decrypted_poly, Polynomial)
        assert len(decrypted_poly.coef) > 0

        # Testa decrypt_and_decode com dicionário
        recovered_data = factory.decrypt_and_decode(
            ciphertext_dict, secret_key, expected_length=len(test_data)
        )

        # Verifica o resultado
        assert isinstance(recovered_data, np.ndarray)
        assert len(recovered_data) == len(test_data)

        # Verifica precisão
        np.testing.assert_allclose(recovered_data, test_data, rtol=1e-1)

    def test_encryption_with_custom_level(self):
        """Testa criptografia com nível personalizado."""
        factory = create_ckks_factory()

        # Gera chaves
        n_degree = factory.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = factory.crypto_params.get_polynomial_modulus_ring()
        q_mod = factory.crypto_params.MODULUS_CHAIN[-1]
        sigma_err = factory.crypto_params.GAUSSIAN_NOISE_STDDEV

        secret_key = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        pk_a = factory.crypto_params.generate_uniform_random_poly(n_degree, q_mod)
        e_err = factory.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        neg_a_s = -factory.crypto_params.poly_mul_mod(
            pk_a, secret_key, q_mod, ring_poly_mod
        )
        pk_b = (neg_a_s + e_err) % ring_poly_mod
        pk_b_final = factory.crypto_params.poly_ring_mod(pk_b, ring_poly_mod, q_mod)
        public_key = (pk_b_final, pk_a)

        # Testa com nível personalizado
        custom_level = len(factory.crypto_params.MODULUS_CHAIN) - 2  # Não o mais alto
        test_data = [5.5, -3.3]

        # Verifica se o nível personalizado é válido
        assert custom_level >= 0
        assert custom_level < len(factory.crypto_params.MODULUS_CHAIN)

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
            rtol=2e-1,  # Tolerância maior para níveis mais baixos
        )
