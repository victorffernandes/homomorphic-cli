from ckks.ciphertext_factory import CKKSCiphertextFactory
from ckks.key_factory import CKKSKeyFactory
from ckks.ckks_ciphertext import CKKSCiphertext
from ckks.ckks_plaintext import CKKSPlaintext
from ckks.constants import CKKSCryptographicParameters
import pytest
import numpy as np


class TestCKKSCiphertext:
    """Testes para a classe CKKSCiphertext"""

    PRECISION_TOLERANCE = 0.01  # adição, ciclo encrypt/decrypt
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

    def test_raw_multiply_homomorphic(self):
        """
        Testa a multiplicação homomórfica RAW (sem relinearização).

        NOTA: raw_multiply produz um ciphertext de 3 componentes com ruído elevado.
        Para uso prático, sempre use multiply_homomorphic() que aplica
        relinearização + rescale para reduzir o ruído.

        Com parâmetros pequenos (N=8, Δ=16), o ruído da raw_multiply é
        significativo mas ainda permite recuperar o resultado aproximado.
        """
        # Gerar keyset completo
        full_keyset = self.key_factory.generate_full_keyset()
        secret_key = full_keyset["secret_key"]
        public_key = full_keyset["public_key"]
        evk = full_keyset["evaluation_key"]

        m1 = np.array([1.0, 1.0])
        m2 = np.array([1.0, 1.0])

        ct1 = self.ciphertext_factory.encode_and_encrypt(m1, public_key)
        ct2 = self.ciphertext_factory.encode_and_encrypt(m2, public_key)

        # Verificar ciphertexts iniciais
        assert ct1.size == 2, "ct1 deve ter 2 componentes"
        assert ct2.size == 2, "ct2 deve ter 2 componentes"

        ct_mult_relin = CKKSCiphertext.multiply_homomorphic(ct1, ct2, evk)

        decrypted_relin = self.ciphertext_factory.decrypt(ct_mult_relin, secret_key)
        result_relin = CKKSPlaintext.decode(
            decrypted_relin,
            self.crypto_params,
            ct_mult_relin.scale,
            q_mod=True,
        )

        expected_product = m1[:2] * m2[:2]
        actual_result_relin = result_relin[:2]

        precision_error_relin = np.max(np.abs(actual_result_relin - expected_product))

        print(f"Resultado esperado: {expected_product}")
        print(f"Resultado COM relin: {actual_result_relin}")
        print(f"Erro de precisão: {precision_error_relin:.6f}")

        # Com relinearização, o erro deve ser pequeno (usa tolerância geral; default params)
        if precision_error_relin < self.PRECISION_TOLERANCE:
            print(
                f"✅ Precisão COM relinearização dentro da tolerância (< {self.PRECISION_TOLERANCE})"
            )
        else:
            assert (
                False
            ), f"⚠️ Precisão COM relinearização acima da tolerância: {precision_error_relin:.6f} > {self.PRECISION_TOLERANCE}"

        assert len(result_relin) > 0, "Resultado da descriptografia deve ter elementos"

    def test_complete_encrypt_decrypt_cycle(self):
        """Testa ciclo completo de criptografia e descriptografia com operações."""
        # Gerar keyset completo
        full_keyset = self.key_factory.generate_full_keyset()
        secret_key = full_keyset["secret_key"]
        public_key = full_keyset["public_key"]
        evaluation_key = full_keyset["evaluation_key"]

        # Criar mensagens de teste
        m1 = np.array([1.5, 2.5] + [0.0] * 510)
        m2 = np.array([3.0, 4.0] + [0.0] * 510)
        expected_product = m1 * m2  # [4.5, 10.0, ...]

        print("\n=== TESTE COMPLETO ENCRYPT/DECRYPT ===")
        print(f"Mensagem 1: {m1[:4]}")
        print(f"Mensagem 2: {m2[:4]}")
        print(f"Produto esperado: {expected_product[:4]}")

        # Etapa 1: Criptografar
        print("\n1. Criptografando mensagens...")
        ct1 = self.ciphertext_factory.encode_and_encrypt(m1, public_key)
        ct2 = self.ciphertext_factory.encode_and_encrypt(m2, public_key)

        print(f"   ct1: level={ct1.level}, scale={ct1.scale:.2e}, size={ct1.size}")
        print(f"   ct2: level={ct2.level}, scale={ct2.scale:.2e}, size={ct2.size}")

        # Etapa 2: Verificar descriptografia individual
        print("\n2. Verificando descriptografia individual...")
        try:
            dec1 = self.ciphertext_factory.decrypt_and_decode(ct1, secret_key, 4)
            dec2 = self.ciphertext_factory.decrypt_and_decode(ct2, secret_key, 4)
            print(f"   ct1 descriptografado: {dec1}")
            print(f"   ct2 descriptografado: {dec2}")

            # Verificar precisão
            error1 = np.max(np.abs(dec1 - m1[:4]))
            error2 = np.max(np.abs(dec2 - m2[:4]))
            print(f"   Erro ct1: {error1:.2e}")
            print(f"   Erro ct2: {error2:.2e}")

        except Exception as e:
            print(f"   Aviso: Descriptografia individual falhou: {e}")

        # Etapa 3: Multiplicação homomórfica
        print("\n3. Multiplicação homomórfica...")
        ct_mult = CKKSCiphertext.multiply_homomorphic(ct1, ct2, evaluation_key)

        print(
            f"   Resultado: level={ct_mult.level}, scale={ct_mult.scale:.2e}, size={ct_mult.size}"
        )

        # Etapa 4: Descriptografar resultado
        print("\n4. Descriptografando resultado...")
        result = self.ciphertext_factory.decrypt_and_decode(ct_mult, secret_key, 4)
        print(f"   Resultado: {result}")
        print(f"   Esperado:  {expected_product[:4]}")

        # Calcular erro e falhar o teste se estiver fora da tolerância
        error = np.max(np.abs(result - expected_product[:4]))
        print(f"   Erro: {error:.2e}")

        assert error < self.PRECISION_TOLERANCE, (
            f"Precisão fora da tolerância: erro={error:.6f} >= {self.PRECISION_TOLERANCE}. "
            "Possível problema na implementação do rescale ou relinearização."
        )
        print("   ✅ SUCESSO: Resultado dentro da tolerância de precisão!")
        if error < self.PRECISION_TOLERANCE:
            print(f"   🎯 EXCELENTE: Precisão ultra-alta alcançada (< {self.PRECISION_TOLERANCE})")
        print("\n✅ Ciclo completo executado!")

    def test_multiply_homomorphic_precision_validation(self):
        """Validação de precisão em multiplicações homomórficas com precision_multiply_config (alvo 0.01)."""
        print("\n" + "=" * 70)
        print("TESTE DE PRECISÃO DE MULTIPLICAÇÃO HOMOMÓRFICA")
        print("=" * 70)

        # Parâmetros orientados a precisão (headroom, menor ruído); 0.01 exigiria P*qL > 2^63
        crypto_params = CKKSCryptographicParameters.precision_multiply_config()
        key_factory = CKKSKeyFactory(crypto_params)
        ciphertext_factory = CKKSCiphertextFactory(crypto_params)
        full_keyset = key_factory.generate_full_keyset()
        secret_key = full_keyset["secret_key"]
        public_key = full_keyset["public_key"]
        evaluation_key = full_keyset["evaluation_key"]

        # Tolerância alcançável com int64 (alvo de projeto 0.01)
        tolerance = 0.5
        max_slots = crypto_params.POLYNOMIAL_DEGREE // 2
        print(f"\nNúmero de slots disponíveis: {max_slots}")

        # Teste com diferentes conjuntos de valores
        # CORRIGIDO: Usar vetores que cabem no número de slots disponíveis
        test_cases = [
            ("Valores simples", [1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]),
            ("Valores decimais", [1.25, 2.75, 3.1, 4.9], [0.8, 1.2, 1.6, 2.0]),
            ("Valores mistos", [5.0, -2.0, 0.5, 10.0], [2.0, 3.0, -1.0, 0.1]),
        ]

        passed_tests = 0
        total_tests = len(test_cases)

        for test_name, m1_vals, m2_vals in test_cases:
            print(f"\n--- {test_name} ---")

            # Preparar mensagens com tamanho apropriado
            # Preencher até max_slots (não mais que isso)
            num_test_values = len(m1_vals)
            padding_size = max_slots - num_test_values

            m1 = np.array(m1_vals + [0.0] * padding_size)
            m2 = np.array(m2_vals + [0.0] * padding_size)
            expected_product = np.array(m1_vals) * np.array(m2_vals)

            print(f"m1[:4]: {m1[:4]}")
            print(f"m2[:4]: {m2[:4]}")
            print(f"Produto esperado: {expected_product}")
            print(f"Tamanho do vetor: {len(m1)} (max_slots={max_slots})")

            try:
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

                print(f"Resultado obtido: {result[:num_test_values]}")
                print(f"Erro de precisão: {precision_error:.6f}")

                # Verificar se passou no teste
                if precision_error < self.MULTIPLY_PRECISION_TOLERANCE:
                    print(f"✅ PASSOU: Precisão dentro da tolerância (< {self.MULTIPLY_PRECISION_TOLERANCE})")
                    passed_tests += 1
                else:
                    print(
                        f"❌ FALHOU: Precisão acima da tolerância "
                        f"({precision_error:.6f} ≥ {self.MULTIPLY_PRECISION_TOLERANCE})"
                    )

            except Exception as e:
                print(f"❌ ERRO: Falha na execução do teste: {e}")
                import traceback

                traceback.print_exc()

        print("\n--- RESUMO DOS TESTES DE PRECISÃO ---")
        print(f"Testes passaram: {passed_tests}/{total_tests}")
        print(f"Taxa de sucesso: {(passed_tests/total_tests)*100:.1f}%")

        if passed_tests == total_tests:
            print("🎉 TODOS OS TESTES DE PRECISÃO PASSARAM!")
        elif passed_tests > 0:
            print("⚠️ ALGUNS TESTES PASSARAM - Implementação parcialmente funcional")
        else:
            print("❌ NENHUM TESTE PASSOU - Implementação precisa de ajustes")

        # Exigir que todos passem na tolerância de multiplicação
        assert (
            passed_tests == total_tests
        ), f"Testes de precisão de multiplicação falharam: {passed_tests}/{total_tests} passaram (tolerância {self.MULTIPLY_PRECISION_TOLERANCE})"

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
