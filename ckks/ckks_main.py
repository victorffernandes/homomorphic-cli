import numpy as np
from .ckks_ciphertext import CKKSCiphertext
from .constants import CKKSCryptographicParameters
from .ciphertext_factory import CKKSCiphertextFactory
from .key_factory import CKKSKeyFactory

# Criação da instância global dos parâmetros
crypto_params = CKKSCryptographicParameters()

# Criação das factories
ciphertext_factory = CKKSCiphertextFactory(crypto_params)
key_factory = CKKSKeyFactory(crypto_params)


# --- FUNÇÃO DE LOG PARA DEPURAÇÃO ---
def log_poly(name, poly, q_mod=None):
    """Imprime um resumo de um polinômio para depuração."""
    # Garante que os coeficientes sejam um array numpy para as operações
    coeffs = np.array(poly.coef, dtype=np.int64)
    print(f"--- LOG: {name} ---")
    print(f"    Primeiros 8 coefs: {coeffs[:8]}")
    print(f"    Max coef: {np.max(coeffs)}")
    print(f"    Min coef: {np.min(coeffs)}")
    if q_mod:
        # Norma L-infinito: maior valor absoluto após o centered lift
        centered_coeffs = np.where(coeffs > q_mod // 2, coeffs - q_mod, coeffs)
        linf_norm = np.max(np.abs(centered_coeffs))
        print(f"    Norma L-infinito (vs q={q_mod}): {linf_norm}")
    print("-" * 20)


# --- Demonstração de Uso ---
if __name__ == "__main__":
    print(
        f"Parâmetros: N={crypto_params.POLYNOMIAL_DEGREE}, "
        f"DELTA≈2^{int(np.log2(crypto_params.SCALING_FACTOR))}, "
        f"Q_CHAIN de ~{crypto_params.MODULUS_CHAIN[0].bit_length()}"
    )
    print("-" * 50)

    # Gera chaves usando a factory
    keyset = key_factory.generate_full_keyset()
    sk = keyset["secret_key"]
    pk = keyset["public_key"]
    evk = keyset["evaluation_key"]

    print("Chaves geradas usando CKKSKeyFactory.")
    print("-" * 50)

    num_plaintext_elements = crypto_params.POLYNOMIAL_DEGREE // 2
    m1 = np.array([1, 1, 1, 1] + [0] * (num_plaintext_elements - 4))
    m2 = np.array([0.5, -0.6, 0.7, 0.8] + [0] * (num_plaintext_elements - 4))

    print(f"Texto claro m1 (primeiros 4): {m1[:4]}")
    print(f"Texto claro m2 (primeiros 4): {m2[:4]}")

    # Codifica e criptografa usando a factory
    ct1_obj = ciphertext_factory.encode_and_encrypt(m1, pk)
    ct2_obj = ciphertext_factory.encode_and_encrypt(m2, pk)

    print(f"Nível inicial dos textos cifrados: {ct1_obj.level}")
    print("-" * 50)

    print("--- TESTE DE ADIÇÃO HOMOMÓRFICA ---")
    poly_mod_ring = crypto_params.get_polynomial_modulus_ring()
    modulus_chain = crypto_params.MODULUS_CHAIN

    # Usa o método estático da classe CKKSCiphertext
    ct_add_obj = CKKSCiphertext.add_homomorphic(ct1_obj, ct2_obj)

    # Usa a factory para decrypt/decode
    decoded_add_vector = ciphertext_factory.decrypt_and_decode(ct_add_obj, sk, len(m1))

    print(f"Resultado esperado (m1 + m2): {np.round(m1 + m2, 4)[:4]}")
    print(f"Resultado obtido (Adição):   {np.round(decoded_add_vector, 4)[:4]}")
    print("-" * 50)

    print("--- TESTE DE MULTIPLICAÇÃO HOMOMÓRFICA ---")
    # Converte para formato dict para compatibilidade com funções existentes
    ct1 = ct1_obj.to_dict()
    ct2 = ct2_obj.to_dict()

    ct_mult_3part = raw_multiply_homomorphic(ct1, ct2, poly_mod_ring, modulus_chain)
    ct_mult_relin = relinearize(ct_mult_3part, evk, poly_mod_ring, modulus_chain)

    # --- DIAGNÓSTICO INTERMEDIÁRIO ---
    print("\n\n--- DIAGNÓSTICO: Testando o resultado ANTES do rescale ---\n")
    scale_before_rescale = ct_mult_relin["scale"]
    q_level_before_rescale = modulus_chain[ct_mult_relin["level"]]

    ct_mult_relin_obj = CKKSCiphertext.from_dict(ct_mult_relin, crypto_params)
    decoded_vector_before_rescale = ciphertext_factory.decrypt_and_decode(
        ct_mult_relin_obj, sk, len(m1)
    )

    print(f"Escala esperada antes do rescale: {crypto_params.SCALING_FACTOR**2}")
    print(f"Resultado esperado (m1*m2): {np.round(m1*m2, 4)[:8]}")
    print(f"Resultado obtido ANTES do rescale: {decoded_vector_before_rescale[:8]}")

    # Ajuste para compatibilidade de dimensões
    min_len = min(len(m1 * m2), len(decoded_vector_before_rescale))
    error_before = np.max(
        np.abs((m1 * m2)[:min_len] - decoded_vector_before_rescale[:min_len])
    )
    print(f"--> Erro máximo ANTES do rescale: {error_before:.10f}\n")
    print("--- FIM DO DIAGNÓSTICO ---\n\n")

    # --- CONTINUAÇÃO DO FLUXO NORMAL ---
    ct_mult_relin_obj = CKKSCiphertext.from_dict(ct_mult_relin, crypto_params)
    ct_mult_final_obj = CKKSCiphertext.rescale(ct_mult_relin_obj)

    print(f"Escala final é {ct_mult_final_obj.scale:.4f}")
    print(f"Nível final do texto cifrado: {ct_mult_final_obj.level}")

    decoded_mult_vector = ciphertext_factory.decrypt_and_decode(
        ct_mult_final_obj, sk, len(m1)
    )

    print(f"Resultado esperado (m1 * m2): {np.round(m1 * m2, 4)[:4]}")
    print(f"Resultado obtido (Multiplicação): {np.round(decoded_mult_vector, 4)[:4]}")

    # Ajuste para compatibilidade de dimensões
    min_len = min(len(m1 * m2), len(decoded_mult_vector))
    error = np.max(np.abs((m1 * m2)[:min_len] - decoded_mult_vector[:min_len]))
    print(f"\nErro máximo absoluto na multiplicação: {error:.10f}")

    print("\n" + "=" * 70)
    print("=== EXEMPLO: SOMA E MULTIPLICAÇÃO DO MESMO TEXTO CIFRADO ===")
    print("=" * 70)

    # Criar um array de teste com valores mais simples para melhor visualização
    test_array = np.array([2.0, 3.0, 4.0, 5.0] + [0.0] * (num_plaintext_elements - 4))
    print(f"Array original: {test_array[:4]}")

    # Criptografar o array
    ct_test = ciphertext_factory.encode_and_encrypt(test_array, pk)
    print(f"Array criptografado (level={ct_test.level}, scale={ct_test.scale:.2e})")

    # === OPERAÇÃO 1: SOMA HOMOMÓRFICA (ct + ct) ===
    print("\n--- SOMA HOMOMÓRFICA: ct + ct ---")
    ct_sum = CKKSCiphertext.add_homomorphic(ct_test, ct_test)

    # Decodificar resultado da soma
    result_sum = ciphertext_factory.decrypt_and_decode(ct_sum, sk, 4)
    expected_sum = test_array[:4] + test_array[:4]  # 2 * test_array

    print(f"Array esperado (2 * original): {expected_sum}")
    print(f"Array obtido (soma homomórfica): {np.round(result_sum, 6)}")

    error_sum = np.max(np.abs(expected_sum - result_sum))
    print(f"Erro máximo na soma: {error_sum:.2e}")

    # === OPERAÇÃO 2: MULTIPLICAÇÃO HOMOMÓRFICA (ct * ct) ===
    print("\n--- MULTIPLICAÇÃO HOMOMÓRFICA: ct * ct ---")

    # Usar o método completo com rescale automático
    ct_mult_complete = CKKSCiphertext.multiply_homomorphic(
        ct_test, ct_test, keyset["evaluation_key"]
    )

    # Analisar escalas
    original_scale = ct_test.scale
    mult_scale = ct_mult_complete.scale

    print(f"Escala original: {original_scale:.2e}")
    print(f"Escala após multiplicação com rescale: {mult_scale:.2e}")
    print(f"Nível original: {ct_test.level}")
    print(f"Nível após multiplicação: {ct_mult_complete.level}")

    # Descriptografar resultado da multiplicação
    result_mult = ciphertext_factory.decrypt_and_decode(ct_mult_complete, sk, 4)
    expected_mult = test_array[:4] * test_array[:4]  # test_array²

    print(f"\nResultados da multiplicação:")
    print(f"  Esperado: {expected_mult}")
    print(f"  Obtido: {np.round(result_mult, 6)}")

    error_mult = np.max(np.abs(expected_mult - result_mult))
    print(f"  Erro máximo: {error_mult:.6f}")

    # Verificar se a precisão está dentro do requisito (0.001)
    precision_target = 0.001
    if error_mult < precision_target:
        precision_status = f"✅ PRECISÃO ALCANÇADA (< {precision_target})"
    else:
        precision_status = f"⚠️ PRECISÃO INSUFICIENTE (≥ {precision_target})"

    print(f"  {precision_status}")
    # === DEMONSTRAÇÃO DE PRESERVAÇÃO DAS OPERAÇÕES ===
    print("\n--- VERIFICAÇÃO DAS PROPRIEDADES HOMOMÓRFICAS ---")
    print("✅ Soma homomórfica: Enc(a) + Enc(a) = Enc(2a)")
    print(f"  Original: {test_array[:4]}")
    print(f"  2 × Original: {2 * test_array[:4]}")
    print(f"  Soma homomórfica: {np.round(result_sum, 3)}")
    print(f"  Diferença: {np.round(2 * test_array[:4] - result_sum, 6)}")

    print("\n🔧 Multiplicação homomórfica: Enc(a) × Enc(a) = Enc(a²)")
    print(f"  Original: {test_array[:4]}")
    print(f"  Original²: {test_array[:4]**2}")
    print(f"  Mult homomórfica: {np.round(result_mult, 6)}")
    print(f"  Diferença: {np.round(test_array[:4]**2 - result_mult, 6)}")

    # === INFORMAÇÕES TÉCNICAS ===
    print("\n--- INFORMAÇÕES TÉCNICAS ---")
    print(f"Ciphertext original: level={ct_test.level}, size={ct_test.size}")
    print(f"Após soma: level={ct_sum.level}, size={ct_sum.size}")
    print(
        f"Após multiplicação: level={ct_mult_complete.level}, size={ct_mult_complete.size}"
    )
    print(f"Rescale aplicado: {ct_test.level - ct_mult_complete.level} níveis")

    print("\nGestão de escalas:")
    print(f"  Escala base: {original_scale:.2e}")
    print(f"  Escala após mult: {mult_scale:.2e}")
    print(f"  Razão de redução: {original_scale/mult_scale:.2f}")

    print("\n" + "=" * 70)
    if error_mult < precision_target:
        success_msg = "✅ MULTIPLICAÇÃO CKKS IMPLEMENTADA COM SUCESSO!"
        precision_msg = f"Precisão alcançada: {error_mult:.6f} < {precision_target}"
    else:
        success_msg = (
            "⚠️ MULTIPLICAÇÃO CKKS IMPLEMENTADA (ajuste de precisão necessário)"
        )
        precision_msg = f"Precisão atual: {error_mult:.6f} ≥ {precision_target}"

    print(success_msg)
    print(precision_msg)
    print("=" * 70)
