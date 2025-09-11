import numpy as np
from .ckks import CKKSCiphertext
from numpy.polynomial import Polynomial
from .constants import CKKSCryptographicParameters
from .factories import CKKSCiphertextFactory, CKKSKeyFactory

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


# --- Funções de Operações Homomórficas ---


def multiply_homomorphic_step1(ct1, ct2, ring_poly_mod, q_chain):
    level = ct1["level"]
    q_mod = q_chain[level]
    d0 = crypto_params.poly_mul_mod(ct1["c0"], ct2["c0"], q_mod, ring_poly_mod)
    d1_p1 = crypto_params.poly_mul_mod(ct1["c0"], ct2["c1"], q_mod, ring_poly_mod)
    d1_p2 = crypto_params.poly_mul_mod(ct1["c1"], ct2["c0"], q_mod, ring_poly_mod)
    d1 = d1_p1 + d1_p2
    d2 = crypto_params.poly_mul_mod(ct1["c1"], ct2["c1"], q_mod, ring_poly_mod)
    return {
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "level": level,
        "scale": ct1["scale"] * ct2["scale"],
    }


def relinearize(ct_3part, rlk, ring_poly_mod, q_chain):
    level = ct_3part["level"]
    q_mod = q_chain[level]
    d0, d1, d2 = ct_3part["d0"], ct_3part["d1"], ct_3part["d2"]
    rlk_b, rlk_a = rlk
    c0_new = d0 + crypto_params.poly_mul_mod(d2, rlk_b, q_mod, ring_poly_mod)
    c1_new = d1 + crypto_params.poly_mul_mod(d2, rlk_a, q_mod, ring_poly_mod)
    return {
        "c0": crypto_params.poly_ring_mod(c0_new, ring_poly_mod, q_mod),
        "c1": crypto_params.poly_ring_mod(c1_new, ring_poly_mod, q_mod),
        "level": level,
        "scale": ct_3part["scale"],
    }


def rescale(ct, ring_poly_mod, q_chain, delta_scale):
    level = ct["level"]
    if level == 0:
        raise ValueError("Não há mais níveis para rescalonar.")

    q_next = q_chain[level - 1]

    # LOGS ADICIONADOS AQUI PARA VER A DESTRUIÇÃO DOS DADOS
    c0_coeffs_obj = ct["c0"].coef
    c0_coeffs_numeric = c0_coeffs_obj.astype(np.float64)
    print("\n--- LOG: DENTRO DO RESCALE (para c0) ---")
    print(f"    Coefs ANTES da divisão: {c0_coeffs_numeric[:8]}")
    c0_divided = c0_coeffs_numeric / delta_scale
    print(f"    Coefs DEPOIS da divisão por DELTA: {c0_divided[:8]}")
    c0_rounded = np.round(c0_divided)
    print(f"    Coefs DEPOIS do round: {c0_rounded[:8]}")
    print("-" * 20)

    c0_rescaled_coeffs = c0_rounded.astype(np.int64)

    c1_coeffs_numeric = ct["c1"].coef.astype(np.float64)
    c1_rescaled_coeffs = np.round(c1_coeffs_numeric / delta_scale).astype(np.int64)

    c0_rescaled = Polynomial(c0_rescaled_coeffs)
    c1_rescaled = Polynomial(c1_rescaled_coeffs)

    new_scale = ct["scale"] / delta_scale

    return {
        "c0": crypto_params.poly_ring_mod(c0_rescaled, ring_poly_mod, q_next),
        "c1": crypto_params.poly_ring_mod(c1_rescaled, ring_poly_mod, q_next),
        "level": level - 1,
        "scale": new_scale,
    }


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
    rlk = keyset["relinearization_key"]

    print("Chaves geradas usando CKKSKeyFactory.")
    print("-" * 50)

    num_plaintext_elements = crypto_params.get_maximum_plaintext_slots()
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

    ct_mult_3part = multiply_homomorphic_step1(ct1, ct2, poly_mod_ring, modulus_chain)
    ct_mult_relin = relinearize(ct_mult_3part, rlk, poly_mod_ring, modulus_chain)

    # --- DIAGNÓSTICO INTERMEDIÁRIO ---
    print("\n\n--- DIAGNÓSTICO: Testando o resultado ANTES do rescale ---\n")
    scale_before_rescale = ct_mult_relin["scale"]
    q_level_before_rescale = modulus_chain[ct_mult_relin["level"]]

    ct_mult_relin_obj = CKKSCiphertext.from_dict(ct_mult_relin, crypto_params)
    decoded_vector_before_rescale = ciphertext_factory.decrypt_and_decode(
        ct_mult_relin_obj, sk, len(m1)
    )

    print(
        f"Escala esperada antes do rescale: {crypto_params.get_scaling_factor_squared()}"
    )
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
    ct_mult_final = rescale(
        ct_mult_relin, poly_mod_ring, modulus_chain, crypto_params.SCALING_FACTOR
    )

    print(f"Escala final é {ct_mult_final['scale']:.4f}")
    print(f"Nível final do texto cifrado: {ct_mult_final['level']}")

    ct_mult_final_obj = CKKSCiphertext.from_dict(ct_mult_final, crypto_params)
    decoded_mult_vector = ciphertext_factory.decrypt_and_decode(
        ct_mult_final_obj, sk, len(m1)
    )

    print(f"Resultado esperado (m1 * m2): {np.round(m1 * m2, 4)[:4]}")
    print(f"Resultado obtido (Multiplicação): {np.round(decoded_mult_vector, 4)[:4]}")

    # Ajuste para compatibilidade de dimensões
    min_len = min(len(m1 * m2), len(decoded_mult_vector))
    error = np.max(np.abs((m1 * m2)[:min_len] - decoded_mult_vector[:min_len]))
    print(f"\nErro máximo absoluto na multiplicação: {error:.10f}")
