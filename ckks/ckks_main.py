import numpy as np
from numpy.polynomial import Polynomial
from constants import CKKSCryptographicParameters

# Criação da instância global dos parâmetros
crypto_params = CKKSCryptographicParameters()


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


# --- Funções do Esquema CKKS ---
def keygen(n_degree, ring_poly_mod, q_mod, sigma_err):
    sk_s = crypto_params.generate_gaussian_poly(n_degree, sigma_err)
    pk_a = crypto_params.generate_uniform_random_poly(n_degree, q_mod)
    e_err = crypto_params.generate_gaussian_poly(n_degree, sigma_err)
    neg_a_s = -crypto_params.poly_mul_mod(pk_a, sk_s, q_mod, ring_poly_mod)
    pk_b = (neg_a_s + e_err) % ring_poly_mod
    return sk_s, (crypto_params.poly_coeffs_mod_q(pk_b, q_mod), pk_a)


def create_relin_key(sk_s, n_degree, ring_poly_mod, q_chain, sigma_err):
    print("Gerando chave de relinearização (RLK)...")
    q_mod = q_chain[-1]
    sk_s_squared = crypto_params.poly_mul_mod(sk_s, sk_s, q_mod, ring_poly_mod)
    rlk_a = crypto_params.generate_uniform_random_poly(n_degree, q_mod)
    e_err = crypto_params.generate_gaussian_poly(n_degree, sigma_err)
    neg_a_s = -crypto_params.poly_mul_mod(rlk_a, sk_s, q_mod, ring_poly_mod)
    rlk_b = (neg_a_s + e_err + sk_s_squared) % ring_poly_mod
    return (crypto_params.poly_coeffs_mod_q(rlk_b, q_mod), rlk_a)


def ckks_encode_real(real_vector, delta_scale, n_poly_coeffs):
    # num_input_elements = n_poly_coeffs // 2
    z = np.zeros(n_poly_coeffs // 2 + 1, dtype=np.float64)
    z[: len(real_vector)] = np.array(real_vector, dtype=np.float64) * delta_scale
    poly_real_coeffs = np.fft.irfft(z, n=n_poly_coeffs)
    return Polynomial(np.round(poly_real_coeffs).astype(np.int64))


def ckks_decode_real(message_poly, delta_scale, n_poly_coeffs, q_mod):
    num_output_elements = n_poly_coeffs - 1
    coeffs = message_poly.coef

    corrected_coeffs = np.where(coeffs > q_mod // 2, coeffs - q_mod, coeffs)
    log_poly(
        "Polinômio para DECODIFICAR (após centered lift)", Polynomial(corrected_coeffs)
    )

    coeffs_for_fft = corrected_coeffs.astype(np.float64)

    decoded_scaled_spectrum = np.fft.rfft(coeffs_for_fft, n=n_poly_coeffs)
    return np.real(decoded_scaled_spectrum[:num_output_elements]) / delta_scale


def encrypt(message_poly, pk, n_degree, ring_poly_mod, q_mod, sigma_err):
    pk_b, pk_a = pk
    u = crypto_params.generate_gaussian_poly(n_degree, sigma_err)
    e1 = crypto_params.generate_gaussian_poly(n_degree, sigma_err)
    e2 = crypto_params.generate_gaussian_poly(n_degree, sigma_err)
    c0 = crypto_params.poly_mul_mod(pk_b, u, q_mod, ring_poly_mod) + e1 + message_poly
    c1 = crypto_params.poly_mul_mod(pk_a, u, q_mod, ring_poly_mod) + e2
    return {
        "c0": crypto_params.poly_ring_mod(c0, ring_poly_mod, q_mod),
        "c1": crypto_params.poly_ring_mod(c1, ring_poly_mod, q_mod),
        "level": len(crypto_params.MODULUS_CHAIN) - 1,
        "scale": crypto_params.SCALING_FACTOR,
    }


def decrypt(ct, sk, ring_poly_mod, q_chain):
    level = ct["level"]
    q_mod = q_chain[level]
    c0, c1 = ct["c0"], ct["c1"]
    c1_s = crypto_params.poly_mul_mod(c1, sk, q_mod, ring_poly_mod)
    decrypted_poly = c0 + c1_s
    final_poly = crypto_params.poly_ring_mod(decrypted_poly, ring_poly_mod, q_mod)
    log_poly(f"Polinômio DESCRIPTOGRAFADO (nível {level})", final_poly, q_mod)
    return final_poly


def add_homomorphic(ct1, ct2, ring_poly_mod, q_chain):
    level = ct1["level"]
    q_mod = q_chain[level]
    c0_add = ct1["c0"] + ct2["c0"]
    c1_add = ct1["c1"] + ct2["c1"]
    return {
        "c0": crypto_params.poly_ring_mod(c0_add, ring_poly_mod, q_mod),
        "c1": crypto_params.poly_ring_mod(c1_add, ring_poly_mod, q_mod),
        "level": level,
        "scale": ct1["scale"],
    }


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

    Q_INITIAL = crypto_params.get_initial_modulus()
    sk, pk = keygen(
        crypto_params.POLYNOMIAL_DEGREE,
        crypto_params.get_polynomial_modulus_ring(),
        Q_INITIAL,
        crypto_params.GAUSSIAN_NOISE_STDDEV,
    )
    rlk = create_relin_key(
        sk,
        crypto_params.POLYNOMIAL_DEGREE,
        crypto_params.get_polynomial_modulus_ring(),
        crypto_params.MODULUS_CHAIN,
        crypto_params.GAUSSIAN_NOISE_STDDEV,
    )
    print("Chaves geradas.")
    print("-" * 50)

    num_plaintext_elements = crypto_params.get_maximum_plaintext_slots()
    m1 = np.array([1, 1, 1, 1] + [0] * (num_plaintext_elements - 4))
    m2 = np.array([0.5, -0.6, 0.7, 0.8] + [0] * (num_plaintext_elements - 4))

    print(f"Texto claro m1 (primeiros 4): {m1[:4]}")
    print(f"Texto claro m2 (primeiros 4): {m2[:4]}")

    encoded_m1 = ckks_encode_real(
        m1, crypto_params.SCALING_FACTOR, crypto_params.POLYNOMIAL_DEGREE
    )
    encoded_m2 = ckks_encode_real(
        m2, crypto_params.SCALING_FACTOR, crypto_params.POLYNOMIAL_DEGREE
    )

    ct1 = encrypt(
        encoded_m1,
        pk,
        crypto_params.POLYNOMIAL_DEGREE,
        crypto_params.get_polynomial_modulus_ring(),
        Q_INITIAL,
        crypto_params.GAUSSIAN_NOISE_STDDEV,
    )
    ct2 = encrypt(
        encoded_m2,
        pk,
        crypto_params.POLYNOMIAL_DEGREE,
        crypto_params.get_polynomial_modulus_ring(),
        Q_INITIAL,
        crypto_params.GAUSSIAN_NOISE_STDDEV,
    )
    print(f"Nível inicial dos textos cifrados: {ct1['level']}")
    print("-" * 50)

    print("--- TESTE DE ADIÇÃO HOMOMÓRFICA ---")
    poly_mod_ring = crypto_params.get_polynomial_modulus_ring()
    modulus_chain = crypto_params.MODULUS_CHAIN

    ct_add = add_homomorphic(ct1, ct2, poly_mod_ring, modulus_chain)
    decrypted_add_poly = decrypt(ct_add, sk, poly_mod_ring, modulus_chain)
    q_level_add = modulus_chain[ct_add["level"]]
    decoded_add_vector = ckks_decode_real(
        decrypted_add_poly,
        ct_add["scale"],
        crypto_params.POLYNOMIAL_DEGREE,
        q_level_add,
    )
    print(f"Resultado esperado (m1 + m2): {np.round(m1 + m2, 4)[:4]}")
    print(f"Resultado obtido (Adição):   {np.round(decoded_add_vector, 4)[:4]}")
    print("-" * 50)

    print("--- TESTE DE MULTIPLICAÇÃO HOMOMÓRFICA ---")
    ct_mult_3part = multiply_homomorphic_step1(ct1, ct2, poly_mod_ring, modulus_chain)
    ct_mult_relin = relinearize(ct_mult_3part, rlk, poly_mod_ring, modulus_chain)

    # --- DIAGNÓSTICO INTERMEDIÁRIO ---
    print("\n\n--- DIAGNÓSTICO: Testando o resultado ANTES do rescale ---\n")
    scale_before_rescale = ct_mult_relin["scale"]
    q_level_before_rescale = modulus_chain[ct_mult_relin["level"]]

    decrypted_poly_before_rescale = decrypt(
        ct_mult_relin, sk, poly_mod_ring, modulus_chain
    )
    decoded_vector_before_rescale = ckks_decode_real(
        decrypted_poly_before_rescale,
        scale_before_rescale,
        crypto_params.POLYNOMIAL_DEGREE,
        q_level_before_rescale,
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

    decrypted_mult_poly = decrypt(ct_mult_final, sk, poly_mod_ring, modulus_chain)
    q_level_mult = modulus_chain[ct_mult_final["level"]]
    decoded_mult_vector = ckks_decode_real(
        decrypted_mult_poly,
        ct_mult_final["scale"],
        crypto_params.POLYNOMIAL_DEGREE,
        q_level_mult,
    )

    print(f"Resultado esperado (m1 * m2): {np.round(m1 * m2, 4)[:4]}")
    print(f"Resultado obtido (Multiplicação): {np.round(decoded_mult_vector, 4)[:4]}")

    # Ajuste para compatibilidade de dimensões
    min_len = min(len(m1 * m2), len(decoded_mult_vector))
    error = np.max(np.abs((m1 * m2)[:min_len] - decoded_mult_vector[:min_len]))
    print(f"\nErro máximo absoluto na multiplicação: {error:.10f}")
