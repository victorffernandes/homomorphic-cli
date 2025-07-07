import numpy as np
from numpy.polynomial import Polynomial
# from scipy.fft import fft, ifft # Você pode manter isso se quiser comparar ou usar no futuro

# --- Parâmetros do Esquema CKKS (Simplificado) ---
N = 16
Q_CHAIN = [103427, 1000000037, 1000000063, 1000000073]
Q = 103427
DELTA = 2**12 # Mantendo o DELTA alto que mostrou melhora no DC
SIGMA = 3.2
POLY_MOD_RING_COEFFS = [0] * N + [1]
POLY_MOD_RING_COEFFS[0] = 1
POLY_MOD_RING = Polynomial(POLY_MOD_RING_COEFFS)

# --- Funções Auxiliares para Polinômios---
def poly_coeffs_mod_q(p_numpy, q_coeff):
    coeffs = np.mod(np.round(p_numpy.coef).astype(np.int64), q_coeff)
    return Polynomial(coeffs)

def poly_ring_mod(p_numpy, ring_poly_mod, q_coeff):
    remainder_poly = p_numpy % ring_poly_mod
    return poly_coeffs_mod_q(remainder_poly, q_coeff)

def generate_gaussian_poly(degree_n, sigma_val):
    coeffs = np.round(np.random.normal(0, sigma_val, size=degree_n)).astype(np.int64)
    return Polynomial(coeffs)

def generate_uniform_random_poly(degree_n, q_bound):
    coeffs = np.random.randint(0, q_bound, size=degree_n)
    return Polynomial(coeffs)

# --- Geração de Chaves (Key Generation)---
def keygen(n_degree, ring_poly_mod, q_mod, sigma_err):
    sk_s = generate_gaussian_poly(n_degree, sigma_err)
    pk_a = generate_uniform_random_poly(n_degree, q_mod)
    e_err = generate_gaussian_poly(n_degree, sigma_err)
    neg_a_s = -pk_a * sk_s
    pk_b = poly_ring_mod(neg_a_s + e_err, ring_poly_mod, q_mod)
    return sk_s, (pk_b, pk_a)

def relin_keygen(sk, n, q, p, sigma):
    """Gera a chave de relinearização (evk), uma criptografia de P*s^2."""
    a_prime = generate_uniform_random_poly(n, q * p) # Gerado em um módulo maior
    e_prime = generate_uniform_random_poly(n, sigma)
    sk_squared = sk * sk
    
    # evk é gerado com P e depois reduzido
    evk0 = -a_prime * sk + e_prime + p * sk_squared
    evk1 = a_prime
    
    # A chave é um par de polinômios em R_{q*p}
    return (evk0, evk1)


# --- NOVA Codificação Correta para Dados Reais (N/2 entradas) ---
def ckks_encode_real(real_vector_n_half_elements, delta_scale, n_poly_coeffs):
    """
    Codifica N/2 elementos reais em um polinômio de N coeficientes reais.
    Usa np.fft.irfft para garantir que os coeficientes do polinômio sejam reais.
    """
    num_input_elements = n_poly_coeffs // 2
    if len(real_vector_n_half_elements) != num_input_elements:
        raise ValueError(f"O vetor de entrada deve ter {num_input_elements} elementos reais.")

    scaled_real_vector = np.array(real_vector_n_half_elements) * delta_scale
    print(f"DEBUG ENCODE REAL: scaled_real_vector (N/2 elementos): {scaled_real_vector}")

    # Prepara o espectro de entrada para irfft (tamanho N/2 + 1)
    # A entrada para irfft é o espectro de frequência não redundante de um sinal real.
    spectrum_for_irfft = np.zeros(num_input_elements + 1, dtype=np.complex128)
    
    # Mapeia os N/2 valores reais para as componentes do espectro
    # scaled_real_vector[0] -> componente DC (frequência 0)
    # scaled_real_vector[1] até scaled_real_vector[N/2-1] -> componentes de frequência positiva
    spectrum_for_irfft[:num_input_elements] = scaled_real_vector 
    
    # A componente de Nyquist (índice N/2) é definida como 0, pois temos N/2 entradas.
    # Se tivéssemos um (N/2)-ésimo valor, poderíamos colocá-lo aqui (deve ser real).
    # Se scaled_real_vector já tivesse N/2+1 elementos, usaríamos todos.
    # No nosso caso, estamos mapeando N/2 valores, então a última entrada de spectrum_for_irfft
    # (correspondente à frequência de Nyquist se existisse um (N/2)-ésimo z_k) fica 0
    # ou poderia ser o último elemento de scaled_real_vector se N/2 fosse usado para Nyquist
    # e os outros N/2-1 para freq positivas.
    # Para ser claro: z0,...,z_{N/2-1}
    # spectrum_for_irfft[0] = z0_scaled
    # spectrum_for_irfft[1] = z1_scaled
    # ...
    # spectrum_for_irfft[N/2-1] = z_{N/2-1}_scaled
    # spectrum_for_irfft[N/2] = 0 (Nyquist, não temos um z_{N/2})

    print(f"DEBUG ENCODE REAL: spectrum_for_irfft (N/2+1 elementos): {spectrum_for_irfft}")

    # np.fft.irfft produzirá um output de n_poly_coeffs elementos REAIS
    poly_real_coeffs = np.fft.irfft(spectrum_for_irfft, n=n_poly_coeffs)
    print(f"DEBUG ENCODE REAL: poly_real_coeffs (saída da irfft, N elementos): {poly_real_coeffs}") # print primeiros 8

    message_coeffs_int = np.round(poly_real_coeffs).astype(np.int64)
    print(f"DEBUG ENCODE REAL: message_coeffs_int (arredondado, N elementos): {message_coeffs_int[:8]}") # print primeiros 8
    return Polynomial(message_coeffs_int)

# --- NOVA Decodificação Correta para Dados Reais ---
def ckks_decode_real(message_poly_m_prime, delta_scale, n_poly_coeffs):
    """
    Decodifica um polinômio de N coeficientes reais para N/2 elementos reais.
    Usa np.fft.rfft.
    """
    num_output_elements = n_poly_coeffs // 2
    
    coeffs_to_transform = message_poly_m_prime.coef.copy() # Garante que é um array e não uma view
    # Garante que o vetor de coeficientes tenha tamanho n_poly_coeffs para a rfft
    if len(coeffs_to_transform) < n_poly_coeffs:
        coeffs_to_transform = np.pad(coeffs_to_transform, (0, n_poly_coeffs - len(coeffs_to_transform)), 'constant')
    elif len(coeffs_to_transform) > n_poly_coeffs:
        coeffs_to_transform = coeffs_to_transform[:n_poly_coeffs]
    
    print(f"DEBUG DECODE REAL: coeffs_to_transform (para rfft, N elementos): {coeffs_to_transform[:8]}")

    # np.fft.rfft de dados reais produz um espectro de N/2 + 1 componentes complexas
    decoded_scaled_half_spectrum = np.fft.rfft(coeffs_to_transform, n=n_poly_coeffs)
    print(f"DEBUG DECODE REAL: decoded_scaled_half_spectrum (N/2+1 elementos): {decoded_scaled_half_spectrum[:num_output_elements+1]}")


    # Estamos interessados nos primeiros N/2 componentes, que correspondem aos nossos dados de entrada originais
    # Como os dados originais eram reais, a parte real desses componentes é o que queremos.
    # Pequenos erros numéricos podem introduzir partes imaginárias minúsculas.
    decoded_real_values = np.real(decoded_scaled_half_spectrum[:num_output_elements]) / delta_scale
    
    return decoded_real_values

# --- Criptografia (Encryption)---
def encrypt(message_poly_m, public_key_pk, n_degree, ring_poly_mod, q_mod, sigma_err):
    pk_b, pk_a = public_key_pk
    u_rand_poly = generate_gaussian_poly(n_degree, sigma_err)
    e1_err_poly = generate_gaussian_poly(n_degree, sigma_err)
    e2_err_poly = generate_gaussian_poly(n_degree, sigma_err)
    tmp_pkb_u = pk_b * u_rand_poly
    c0 = poly_ring_mod(tmp_pkb_u + e1_err_poly + message_poly_m, ring_poly_mod, q_mod)
    tmp_pka_u = pk_a * u_rand_poly
    c1 = poly_ring_mod(tmp_pka_u + e2_err_poly, ring_poly_mod, q_mod)
    return c0, c1

# --- Descriptografia (Decryption)---
def decrypt(ciphertext_ct, secret_key_sk_s, ring_poly_mod, q_mod):
    c0, c1 = ciphertext_ct
    c1_s = c1 * secret_key_sk_s
    decrypted_poly_scaled = poly_ring_mod(c0 + c1_s, ring_poly_mod, q_mod)
    coeffs = decrypted_poly_scaled.coef
    corrected_coeffs = np.where(coeffs > q_mod // 2, coeffs - q_mod, coeffs)
    return Polynomial(corrected_coeffs)

def add_homomorphic(ct1, ct2):
    ct10, ct11 = ct1
    ct20, ct21 = ct2

    return ct10 + ct20, ct11 + ct21

def Multevk(c1, c2, evk, P, q, ring_poly):
    """
    Calcula o produto homomórfico de c1 e c2 usando a chave de avaliação evk.
    Implementa a fórmula: cmult = (d0, d1) + floor(P^-1 * d2 * evk)
    """
    # --- Passo 1: Multiplicação Bruta ---
    # c1 = (b1, a1), c2 = (b2, a2)
    b1, a1 = c1
    b2, a2 = c2
    
    # Calcula (d0, d1, d2) = (b1b2, a1b2 + a2b1, a1a2) (mod q)
    d0 = poly_ring_mod(b1 * b2, ring_poly, q)
    d1 = poly_ring_mod(a1 * b2 + a2 * b1, ring_poly, q)
    d2 = poly_ring_mod(a1 * a2, ring_poly, q)

    # --- Passo 2: Relinearização com a evk ---
    evk0, evk1 = evk
    
    # Calcula (d2 * evk) sobre os inteiros (apenas redução pelo anel de polinômios)
    # A lógica de generate_relinkey agora gera a evk em um módulo maior
    d2_evk0_full =  poly_ring_mod(d2 * evk0, ring_poly, q)
    d2_evk1_full = poly_ring_mod(d2 * evk1 , ring_poly, q)

    # Calcula floor(P^-1 * ...) usando aritmética inteira robusta
    corr0_coeffs = d2_evk0_full / P
    corr1_coeffs = d2_evk1_full / P
    
    corr0 = Polynomial(corr0_coeffs.coef.astype(np.int64))
    corr1 = Polynomial(corr1_coeffs.coef.astype(np.int64))

    # Adiciona o termo de correção a (d0, d1) para obter o texto cifrado final
    cmult_b = poly_ring_mod(d0 + corr0_coeffs, ring_poly, q)
    cmult_a = poly_ring_mod(d1 + corr1_coeffs, ring_poly, q)
    
    return (cmult_b, cmult_a)


def rescale(ct, q, delta):
    """Rescala um texto cifrado, dividindo-o por delta e mudando o módulo."""
    c0, c1 = ct
    
    # Centraliza os coeficientes antes da divisão
    q_ = q / delta
    c0_centered = c0 * (q / q_)
    c1_centered = c1 * (q / q_)

    # # A "divisão" robusta é feita com arredondamento
    # c0_rescaled = np.round(c0_centered)
    # c1_rescaled = np.round(c1_centered)
    
    # Retorna o texto cifrado rescalado (agora em um anel com módulo q/delta)
    return (poly_ring_mod(Polynomial(c0_centered.coef.astype(np.int64)), POLY_MOD_RING, q_),
            poly_ring_mod(Polynomial(c1_centered.coef.astype(np.int64)), POLY_MOD_RING, q_))

# --- Demonstração de Uso ---
if __name__ == "__main__":
    print(f"Parâmetros: N={N}, Q={Q}, DELTA={DELTA}, SIGMA={SIGMA}")
    print(f"Anel R_Q = Z_{Q}[X]/(X^{N}+1)")
    print("-" * 40)
    P = 2**16

    # 1. Geração de Chaves
    sk, pk = keygen(N, POLY_MOD_RING, Q, SIGMA)
    evk = relin_keygen(sk, N, Q, P, SIGMA)
    print("Chave Secreta (sk) gerada.")
    print("Chave Pública (pk) gerada.")
    print("-" * 40)

    # 2. Preparar Texto Claro (AGORA N/2 elementos) e Codificar
    num_plaintext_elements = N // 2
    # plaintext_real_vector_half = np.random.rand(num_plaintext_elements) * 2 - 1
    plaintext_real_vector_half = np.array([0.1 * i for i in range(num_plaintext_elements)]) - 0.7
    
    print(f"Vetor de Texto Claro Original (z) ({num_plaintext_elements} elementos): \n{np.round(plaintext_real_vector_half, 3)}")

    encoded_message_poly = ckks_encode_real(plaintext_real_vector_half, DELTA, N)
    print(f"Polinômio da Mensagem Codificada (m) (coefs. primeiros 8): \n{encoded_message_poly.coef[:8]}")
    print("-" * 40)

    # 3. Criptografar
    ct = encrypt(encoded_message_poly, pk, N, POLY_MOD_RING, Q, SIGMA)
    print(f"Mensagem Criptografada (ct = (c0,c1)) {ct[0]}, {ct[1]}.")
    print("-" * 40)

    ct = Multevk(ct, ct, evk, P, Q, POLY_MOD_RING)
    ct = rescale(ct, Q, DELTA)
    Q = Q/DELTA

    # 4. Descriptografar
    decrypted_scaled_poly = decrypt(ct, sk, POLY_MOD_RING, Q)
    print(f"Polinômio Descriptografado Escalonado (m') (coefs. primeiros 8): \n{decrypted_scaled_poly.coef[:8]}")
    print("-" * 40)

    # 5. Decodificar
    decoded_approx_vector_half = ckks_decode_real(decrypted_scaled_poly, DELTA, N)
    print(f"Vetor Decodificado Aproximado (z') ({num_plaintext_elements} elementos): \n{np.round(decoded_approx_vector_half, 3)}")
    print("-" * 40)

    # Comparar original e decodificado
    error = plaintext_real_vector_half * 2 - decoded_approx_vector_half
    print(f"Erro (z - z') ({num_plaintext_elements} elementos): \n{np.round(error, 5)}")
    print(f"Erro Máximo Absoluto: {np.max(np.abs(error)):.5f}")
    print(f"Erro Médio Quadrático: {np.sqrt(np.mean(error**2)):.5f}")
