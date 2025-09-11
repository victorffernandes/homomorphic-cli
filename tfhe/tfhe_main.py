import numpy as np
from .constants import TFHECryptographicParameters

# Criação da instância global dos parâmetros
crypto_params = TFHECryptographicParameters()

# Inicialização das variáveis usando os parâmetros centralizados
secret = crypto_params.create_secret_key()
rng = crypto_params.get_random_generator()
xN_1 = crypto_params.get_polynomial_modulus()


def error():
    return (
        rng.normal(
            scale=crypto_params.NOISE_PARAMETER, size=(crypto_params.POLYNOMIAL_DEGREE)
        )
        % 1
    )


def tlwe(s):
    lwe_dim = crypto_params.LWE_DIMENSION
    poly_deg = crypto_params.POLYNOMIAL_DEGREE

    a = np.array(rng.uniform(low=0, high=1, size=(lwe_dim, poly_deg))) % 1
    b = np.array([0])
    for ax in a:
        product = np.polymul(np.poly1d(ax), np.poly1d(s))
        b = np.polyadd(b, product)
    b = np.polyadd(b, error())
    b = mod_xN_1(b)

    z = np.append(a, [b])
    z.shape = (lwe_dim + 1, poly_deg)

    return z


def trivial_tlwe(m):
    lwe_dim = crypto_params.LWE_DIMENSION
    poly_deg = crypto_params.POLYNOMIAL_DEGREE

    a = np.array(np.zeros(shape=(lwe_dim, poly_deg)))
    b = np.array(
        [m] + [0] * (poly_deg - 1)
    )  # b tem tamanho N com m na primeira posição

    a = np.append(a, [b], axis=0)

    return a


def sum_tlwe(s1, s2):
    lwe_dim = crypto_params.LWE_DIMENSION
    poly_deg = crypto_params.POLYNOMIAL_DEGREE

    a = np.array(np.zeros(shape=(lwe_dim + 1, poly_deg)))
    for i in range(lwe_dim):
        a[i] = mod_xN_1(np.polyadd(np.poly1d(s1[i]), np.poly1d(s2[i]))).coeffs

    # print("sum tlwe")
    # print(np.poly1d(a[0]), sum_b)

    return a


def phase_s(sample, s):
    sa = np.array([0])
    a = sample[:-1]
    b = sample[-1]
    for ax in a:
        product = np.polymul(np.poly1d(ax), np.poly1d(s))
        sa = np.polyadd(sa, product)
    r = np.polysub(b, sa)
    r = mod_xN_1(r)
    return r.coeffs


def mod_xN_1(P):
    z = np.poly1d(P.coeffs % 1)
    _, resto = np.polydiv(z, xN_1)
    return np.poly1d(resto)


def decompose(a, bg, level_count):  # a kxN
    a_ = []
    for a_i in a:
        a_t = a_i
        m = 1 / bg**level_count
        a_t = np.around(a_i * bg**level_count) * m
        a_.append(a_t)

    r = []
    for a_i in a_:
        a__ = []
        for a_ij in a_i:
            a_ijp = []
            residual = a_ij
            for p in range(level_count):
                z = round(residual * bg**p)
                a_ijp.append(z)
                residual = residual - (z / bg**p)
            a__.append(a_ijp)
        r.append(a__)
    u = [] * (crypto_params.LWE_DIMENSION + 1)
    for i in range(crypto_params.LWE_DIMENSION + 1):
        u.append([])
        for p in range(level_count):
            f = []
            for j in range(crypto_params.POLYNOMIAL_DEGREE):
                f.append(r[i][j][p])
            u[i].append(f)
    return u


def norm(m, level_count):
    r = []
    for i in range(crypto_params.LWE_DIMENSION + 1):
        z = []
        for p in range(level_count):
            z.append(np.linalg.vector_norm(m[i][p]))
        r.append(z)
    print(z)


def H(level_count, bg):
    lwe_dim = crypto_params.LWE_DIMENSION
    l_ = (lwe_dim + 1) * level_count
    h = np.zeros(shape=(l_, lwe_dim + 1))
    for i in range(lwe_dim + 1):
        for j in range(level_count):
            h[i * level_count + j][i] = 1 / bg**j
    return h


def tgsw(m, level_count, H):
    lwe_dim = crypto_params.LWE_DIMENSION
    l_ = (lwe_dim + 1) * level_count

    tgsw_s = []
    tlwe_s = []  # (k+1)*l, k+1
    # H_m = H * m  # (k+1)*l, k+1 - Comentado - não utilizado

    for i in range(l_):
        tlwe_s.append(tlwe(secret))
        # sum_result = sum_tlwe(tlwe_s[:-1], H_m[i])  # Comentado - não utilizado
        # tgsw_s.append(sum_tlwe(tlwe_s[i], H_m[i]))
        tgsw_s.append(tlwe_s[i])

    # print(tlwe_s)
    # print(H_m)
    # print(tgsw_s)
    return tgsw_s


def phase_tgsw(tgsw_s, secret, level_count):
    lwe_dim = crypto_params.LWE_DIMENSION
    msg = []
    for i in range((lwe_dim + 1) * level_count):
        msg.append(phase_s(tgsw_s[i], secret))
    return msg


# Configuração de exemplo
level_count = 4

if __name__ == "__main__":
    # Demonstração de uso
    crypto_params.print_parameters_summary()

    # Exemplo de uso comentado para evitar erro durante import
    # s = tgsw(0, level_count, H(level_count, 2))
    # print(phase_tgsw(s, secret, level_count))
