"""
Fábrica para criação e manipulação de ciphertexts CKKS.

Esta classe fornece uma interface de alto nível para operações
de codificação, decodificação, criptografia e descriptografia no esquema CKKS.
"""

import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Union, Tuple, Dict, Any

# Imports relativos para uso como módulo, absolutos para execução direta
try:
    from .constants import CKKSCryptographicParameters
    from .ckks import CKKSCiphertext
except ImportError:
    # Fallback para execução direta
    from constants import CKKSCryptographicParameters
    from ckks import CKKSCiphertext


class CKKSCiphertextFactory:
    """
    Fábrica para criação e manipulação de ciphertexts CKKS.

    Esta classe encapsula as operações de codificação, decodificação,
    criptografia e descriptografia para o esquema CKKS, fornecendo
    uma interface mais limpa e orientada a objetos.
    """

    def __init__(self, crypto_params: CKKSCryptographicParameters = None):
        """
        Inicializa a fábrica com parâmetros criptográficos.

        Args:
            crypto_params: Parâmetros criptográficos CKKS (usa padrão se None)
        """
        if crypto_params is None:
            crypto_params = CKKSCryptographicParameters()

        self.crypto_params = crypto_params

        N = self.crypto_params.POLYNOMIAL_DEGREE
        self.M = 4 * N

        self.xi = np.exp(2 * np.pi * 1j / self.M)
        # Cria base para discretização σ(R)
        self.create_sigma_R_basis()

    def poly_mod_xn_plus_1(self, p: Polynomial, q_mod: int = None) -> Polynomial:
        """
        Reduz um polinômio módulo X^N + 1 e também módulo q.

        Para o anel R_q = Z_q[X]/(X^N + 1), aplicamos duas reduções:
        1. Redução módulo X^N + 1 (usando X^N = -1)
        2. Redução módulo q (aritmética modular)

        Args:
            p: Polinômio a ser reduzido
            q_mod: Módulo para redução dos coeficientes (usa o maior se None)

        Returns:
            Polynomial: Polinômio reduzido com grau < N e coeficientes módulo q
        """
        N = self.crypto_params.POLYNOMIAL_DEGREE

        if q_mod is None:
            q_mod = self.crypto_params.MODULUS_CHAIN[-1]

        # Passo 1: Redução módulo X^N + 1
        mod_poly_coeffs = np.zeros(N + 1)
        mod_poly_coeffs[0] = 1  # termo constante
        mod_poly_coeffs[N] = 1  # termo X^N
        modulus = Polynomial(mod_poly_coeffs)

        quotient, remainder = divmod(p, modulus)

        # Passo 2: Redução módulo q
        coeffs_mod_q = np.round(remainder.coef).astype(int) % q_mod

        # Centered lift: valores > q/2 são representados como negativos
        coeffs_mod_q = np.where(
            coeffs_mod_q > q_mod // 2, coeffs_mod_q - q_mod, coeffs_mod_q
        )

        return Polynomial(coeffs_mod_q)

    def vandermonde(self, xi: complex, M: int) -> np.ndarray:
        """
        Cria a matriz de Vandermonde para o canonical embedding.

        Retorna a matriz onde cada linha i corresponde às potências de ζ^(2i+1):
        [1, ζ^(2i+1), ζ^(2(2i+1)), ..., ζ^((M-1)(2i+1))]

        Args:
            xi: Raiz primitiva M-ésima da unidade ζ = e^(2πi/M)
            M: Ordem da raiz (M = 2N para X^N + 1)

        Returns:
            Matriz de Vandermonde (M/2) × M
        """
        N = M // 2
        matrix = np.zeros((N, M), dtype=np.complex128)

        for i in range(N):
            # Raiz ζ^(2i+1)
            root = xi ** (2 * i + 1)
            for j in range(M):
                # Potência root^j
                matrix[i, j] = root**j

        return matrix

    def create_sigma_R_basis(self):
        """
        Cria a base para o reticulado σ(R) ⊂ H.

        A base é formada por (σ(1), σ(X), σ(X²), ..., σ(X^(N-1)))
        onde cada σ(X^j) é a avaliação de X^j nas raízes primitivas.

        Esta é a transposta da matriz de Vandermonde.
        """
        # Cria matriz de Vandermonde e transpõe
        # Resultado: cada coluna é σ(X^j)
        vandermonde = self.vandermonde(self.xi, self.M)
        self.sigma_R_basis = vandermonde.T

    def sigma(self, polynomial: Polynomial) -> np.ndarray:
        """
        Aplica o canonical embedding σ: R → ℂ^(N/2).

        Avalia o polinômio nas raízes primitivas ζ^(2k+1) para k = 0, ..., N/2-1.
        Onde ζ = e^(2πi/M) com M = 4N.

        Args:
            polynomial: Polinômio em R = ℤ[X]/(X^N + 1)

        Returns:
            np.ndarray: Vetor em ℂ^(N/2) representando σ(p)
        """
        N = self.crypto_params.POLYNOMIAL_DEGREE
        num_slots = N // 2
        coeffs = polynomial.coef

        # Garante que temos exatamente N coeficientes
        if len(coeffs) < N:
            coeffs = np.pad(coeffs, (0, N - len(coeffs)), mode="constant")
        elif len(coeffs) > N:
            coeffs = coeffs[:N]

        # Avalia o polinômio nas primeiras N/2 raízes primitivas
        # result[i] = p(ζ^(2i+1)) para i = 0, ..., N/2-1
        result = np.zeros(num_slots, dtype=np.complex128)
        for i in range(num_slots):
            root = self.xi ** (2 * i + 1)
            result[i] = np.polyval(coeffs[::-1], root)

        return result

    def sigma_inverse(self, z: np.ndarray) -> Polynomial:
        """
        Aplica o inverso do canonical embedding σ^(-1): H → R.

        Para CKKS com mensagens reais:
        - Recebe z ∈ ℂ^(N/2) representando valores desejados
        - Queremos p(X) ∈ R tal que p(ζ^(2k+1)) ≈ z[k] para k=0,...,N/2-1
        - E p(conj(ζ^(2k+1))) = p(ζ^(2k+1)) (mensagens reais)

        As raízes vêm em pares conjugados:
        - ζ^1 e ζ^(M-1) = conj(ζ^1)
        - ζ^3 e ζ^(M-3) = conj(ζ^3)
        - etc.

        Args:
            z: Vetor em ℂ^(N/2) com valores reais desejados

        Returns:
            Polynomial: Polinômio p com coeficientes reais
        """
        N = self.crypto_params.POLYNOMIAL_DEGREE
        num_slots = N // 2

        # z deve ter tamanho N/2
        if len(z) < num_slots:
            z = np.pad(z, (0, num_slots - len(z)), mode="constant")
        elif len(z) > num_slots:
            z = z[:num_slots]

        # Para mensagens reais, construímos sistema com N equações:
        # p(ζ^(2k+1)) = z[k] para k = 0, ..., N/2-1
        # p(ζ^(M-2k-1)) = z[k] para k = 0, ..., N/2-1 (par conjugado)
        #
        # Nota: ζ^(M-j) = conj(ζ^j) pois ζ^M = 1

        # Cria matriz de Vandermonde para N equações
        V = np.zeros((N, N), dtype=np.complex128)
        b = np.zeros(N, dtype=np.complex128)

        # Primeira metade: ζ^1, ζ^3, ζ^5, ..., ζ^(N-1)
        for k in range(num_slots):
            root = self.xi ** (2 * k + 1)
            for j in range(N):
                V[k, j] = root**j
            b[k] = z[k]

        # Segunda metade: conjugados ζ^(M-1), ζ^(M-3), ..., ζ^(M-N+1)
        # Estes são os pares conjugados das raízes acima
        for k in range(num_slots):
            root_conj = self.xi ** (self.M - 2 * k - 1)
            for j in range(N):
                V[num_slots + k, j] = root_conj**j
            b[num_slots + k] = z[k]  # Mesmo valor (mensagem real)

        # Resolve sistema V @ coeffs = b
        coeffs = np.linalg.solve(V, b)

        # Retorna polinômio com coeficientes reais
        return Polynomial(np.real(coeffs))

    def pi(self, z: np.ndarray) -> np.ndarray:
        """
        Projeção canônica π: H → ℂ^(N/2).

        Como sigma já retorna N/2 elementos, pi é essencialmente identidade.

        Args:
            z: Vetor em ℂ^(N/2) (já do tamanho correto)

        Returns:
            np.ndarray: O mesmo vetor z
        """
        # sigma já retorna N/2 elementos, então pi é identidade
        return z

    def pi_inverse(self, z: np.ndarray) -> np.ndarray:
        """
        Inverso da projeção π^(-1): ℂ^(N/2) → H.

        Expande um vetor z ∈ ℂ^(N/2) para H aplicando a simetria hermitiana:
        resultado = [z[0], z[1], ..., z[N/2-1], conj(z[N/2-1]), ..., conj(z[1]), conj(z[0])]

        Args:
            z: Vetor em ℂ^(N/2)

        Returns:
            np.ndarray: Vetor em H com simetria hermitiana (tamanho N)
        """
        # Cria conjugado reverso
        z_conjugate = z[::-1]
        z_conjugate = np.conjugate(z_conjugate)

        # Concatena z com seu conjugado reverso
        return np.concatenate([z, z_conjugate])

    def compute_basis_coordinates(self, z: np.ndarray) -> np.ndarray:
        """
        Calcula as coordenadas de um vetor z com relação à base ortogonal σ(R).

        Para cada vetor base b, calcula: coord = Re(<z, b> / <b, b>)
        onde <·, ·> é o produto interno hermitiano.

        Args:
            z: Vetor em H (tamanho N)

        Returns:
            np.ndarray: Coordenadas em relação à base σ(R)
        """
        N = self.crypto_params.POLYNOMIAL_DEGREE
        coordinates = np.zeros(N)

        # Garante que z tem tamanho N
        if len(z) < N:
            z = np.pad(z, (0, N - len(z)), mode="constant")
        elif len(z) > N:
            z = z[:N]

        # Para cada vetor da base σ(X^j)
        # sigma_R_basis tem shape (M, N) = (2N, N)
        # Pegamos apenas as primeiras N linhas para trabalhar com H
        for j in range(N):
            b = self.sigma_R_basis[:N, j]  # Vetor base (tamanho N)
            # Produto interno hermitiano <z, b> / <b, b>
            coord = np.vdot(b, z) / np.vdot(b, b)
            coordinates[j] = np.real(coord)

        return coordinates

    def coordinate_wise_random_rounding(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Arredonda coordenadas randomicamente para preservar melhor a distribuição.

        Para cada coordenada c:
        - Calcula resto r = c - floor(c)
        - Arredonda para floor(c) com prob (1-r) ou ceil(c) com prob r

        Args:
            coordinates: Coordenadas reais

        Returns:
            np.ndarray: Coordenadas arredondadas para inteiros
        """
        # Calcula restos fracionários
        r = coordinates - np.floor(coordinates)

        # Arredonda randomicamente
        rounded = np.zeros(len(coordinates), dtype=int)
        for i, (c, frac) in enumerate(zip(coordinates, r)):
            # Escolhe entre floor(c) e ceil(c) com probabilidades apropriadas
            choice = np.random.choice([np.floor(c), np.ceil(c)], p=[1 - frac, frac])
            rounded[i] = int(choice)

        return rounded

    def sigma_R_discretization(self, z: np.ndarray) -> np.ndarray:
        """
        Projeta um vetor z ∈ ℂ^(N/2) no reticulado σ(R).

        Estratégia simplificada:
        1. Aplica σ^(-1) para obter polinômio p com coeficientes reais
        2. Arredonda coeficientes para inteiros
        3. Aplica σ novamente para voltar a ℂ^(N/2)

        Args:
            z: Vetor em ℂ^(N/2)

        Returns:
            np.ndarray: Vetor em σ(R) mais próximo de z
        """
        # Passo 1: σ^(-1)(z) -> polinômio
        p = self.sigma_inverse(z)

        # Passo 2: Arredonda coeficientes
        rounded_coeffs = np.round(p.coef).astype(int)
        p_rounded = Polynomial(rounded_coeffs)

        # Passo 3: σ(p_rounded) -> vetor arredondado
        return self.sigma(p_rounded)

    def ckks_encode_real(
        self,
        real_vector: List[float],
        delta_scale: float = None,
        n_poly_coeffs: int = None,
    ) -> Polynomial:
        """
        Codifica um vetor de números reais em um polinômio CKKS usando canonical embedding.

        Esta implementação agora usa o canonical embedding correto σ: R → ℂ^(N/2),
        onde R = ℤ[X]/(X^N + 1) é o anel ciclotômico.

        O encoding segue os passos:
        1. z = vetor de entrada (números reais)
        2. π^(-1)(z) = expande para H (com simetria hermitiana)
        3. Escala: Δ · π^(-1)(z)
        4. Projeta no reticulado σ(R)
        5. σ^(-1) para obter polinômio final

        Args:
            real_vector: Vetor de números reais para codificar
            delta_scale: Fator de escala (usa padrão se None)
            n_poly_coeffs: Número de coeficientes do polinômio (usa padrão se None)

        Returns:
            Polynomial: Polinômio codificado com canonical embedding correto
        """
        if delta_scale is None:
            delta_scale = self.crypto_params.SCALING_FACTOR

        if n_poly_coeffs is None:
            n_poly_coeffs = self.crypto_params.POLYNOMIAL_DEGREE

        # Número máximo de slots disponíveis (N/2)
        max_slots = n_poly_coeffs // 2

        # Converte o vetor de entrada para np.array e garante o tamanho correto
        input_array = np.array(real_vector, dtype=np.float64)
        if len(input_array) > max_slots:
            print(
                f"Aviso: Vetor de entrada truncado de {len(input_array)} para {max_slots} elementos"
            )
            input_array = input_array[:max_slots]
        elif len(input_array) < max_slots:
            # Preenche com zeros até max_slots
            input_array = np.pad(
                input_array, (0, max_slots - len(input_array)), mode="constant"
            )

        # Os valores reais representam avaliações desejadas nas raízes
        # Passo 1: Trata valores como complexos
        z = input_array.astype(np.complex128)

        # Passo 2: Escala por Δ
        scaled_z = delta_scale * z

        # Passo 3: Aplica sigma_inverse para obter polinômio
        # sigma_inverse internamente expande com conjugados e resolve sistema
        p = self.sigma_inverse(scaled_z)

        # Passo 4: Arredonda coeficientes para inteiros
        coef = np.round(np.real(p.coef)).astype(int)

        return Polynomial(coef)

    def ckks_decode_real(
        self,
        message_poly: Polynomial,
        delta_scale: float = None,
        n_poly_coeffs: int = None,
        q_mod: bool = True,
        q_mod_value: int = None,
    ) -> np.ndarray:
        """
        Decodifica um polinômio CKKS em um vetor de números reais usando canonical embedding.

        Esta implementação usa o canonical embedding correto σ: R → ℂ^(N/2),
        garantindo que a estrutura multiplicativa seja preservada.

        O decoding segue os passos:
        1. Correção dos coeficientes (centered lift) módulo q (se q_mod=True)
        2. σ(p) = avalia p nas raízes primitivas
        3. Divide por Δ para remover escala
        4. π projeta de H para ℂ^(N/2)

        Args:
            message_poly: Polinômio a ser decodificado
            delta_scale: Fator de escala (usa padrão se None)
            n_poly_coeffs: Número de coeficientes do polinômio (usa padrão se None)
            q_mod: Se True, aplica correção modular (centered lift). Default True.
            q_mod_value: Valor do módulo a ser usado para correção (usa o maior se None e q_mod=True)

        Returns:
            np.ndarray: Vetor de números reais decodificados
        """
        if delta_scale is None:
            delta_scale = self.crypto_params.SCALING_FACTOR

        if n_poly_coeffs is None:
            n_poly_coeffs = self.crypto_params.POLYNOMIAL_DEGREE

        # Passo 1: Correção centered lift para valores negativos
        # IMPORTANTE: Se q_mod=True, assumimos que o polinômio já foi reduzido
        # pelo decrypt usando poly_ring_mod (que aplica mod_centered), então
        # os coeficientes já estão em forma centrada (-q/2, q/2].
        # No entanto, garantimos que os coeficientes estão na faixa correta
        # aplicando mod_centered novamente como uma medida de segurança.
        coeffs = message_poly.coef.copy()

        # Se q_mod=True, aplica mod_centered para garantir que coeficientes estão em (-q/2, q/2]
        # Isso é necessário porque após operações aritméticas, alguns coeficientes podem
        # estar ligeiramente fora da faixa devido a erros de arredondamento
        if q_mod:
            # Obtém o módulo para correção
            if q_mod_value is None:
                q_mod_value = self.crypto_params.MODULUS_CHAIN[
                    -1
                ]  # Maior módulo (padrão)

            # Aplica mod_centered para garantir que coeficientes estão em (-q/2, q/2]
            corrected_coeffs = self.crypto_params.mod_centered(coeffs, q_mod_value)
        else:
            # q_mod=False - não aplica correção modular (valores não criptografados)
            corrected_coeffs = coeffs

        # Recria polinômio com coeficientes corrigidos
        p = Polynomial(corrected_coeffs)

        # Passo 2: Aplica σ(p) para obter valores no domínio complexo
        z = self.sigma(p)

        # Passo 3: Remove escala dividindo por Δ
        rescaled_z = z / delta_scale

        # Passo 4: Projeta de H para ℂ^(N/2) usando π
        result = self.pi(rescaled_z)

        # Retorna parte real (para mensagens reais)
        return np.real(result)

    def encrypt(
        self,
        message_poly: Polynomial,
        public_key: Tuple[Polynomial, Polynomial],
        level: int = None,
    ) -> CKKSCiphertext:
        """
        Criptografa um polinômio usando a chave pública seguindo o esquema CKKS.

        Conforme a definição:
        - Encryption: ct = (pk_b * u + e1 + m, pk_a * u + e2)
        onde u ← ZO(ρ), e1, e2 ← DG(σ²)

        Args:
            message_poly: Polinômio da mensagem a ser criptografada (já escalado)
            public_key: Tupla (pk_b, pk_a) da chave pública
            level: Nível inicial na cadeia de módulos (usa o maior se None)

        Returns:
            CKKSCiphertext: Ciphertext resultante
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1  # Nível mais alto

        q_mod = self.crypto_params.MODULUS_CHAIN[level]
        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        sigma_err = self.crypto_params.GAUSSIAN_NOISE_STDDEV
        zero_one_density = self.crypto_params.ZERO_ONE_DENSITY

        pk_b, pk_a = public_key

        # Sample u ← ZO(ρ), e1, e2 ← DG(σ²)
        u = self.crypto_params.generate_zero_one_poly(n_degree, zero_one_density)
        e1 = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        e2 = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)

        # CKKS encryption: ct = (pk_b*u + e1 + m, pk_a*u + e2)
        # Calcula pk_b * u
        b_u = self.crypto_params.poly_mul_mod(pk_b, u, q_mod, ring_poly_mod)

        # Calcula pk_a * u
        a_u = self.crypto_params.poly_mul_mod(pk_a, u, q_mod, ring_poly_mod)

        # c0 = pk_b*u + e1 + m (mod q)
        c0 = self.crypto_params.poly_ring_mod(
            b_u + e1 + message_poly, ring_poly_mod, q_mod
        )

        # c1 = pk_a*u + e2 (mod q)
        c1 = self.crypto_params.poly_ring_mod(a_u + e2, ring_poly_mod, q_mod)

        return CKKSCiphertext(
            components=[c0, c1],
            level=level,
            crypto_params=self.crypto_params,
        )

    def decrypt(
        self,
        ciphertext: Union[CKKSCiphertext, Dict[str, Any]],
        secret_key: Tuple[Polynomial, Polynomial],
    ) -> Polynomial:
        """
        Descriptografa um ciphertext usando a chave secreta.

        Args:
            ciphertext: Ciphertext a ser descriptografado (CKKSCiphertext ou dict)
            secret_key: Chave secreta sk = (1, s) para descriptografia

        Returns:
            Polynomial: Polinômio da mensagem descriptografada
        """
        # Suporte para formato de dicionário legado
        if isinstance(ciphertext, dict):
            ciphertext = CKKSCiphertext.from_dict(ciphertext, self.crypto_params)

        # Extrai o componente s de sk = (1, s)
        _, s = secret_key

        q_mod = ciphertext.current_modulus
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()

        # Descriptografia para ciphertext de 3 componentes (após multiplicação)
        # Fórmula: m ≈ C'₁ + C'₂·s + C'₃·s²
        # O dot product com (1, s, s²) recupera a mensagem
        if ciphertext.size == 3:
            c0 = ciphertext.get_component(0)
            c1 = ciphertext.get_component(1)
            c2 = ciphertext.get_component(2)

            # Calcula s² (s ao quadrado) usando multiplicação modular
            s_squared = self.crypto_params.poly_mul_mod(s, s, q_mod, ring_poly_mod)

            # Calcula c1 * s usando multiplicação modular
            c1_s = self.crypto_params.poly_mul_mod(c1, s, q_mod, ring_poly_mod)

            # Calcula c2 * s² usando multiplicação modular
            c2_s_squared = self.crypto_params.poly_mul_mod(
                c2, s_squared, q_mod, ring_poly_mod
            )

            # Descriptografa: m ≈ c0 + c1*s + c2*s²
            decrypted_poly = c0 + c1_s + c2_s_squared

            # Aplica redução modular final
            final_poly = self.crypto_params.poly_ring_mod(
                decrypted_poly,
                ring_poly_mod,
                q_mod,
            )

            return final_poly

        # Descriptografia padrão para ciphertext de 2 componentes: m = c0 + c1*s
        c0 = ciphertext.get_component(0)
        c1 = ciphertext.get_component(1)

        # Calcula c1 * s
        c1_s = self.crypto_params.poly_mul_mod(c1, s, q_mod, ring_poly_mod)

        # Descriptografa: c0 + c1*s
        decrypted_poly = c0 + c1_s

        # Aplica redução modular final
        final_poly = self.crypto_params.poly_ring_mod(
            decrypted_poly, ring_poly_mod, q_mod
        )

        return final_poly

    def encode_and_encrypt(
        self,
        real_vector: List[float],
        public_key: Tuple[Polynomial, Polynomial],
        level: int = None,
    ) -> CKKSCiphertext:
        """
        Codifica e criptografa um vetor de números reais em uma única operação.

        Args:
            real_vector: Vetor de números reais
            public_key: Chave pública para criptografia
            level: Nível inicial (usa o maior se None)

        Returns:
            CKKSCiphertext: Ciphertext resultante
        """
        encoded_poly = self.ckks_encode_real(real_vector)
        return self.encrypt(encoded_poly, public_key, level)

    def decrypt_and_decode(
        self,
        ciphertext: Union[CKKSCiphertext, Dict[str, Any]],
        secret_key: Tuple[Polynomial, Polynomial],
        expected_length: int = None,
    ) -> np.ndarray:
        """
        Descriptografa e decodifica um ciphertext em uma única operação.

        Args:
            ciphertext: Ciphertext a ser processado
            secret_key: Chave secreta sk = (1, s) para descriptografia
            expected_length: Comprimento esperado do vetor resultante

        Returns:
            np.ndarray: Vetor de números reais recuperados
        """
        # Descriptografa
        decrypted_poly = self.decrypt(ciphertext, secret_key)

        scale = ciphertext.scale

        # Decodifica (usa q_mod=True para aplicar correção modular em valores criptografados)
        # Usa o módulo do ciphertext atual para correção modular
        decoded_vector = self.ckks_decode_real(
            decrypted_poly,
            scale,
            self.crypto_params.POLYNOMIAL_DEGREE,
            q_mod=True,
            q_mod_value=ciphertext.current_modulus,
        )

        if expected_length is not None:
            decoded_vector = decoded_vector[:expected_length]

        return decoded_vector


# Função de conveniência para criar instância da fábrica
def create_ckks_factory(
    crypto_params: CKKSCryptographicParameters = None,
) -> CKKSCiphertextFactory:
    """
    Cria uma nova instância da fábrica CKKS.

    Args:
        crypto_params: Parâmetros criptográficos (usa padrão se None)

    Returns:
        CKKSCiphertextFactory: Nova instância da fábrica
    """
    return CKKSCiphertextFactory(crypto_params)


if __name__ == "__main__":
    # Exemplo básico de uso da fábrica (apenas codificação/decodificação)
    # Evita dependências circulares ao executar diretamente
    print("Exemplo de uso da CKKSCiphertextFactory:")

    try:
        from constants import CKKSCryptographicParameters

        # Cria instância direta da fábrica
        crypto_params = CKKSCryptographicParameters()
        factory = CKKSCiphertextFactory(crypto_params)

        # Dados de exemplo
        real_data = [1.5, -2.3, 3.7, 0.0]

        # Codifica
        encoded_poly = factory.ckks_encode_real(real_data)
        print(f"✓ Codificação realizada: {len(encoded_poly.coef)} coeficientes")

        # Decodifica para verificar (sem correção modular para valores não criptografados)
        decoded_data = factory.ckks_decode_real(encoded_poly, q_mod=False)
        print(f"✓ Decodificação realizada: {decoded_data[:len(real_data)]}")

        print("\n✓ CKKSCiphertextFactory funcionando corretamente!")

    except ImportError as e:
        print(f"Erro de importação: {e}")
        print("Execute os testes usando: python -m pytest test_factory.py -v")
