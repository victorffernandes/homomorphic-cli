"""
Constantes centralizadas para o esquema de criptografia homomórfica CKKS.

Esta classe organiza todos os parâmetros criptográficos de forma semântica
para facilitar manutenção e configuração do sistema.
"""

import numpy as np
from numpy.polynomial import Polynomial


class CKKSCryptographicParameters:
    """
    Classe que centraliza todos os parâmetros criptográficos do esquema CKKS.

    Esta classe organiza as constantes de forma semântica, separando:
    - Parâmetros de segurança
    - Parâmetros de precisão
    - Estruturas algébricas
    - Configurações de ruído

    A classe agora suporta parametrização no construtor seguindo a definição KeyGen(1λ).
    """

    def __init__(
        self,
        polynomial_degree: int = 8,  # N - grau do polinômio ciclotômico (REDUZIDO: era 16)
        q0_base: int = 2**10,  # Q0 - módulo base (~1K) (REDUZIDO: era 2^12)
        scaling_factor: int = 2**4,  # DELTA - fator de escala (16) (REDUZIDO: era 2^6)
        gaussian_noise_stddev: float = 0,  # σ - desvio padrão gaussiano
        hamming_weight: int = None,  # h - peso de Hamming (padrão: N/4)
        zero_one_density: float = 0.5,  # ρ - densidade ZO
        total_levels: int = 2,  # Número de níveis na cadeia de módulos
    ):
        """
        Inicializa os parâmetros criptográficos CKKS.

        Conforme definição KeyGen(1λ):
        - Escolhe M = M(λ, qL), h = h(λ, qL), P = P(λ, qL), σ = σ(λ, qL)

        Args:
            polynomial_degree: N - deve ser potência de 2
            q0_base: Q0 - módulo base para construir cadeia
            scaling_factor: DELTA - fator de escala para codificação
            gaussian_noise_stddev: σ - desvio padrão para DG(σ²)
            hamming_weight: h - peso de Hamming para HWT(h) (padrão: N/4)
            zero_one_density: ρ - densidade para ZO(ρ)
            total_levels: Número de níveis na cadeia de módulos
        """
        # === PARÂMETROS ESTRUTURAIS ===
        self.POLYNOMIAL_DEGREE = polynomial_degree  # N

        self.Q0 = q0_base  # Q0

        # === MÓDULOS DA CADEIA CKKS ===
        self.MODULUS_CHAIN = [
            scaling_factor**i * self.Q0 for i in range(0, total_levels + 1)
        ]

        self.P = self.Q0 * 2

        # === PARÂMETROS DE ESCALA ===
        self.SCALING_FACTOR = scaling_factor  # DELTA

        # === PARÂMETROS DE RUÍDO ===
        self.GAUSSIAN_NOISE_STDDEV = gaussian_noise_stddev  # σ

        # === PARÂMETROS PARA DISTRIBUIÇÕES DE CHAVE ===
        if hamming_weight is None:
            hamming_weight = max(1, polynomial_degree // 4)  # Padrão: N/4
        self.HAMMING_WEIGHT = hamming_weight  # h
        self.ZERO_ONE_DENSITY = zero_one_density  # ρ

        # Validar parâmetros após inicialização
        self.validate_parameters()

    # === ESTRUTURAS ALGÉBRICAS ===
    def get_polynomial_modulus_ring(self):
        """
        Retorna o anel de polinômios modulares X^N + 1.

        Returns:
            Polynomial: O polinômio ciclotômico X^N + 1
        """
        return Polynomial([1] + [0] * (self.POLYNOMIAL_DEGREE - 1) + [1])

    # === MÉTODOS DE ACESSO ===
    def get_initial_modulus(self):
        """
        Retorna o módulo inicial (último da cadeia).

        Returns:
            int: O maior módulo da cadeia
        """
        return self.MODULUS_CHAIN[-1]

    def get_maximum_plaintext_slots(self):
        """
        Retorna o número máximo de slots disponíveis para texto claro.

        Returns:
            int: Número de slots (N/2)
        """
        return self.POLYNOMIAL_DEGREE // 2

    def get_scaling_factor_squared(self):
        """
        Retorna o quadrado do fator de escala, usado após multiplicação.

        Returns:
            int: DELTA²
        """
        return self.SCALING_FACTOR**2

    def validate_parameters(self):
        """
        Valida a consistência dos parâmetros criptográficos.

        Raises:
            ValueError: Se algum parâmetro estiver inconsistente
        """
        if (
            self.POLYNOMIAL_DEGREE <= 0
            or (self.POLYNOMIAL_DEGREE & (self.POLYNOMIAL_DEGREE - 1)) != 0
        ):
            raise ValueError("POLYNOMIAL_DEGREE deve ser uma potência de 2 positiva")

        if not self.MODULUS_CHAIN or len(self.MODULUS_CHAIN) == 0:
            raise ValueError("MODULUS_CHAIN não pode estar vazia")

        if self.SCALING_FACTOR <= 0:
            raise ValueError("SCALING_FACTOR deve ser positivo")

        # Verifica se os módulos estão em ordem crescente
        for i in range(1, len(self.MODULUS_CHAIN)):
            if self.MODULUS_CHAIN[i] <= self.MODULUS_CHAIN[i - 1]:
                raise ValueError("MODULUS_CHAIN deve estar em ordem crescente")

    def get_security_level_estimate(self):
        """
        Retorna uma estimativa do nível de segurança baseado nos parâmetros.

        Returns:
            int: Estimativa do nível de segurança em bits
        """
        # Estimativa simplificada baseada no grau do polinômio
        # Para uma análise real, seria necessário considerar outros fatores
        return int(np.log2(self.POLYNOMIAL_DEGREE) * 10)

    def print_parameters_summary(self):
        """
        Imprime um resumo dos parâmetros configurados.
        """
        print("=== PARÂMETROS CRIPTOGRÁFICOS CKKS ===")
        print(f"Grau do polinômio (N): {self.POLYNOMIAL_DEGREE}")
        print(
            f"Fator de escala (DELTA): {self.SCALING_FACTOR} (~2^{int(np.log2(self.SCALING_FACTOR))})"
        )
        print(f"Desvio padrão do ruído (SIGMA): {self.GAUSSIAN_NOISE_STDDEV}")
        print(f"Cadeia de módulos: {len(self.MODULUS_CHAIN)} níveis")
        print(
            f"  - Maior módulo: {self.MODULUS_CHAIN[-1]} "
            f"(~{self.MODULUS_CHAIN[-1].bit_length()} bits)"
        )
        print(
            f"  - Menor módulo: {self.MODULUS_CHAIN[0]} "
            f"(~{self.MODULUS_CHAIN[0].bit_length()} bits)"
        )
        print(f"Slots disponíveis: {self.get_maximum_plaintext_slots()}")
        print(
            f"Nível de segurança estimado: ~{self.get_security_level_estimate()} bits"
        )
        print(f"Peso de Hamming (h): {self.HAMMING_WEIGHT}")
        print(f"Densidade ZO (ρ): {self.ZERO_ONE_DENSITY}")
        print("=" * 40)

    # === FUNÇÕES AUXILIARES PARA OPERAÇÕES POLINOMIAIS ===
    @staticmethod
    def mod_centered(value, modulus):
        reduced = np.mod(value, modulus)

        half_modulus = modulus / 2

        if np.isscalar(reduced):
            if reduced > half_modulus:
                return reduced - modulus
            return reduced
        else:
            # Para arrays
            result = reduced.copy()
            mask = result > half_modulus
            result[mask] = result[mask] - modulus
            return result

    @staticmethod
    def poly_coeffs_mod_q(p_numpy, q_coeff):
        """
        Aplica operação modular centrada aos coeficientes de um polinômio.

        Para um polinômio r, [r]_a denota a mesma operação aplicada coeficiente a coeficiente,
        onde cada coeficiente é reduzido para ℤ_a = (-a/2, a/2].

        Args:
            p_numpy: Polinômio numpy
            q_coeff: Coeficiente modular a

        Returns:
            Polynomial: Polinômio com coeficientes em ℤ_a = (-a/2, a/2]
        """
        coeffs = p_numpy.coef.copy()
        # Aplica mod_centered a todos os coeficientes de uma vez (vetorizado)
        coeffs = CKKSCryptographicParameters.mod_centered(coeffs, q_coeff)
        return Polynomial(coeffs.astype(np.int64))

    @staticmethod
    def poly_ring_mod(p_numpy, ring_poly_mod, q_coeff):
        """
        Aplica redução modular no anel polinomial R_q = ℤ_q[X]/(X^N + 1).

        Este método realiza duas reduções:
        1. Redução polinomial: p_numpy mod (X^N + 1) - reduz grau para < N
        2. Redução dos coeficientes: coeficientes mod q - reduz valores para ℤ_q

        Args:
            p_numpy: Polinômio numpy
            ring_poly_mod: Polinômio de módulo do anel (tipicamente X^N + 1)
            q_coeff: Coeficiente modular

        Returns:
            Polynomial: Polinômio reduzido no anel R_q
        """
        # Primeiro: redução polinomial mod (X^N + 1) usando divisão polinomial
        # Isso garante que o resultado tenha grau < N
        _, remainder = divmod(p_numpy, ring_poly_mod)

        # Segundo: redução dos coeficientes mod q
        # Aplica mod centrado: coeficientes em ℤ_q = (-q/2, q/2]
        return CKKSCryptographicParameters.poly_coeffs_mod_q(remainder, q_coeff)

    @staticmethod
    def poly_mul_mod(p1, p2, q, ring_poly_mod):
        """
        Multiplicação de polinômios com redução modular no anel R_q.

        Realiza a multiplicação p1 * p2 e aplica redução modular no anel
        R_q = ℤ_q[X]/(X^N + 1), garantindo que o resultado tenha grau < N
        e coeficientes em ℤ_q = (-q/2, q/2].

        Args:
            p1: Primeiro polinômio (Polynomial)
            p2: Segundo polinômio (Polynomial)
            q: Módulo para os coeficientes
            ring_poly_mod: Polinômio de módulo do anel (tipicamente X^N + 1)

        Returns:
            Polynomial: Resultado da multiplicação modular em R_q
        """
        # Usa multiplicação nativa de polinômios do NumPy (muito mais eficiente)
        full_poly = p1 * p2

        # Aplica redução no anel R_q = ℤ_q[X]/(X^N + 1)
        return CKKSCryptographicParameters.poly_ring_mod(full_poly, ring_poly_mod, q)

    def poly_mul(self, p1, p2):
        """
        Multiplicação de polinômios.

        Args:
            p1: Primeiro polinômio
            p2: Segundo polinômio

        Returns:
            Polynomial: Resultado da multiplicação
        """

        full_poly = p1 * p2
        return full_poly

    def generate_gaussian_poly(self, degree_n=None, sigma_val=None):
        """
        Gera um polinômio com coeficientes gaussianos (DG(σ²)).

        Args:
            degree_n: Grau do polinômio (usa POLYNOMIAL_DEGREE se None)
            sigma_val: Desvio padrão (usa GAUSSIAN_NOISE_STDDEV se None)

        Returns:
            Polynomial: Polinômio com coeficientes gaussianos
        """
        if degree_n is None:
            degree_n = self.POLYNOMIAL_DEGREE
        if sigma_val is None:
            sigma_val = self.GAUSSIAN_NOISE_STDDEV

        coeffs = np.round(np.random.normal(0, sigma_val, size=degree_n)).astype(
            np.int64
        )
        return Polynomial(coeffs)

    def generate_hamming_weight(self, n=None, hamming_weight=None):
        """
        Gera um vetor da distribuição HWT(h) - Hamming Weight h.

        For a positive integer h, HWT(h) is the set of signed binary vectors
        in {0, ±1}^N whose Hamming weight is exactly h.

        Distribui exatamente h coeficientes não-zero em {-1, +1} aleatoriamente
        nas N posições, com os demais coeficientes sendo 0.

        Args:
            n: Dimensão do vetor (usa POLYNOMIAL_DEGREE se None)
            hamming_weight: Peso de Hamming h desejado (usa HAMMING_WEIGHT se None)

        Returns:
            ndarray: Vetor binário assinado em {0, ±1}^N com peso de Hamming h
        """
        if n is None:
            n = self.POLYNOMIAL_DEGREE
        if hamming_weight is None:
            hamming_weight = min(
                self.HAMMING_WEIGHT, n // 4
            )  # Default: configurado ou N/4

        if hamming_weight > n:
            raise ValueError(
                f"Peso de Hamming {hamming_weight} não pode ser maior que o grau {n}"
            )

        # Inicializa todos os coeficientes como zero
        coeffs = np.zeros(n, dtype=np.int64)

        # Escolhe posições aleatórias para colocar valores não-zero
        positions = np.random.choice(n, size=hamming_weight, replace=False)

        # Para cada posição, escolhe aleatoriamente +1 ou -1
        for pos in positions:
            coeffs[pos] = np.random.choice([-1, 1])

        return coeffs

    def generate_zero_one_poly(self, degree_n=None, density=None):
        """
        Gera um polinômio seguindo distribuição ZO(ρ).

        Cada coeficiente é:
        - +1 com probabilidade ρ/2
        - -1 com probabilidade ρ/2
        - 0 com probabilidade 1-ρ

        Args:
            degree_n: Grau do polinômio (usa POLYNOMIAL_DEGREE se None)
            density: Densidade ρ (usa 0.5 se None)

        Returns:
            Polynomial: Polinômio seguindo distribuição ZO(ρ)
        """
        if degree_n is None:
            degree_n = self.POLYNOMIAL_DEGREE
        if density is None:
            density = self.ZERO_ONE_DENSITY

        if not 0 <= density <= 1:
            raise ValueError(f"Densidade deve estar entre 0 e 1, recebido: {density}")

        coeffs = np.zeros(degree_n, dtype=np.int64)

        for i in range(degree_n):
            rand_val = np.random.random()
            if rand_val < density / 2:
                coeffs[i] = -1
            elif rand_val < density:
                coeffs[i] = 1
            # else: coeff[i] permanece 0

        return Polynomial(coeffs)

    def generate_uniform_random_poly(self, degree_n=None, q_bound=None):
        """
        Gera um polinômio com coeficientes uniformemente aleatórios.

        Args:
            degree_n: Grau do polinômio (usa POLYNOMIAL_DEGREE se None)
            q_bound: Limite superior para os coeficientes (usa módulo inicial se None)

        Returns:
            Polynomial: Polinômio com coeficientes uniformemente aleatórios
        """
        if degree_n is None:
            degree_n = self.POLYNOMIAL_DEGREE
        if q_bound is None:
            q_bound = self.get_initial_modulus()

        coeffs = np.random.randint(0, q_bound, size=degree_n, dtype=np.int64)
        return Polynomial(coeffs)


# Validação automática dos parâmetros
if __name__ == "__main__":
    try:
        # Cria instância com parâmetros padrão para teste
        params = CKKSCryptographicParameters()
        params.print_parameters_summary()
        print("✓ Todos os parâmetros são válidos!")
    except ValueError as e:
        print(f"✗ Erro na validação dos parâmetros: {e}")
