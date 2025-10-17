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
        polynomial_degree: int = 64,  # N - grau do polinômio ciclotômico
        q0_base: int = 131071,  # Q0 - módulo base
        scaling_factor: int = 4096,  # DELTA - fator de escala
        gaussian_noise_stddev: float = 3.2,  # σ - desvio padrão gaussiano
        hamming_weight: int = None,  # h - peso de Hamming (padrão: N/4)
        zero_one_density: float = 0.5,  # ρ - densidade ZO
        modulus_levels: int = 5,  # Número de níveis na cadeia de módulos
    ):
        """
        Inicializa os parâmetros criptográficos CKKS.

        Conforme definição KeyGen(1λ):
        - Escolhe M = M(λ, qL), h = h(λ, qL), P = P(λ, qL), σ = σ(λ, qL)

        Args:
            polynomial_degree: N - deve ser potência de 2
            p_parameter: P - parâmetro para evaluation key
            q0_base: Q0 - módulo base para construir cadeia
            scaling_factor: DELTA - fator de escala para codificação
            gaussian_noise_stddev: σ - desvio padrão para DG(σ²)
            hamming_weight: h - peso de Hamming para HWT(h) (padrão: N/4)
            zero_one_density: ρ - densidade para ZO(ρ)
            modulus_levels: Número de níveis na cadeia de módulos
        """
        # === PARÂMETROS ESTRUTURAIS ===
        self.POLYNOMIAL_DEGREE = polynomial_degree  # N

        # Define P conforme especificação (deve ser escolhido baseado em λ, qL)
        self.p_small = 2

        self.Q0 = q0_base  # Q0

        # === MÓDULOS DA CADEIA CKKS ===
        self.MODULUS_CHAIN = [
            self.p_small**i * self.Q0 for i in range(1, modulus_levels + 1)
        ]

        self.P = max(1000, min(self.MODULUS_CHAIN[-1] // 100, 10000))

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

        if self.GAUSSIAN_NOISE_STDDEV <= 0:
            raise ValueError("GAUSSIAN_NOISE_STDDEV deve ser positivo")

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
    def poly_coeffs_mod_q(self, p_numpy, q_coeff):
        """
        Aplica operação modular aos coeficientes de um polinômio.

        Args:
            p_numpy: Polinômio numpy
            q_coeff: Coeficiente modular

        Returns:
            Polynomial: Polinômio com coeficientes modulares
        """
        coeffs = p_numpy.coef.copy()
        for i in range(len(coeffs)):
            coeffs[i] = int(coeffs[i]) % q_coeff
        return Polynomial(coeffs)

    def poly_ring_mod(self, p_numpy, ring_poly_mod, q_coeff):
        """
        Aplica redução modular no anel polinomial.

        Args:
            p_numpy: Polinômio numpy
            ring_poly_mod: Polinômio de módulo do anel
            q_coeff: Coeficiente modular

        Returns:
            Polynomial: Polinômio reduzido no anel
        """
        remainder_poly = p_numpy % ring_poly_mod
        return self.poly_coeffs_mod_q(remainder_poly, q_coeff)

    def poly_mul_mod(self, p1, p2, q, ring_poly_mod):
        """
        Multiplicação de polinômios com redução modular.

        Args:
            p1: Primeiro polinômio
            p2: Segundo polinômio
            q: Módulo para os coeficientes
            ring_poly_mod: Polinômio de módulo do anel

        Returns:
            Polynomial: Resultado da multiplicação modular
        """
        coeffs1 = [int(c) for c in p1.coef]
        coeffs2 = [int(c) for c in p2.coef]
        prod_coeffs = [0] * (len(coeffs1) + len(coeffs2) - 1)
        for i in range(len(coeffs1)):
            for j in range(len(coeffs2)):
                prod_coeffs[i + j] += coeffs1[i] * coeffs2[j]
        full_poly = Polynomial(prod_coeffs)
        return self.poly_ring_mod(full_poly, ring_poly_mod, q)

    def poly_mul(self, p1, p2):
        """
        Multiplicação de polinômios.

        Args:
            p1: Primeiro polinômio
            p2: Segundo polinômio

        Returns:
            Polynomial: Resultado da multiplicação
        """
        coeffs1 = [int(c) for c in p1.coef]
        coeffs2 = [int(c) for c in p2.coef]
        prod_coeffs = [0] * (len(coeffs1) + len(coeffs2) - 1)
        for i in range(len(coeffs1)):
            for j in range(len(coeffs2)):
                prod_coeffs[i + j] += coeffs1[i] * coeffs2[j]
        full_poly = Polynomial(prod_coeffs)
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

    def generate_hamming_weight_poly(self, degree_n=None, hamming_weight=None):
        """
        Gera um polinômio com peso de Hamming específico (HWT(h)).

        Distribui hamming_weight coeficientes não-zero {-1, +1} aleatoriamente
        nos degree_n coeficientes, com os demais sendo 0.

        Args:
            degree_n: Grau do polinômio (usa POLYNOMIAL_DEGREE se None)
            hamming_weight: Peso de Hamming desejado (usa 64 se None)

        Returns:
            Polynomial: Polinômio binário com peso de Hamming específico
        """
        if degree_n is None:
            degree_n = self.POLYNOMIAL_DEGREE
        if hamming_weight is None:
            hamming_weight = min(
                self.HAMMING_WEIGHT, degree_n // 4
            )  # Default: configurado ou N/4

        if hamming_weight > degree_n:
            raise ValueError(
                f"Peso de Hamming {hamming_weight} não pode ser maior que o grau {degree_n}"
            )

        # Inicializa todos os coeficientes como zero
        coeffs = np.zeros(degree_n, dtype=np.int64)

        # Escolhe posições aleatórias para colocar valores não-zero
        positions = np.random.choice(degree_n, size=hamming_weight, replace=False)

        # Para cada posição, escolhe aleatoriamente +1 ou -1
        for pos in positions:
            coeffs[pos] = np.random.choice([-1, 1])

        return Polynomial(coeffs)

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
