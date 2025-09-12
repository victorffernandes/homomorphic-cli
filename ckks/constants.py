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
    """

    # === PARÂMETROS ESTRUTURAIS ===
    POLYNOMIAL_DEGREE = 2048  # Grau do polinômio ciclotômico (N)

    # === MÓDULOS DA CADEIA CKKS ===
    # Voltando aos módulos originais que funcionavam estruturalmente
    MODULUS_CHAIN = [1099511922689, 1099512004609, 1099512037377]  # Q_CHAIN

    # === PARÂMETROS DE ESCALA ===
    SCALING_FACTOR = 1099511922688  # DELTA - fator de escala para codificação

    # === PARÂMETROS DE RUÍDO ===
    GAUSSIAN_NOISE_STDDEV = 3.2  # SIGMA - desvio padrão do ruído gaussiano

    # === ESTRUTURAS ALGÉBRICAS ===
    @classmethod
    def get_polynomial_modulus_ring(cls):
        """
        Retorna o anel de polinômios modulares X^N + 1.

        Returns:
            Polynomial: O polinômio ciclotômico X^N + 1
        """
        return Polynomial([1] + [0] * (cls.POLYNOMIAL_DEGREE - 1) + [1])

    # === MÉTODOS DE ACESSO ===
    @classmethod
    def get_initial_modulus(cls):
        """
        Retorna o módulo inicial (último da cadeia).

        Returns:
            int: O maior módulo da cadeia
        """
        return cls.MODULUS_CHAIN[-1]

    @classmethod
    def get_maximum_plaintext_slots(cls):
        """
        Retorna o número máximo de slots disponíveis para texto claro.

        Returns:
            int: Número de slots (N/2)
        """
        return cls.POLYNOMIAL_DEGREE // 2

    @classmethod
    def get_scaling_factor_squared(cls):
        """
        Retorna o quadrado do fator de escala, usado após multiplicação.

        Returns:
            int: DELTA²
        """
        return cls.SCALING_FACTOR**2

    @classmethod
    def validate_parameters(cls):
        """
        Valida a consistência dos parâmetros criptográficos.

        Raises:
            ValueError: Se algum parâmetro estiver inconsistente
        """
        if (
            cls.POLYNOMIAL_DEGREE <= 0
            or (cls.POLYNOMIAL_DEGREE & (cls.POLYNOMIAL_DEGREE - 1)) != 0
        ):
            raise ValueError("POLYNOMIAL_DEGREE deve ser uma potência de 2 positiva")

        if not cls.MODULUS_CHAIN or len(cls.MODULUS_CHAIN) == 0:
            raise ValueError("MODULUS_CHAIN não pode estar vazia")

        if cls.SCALING_FACTOR <= 0:
            raise ValueError("SCALING_FACTOR deve ser positivo")

        if cls.GAUSSIAN_NOISE_STDDEV <= 0:
            raise ValueError("GAUSSIAN_NOISE_STDDEV deve ser positivo")

        # Verifica se os módulos estão em ordem crescente
        for i in range(1, len(cls.MODULUS_CHAIN)):
            if cls.MODULUS_CHAIN[i] <= cls.MODULUS_CHAIN[i - 1]:
                raise ValueError("MODULUS_CHAIN deve estar em ordem crescente")

    @classmethod
    def get_security_level_estimate(cls):
        """
        Retorna uma estimativa do nível de segurança baseado nos parâmetros.

        Returns:
            int: Estimativa do nível de segurança em bits
        """
        # Estimativa simplificada baseada no grau do polinômio
        # Para uma análise real, seria necessário considerar outros fatores
        return int(np.log2(cls.POLYNOMIAL_DEGREE) * 10)

    @classmethod
    def print_parameters_summary(cls):
        """
        Imprime um resumo dos parâmetros configurados.
        """
        print("=== PARÂMETROS CRIPTOGRÁFICOS CKKS ===")
        print(f"Grau do polinômio (N): {cls.POLYNOMIAL_DEGREE}")
        print(
            f"Fator de escala (DELTA): {cls.SCALING_FACTOR} (~2^{int(np.log2(cls.SCALING_FACTOR))})"
        )
        print(f"Desvio padrão do ruído (SIGMA): {cls.GAUSSIAN_NOISE_STDDEV}")
        print(f"Cadeia de módulos: {len(cls.MODULUS_CHAIN)} níveis")
        print(
            f"  - Maior módulo: {cls.MODULUS_CHAIN[-1]} (~{cls.MODULUS_CHAIN[-1].bit_length()} bits)"
        )
        print(
            f"  - Menor módulo: {cls.MODULUS_CHAIN[0]} (~{cls.MODULUS_CHAIN[0].bit_length()} bits)"
        )
        print(f"Slots disponíveis: {cls.get_maximum_plaintext_slots()}")
        print(f"Nível de segurança estimado: ~{cls.get_security_level_estimate()} bits")
        print("=" * 40)

    # === FUNÇÕES AUXILIARES PARA OPERAÇÕES POLINOMIAIS ===
    @classmethod
    def poly_coeffs_mod_q(cls, p_numpy, q_coeff):
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

    @classmethod
    def poly_ring_mod(cls, p_numpy, ring_poly_mod, q_coeff):
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
        return cls.poly_coeffs_mod_q(remainder_poly, q_coeff)

    @classmethod
    def poly_mul_mod(cls, p1, p2, q, ring_poly_mod):
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
        return cls.poly_ring_mod(full_poly, ring_poly_mod, q)

    @classmethod
    def poly_mul(cls, p1, p2):
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

    @classmethod
    def generate_gaussian_poly(cls, degree_n=None, sigma_val=None):
        """
        Gera um polinômio com coeficientes gaussianos.

        Args:
            degree_n: Grau do polinômio (usa POLYNOMIAL_DEGREE se None)
            sigma_val: Desvio padrão (usa GAUSSIAN_NOISE_STDDEV se None)

        Returns:
            Polynomial: Polinômio com coeficientes gaussianos
        """
        if degree_n is None:
            degree_n = cls.POLYNOMIAL_DEGREE
        if sigma_val is None:
            sigma_val = cls.GAUSSIAN_NOISE_STDDEV

        coeffs = np.round(np.random.normal(0, sigma_val, size=degree_n)).astype(
            np.int64
        )
        return Polynomial(coeffs)

    @classmethod
    def generate_uniform_random_poly(cls, degree_n=None, q_bound=None):
        """
        Gera um polinômio com coeficientes uniformemente aleatórios.

        Args:
            degree_n: Grau do polinômio (usa POLYNOMIAL_DEGREE se None)
            q_bound: Limite superior para os coeficientes (usa módulo inicial se None)

        Returns:
            Polynomial: Polinômio com coeficientes uniformemente aleatórios
        """
        if degree_n is None:
            degree_n = cls.POLYNOMIAL_DEGREE
        if q_bound is None:
            q_bound = cls.get_initial_modulus()

        coeffs = np.random.randint(0, q_bound, size=degree_n, dtype=np.int64)
        return Polynomial(coeffs)


# Validação automática dos parâmetros
if __name__ == "__main__":
    try:
        CKKSCryptographicParameters.validate_parameters()
        CKKSCryptographicParameters.print_parameters_summary()
        print("✓ Todos os parâmetros são válidos!")
    except ValueError as e:
        print(f"✗ Erro na validação dos parâmetros: {e}")
