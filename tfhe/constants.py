"""
Constantes centralizadas para o esquema de criptografia homomórfica TFHE.

Esta classe organiza todos os parâmetros criptográficos de forma semântica
para facilitar manutenção e configuração do sistema TFHE.
"""

import numpy as np


class TFHECryptographicParameters:
    """
    Classe que centraliza todos os parâmetros criptográficos do esquema TFHE.

    Esta classe organiza as constantes de forma semântica, separando:
    - Parâmetros de dimensão
    - Parâmetros de ruído
    - Configurações de geração de números aleatórios
    """

    # === PARÂMETROS ESTRUTURAIS ===
    LWE_DIMENSION = 1  # k - dimensão dos samples LWE
    POLYNOMIAL_DEGREE = 4  # N - grau do polinômio

    # === PARÂMETROS DE RUÍDO ===
    NOISE_PARAMETER = 0.005  # alpha - parâmetro de ruído gaussiano

    # === CONFIGURAÇÕES INTERNAS ===
    @classmethod
    def create_secret_key(cls):
        """
        Cria uma chave secreta aleatória.

        Returns:
            np.ndarray: Chave secreta binária de tamanho N
        """
        return np.random.randint(2, size=(cls.POLYNOMIAL_DEGREE))

    @classmethod
    def get_random_generator(cls):
        """
        Retorna um gerador de números aleatórios configurado.

        Returns:
            np.random.Generator: Gerador de números aleatórios
        """
        return np.random.default_rng()

    @classmethod
    def get_polynomial_modulus(cls):
        """
        Retorna o polinômio de módulo X^N + 1.

        Returns:
            np.poly1d: Polinômio X^N + 1
        """
        return np.poly1d([1] + [0] * (cls.POLYNOMIAL_DEGREE - 1) + [1])

    # === MÉTODOS DE VALIDAÇÃO ===
    @classmethod
    def validate_parameters(cls):
        """
        Valida a consistência dos parâmetros criptográficos.

        Raises:
            ValueError: Se algum parâmetro estiver inconsistente
        """
        if cls.LWE_DIMENSION <= 0:
            raise ValueError("LWE_DIMENSION deve ser positivo")

        if cls.POLYNOMIAL_DEGREE <= 0:
            raise ValueError("POLYNOMIAL_DEGREE deve ser positivo")

        if cls.NOISE_PARAMETER <= 0:
            raise ValueError("NOISE_PARAMETER deve ser positivo")

    @classmethod
    def get_security_level_estimate(cls):
        """
        Retorna uma estimativa do nível de segurança baseado nos parâmetros.

        Returns:
            int: Estimativa do nível de segurança em bits
        """
        # Estimativa simplificada para TFHE
        # Para uma análise real, seria necessário considerar outros fatores
        return int(cls.POLYNOMIAL_DEGREE * cls.LWE_DIMENSION * 10)

    @classmethod
    def print_parameters_summary(cls):
        """
        Imprime um resumo dos parâmetros configurados.
        """
        print("=== PARÂMETROS CRIPTOGRÁFICOS TFHE ===")
        print(f"Dimensão LWE (k): {cls.LWE_DIMENSION}")
        print(f"Grau do polinômio (N): {cls.POLYNOMIAL_DEGREE}")
        print(f"Parâmetro de ruído (alpha): {cls.NOISE_PARAMETER}")
        print(f"Nível de segurança estimado: ~{cls.get_security_level_estimate()} bits")
        print("=" * 40)


# Validação automática dos parâmetros
if __name__ == "__main__":
    try:
        TFHECryptographicParameters.validate_parameters()
        TFHECryptographicParameters.print_parameters_summary()
        print("✓ Todos os parâmetros são válidos!")
    except ValueError as e:
        print(f"✗ Erro na validação dos parâmetros: {e}")
