import numpy as np
from numpy.polynomial import Polynomial
from .bigint_poly_utils import (
    mod_centered as bi_mod_centered,
    poly_coeffs_mod_q as bi_poly_coeffs_mod_q,
    poly_ring_mod as bi_poly_ring_mod,
    poly_mul_mod as bi_poly_mul_mod,
    generate_uniform_random_poly as bi_generate_uniform_random_poly,
)
from .int_backend import make_int_backend_array


class CKKSCryptographicParameters:
    """
    Classe que centraliza todos os parâmetros criptográficos do esquema CKKS
    em estilo HEAAN.

    Organização:
    - Parâmetros de segurança e tamanho (logN, logQ)
    - Parâmetros de precisão (logp, cadeia de módulos)
    - Estruturas algébricas (R = Z[X]/(X^N + 1))
    - Configurações de ruído (σ, h, ZO)

    Perfis suportados (espelhando run.cpp do HEAAN):
    - Default / básico:     logN=13, logQ=65,  logp=30  (testes básicos)
    - basic_config():       logN=13, logQ=65,  logp=30
    - power_config():       logN=13, logQ=155, logp=30  (potências profundas)
    - high_precision_config(): logN=15, logQ=618, logp=56 (alta precisão)
    - fft_config():         logN=13, logQ=100, logp=42  (FFT)
    - inverse_config():     logN=14, logQ=255, logp=25  (funções inversas)

    Observação: embora o código aceite parâmetros arbitrários (logN, logQ, logp),
    ele é otimizado para a faixa de demonstração do HEAAN; configurações muito
    extremas (por exemplo, logQ≈600) podem exigir mais ajustes na precisão
    numérica e nos testes para manter erros dentro de tolerâncias muito apertadas.
    """

    def __init__(
        self,
        # Defaults aligned with HEAAN basic tests (run.cpp):
        # logN=13, logQ=65, logp=30
        logN: int = 13,  # log2(N) - log do grau do polinômio (N = 2^logN)
        logQ: int = 65,  # log2(Q) - log do módulo base (Q = 2^logQ)
        logp: int = 30,  # log2(Δ) - log do fator de escala (Δ = 2^logp)
        gaussian_noise_stddev: float = 3.2,  # σ - desvio padrão gaussiano (padrão HEAAN)
        hamming_weight: int = 64,  # h - peso de Hamming (padrão HEAAN)
        zero_one_density: float = 0.5,  # ρ - densidade ZO
        total_levels: int = None,  # Número de níveis na cadeia (None = calculado automaticamente)
    ):
        """
        Inicializa os parâmetros criptográficos CKKS seguindo HEAAN.

        Conforme definição KeyGen(1λ) e implementação HEAAN:
        - N = 2^logN (grau do polinômio)
        - Q = 2^logQ (módulo base)
        - QQ = 2^(2*logQ) = Q^2 (módulo especial para key switching)
        - Δ = 2^logp (fator de escala)
        - σ = 3.2 (desvio padrão gaussiano padrão)
        - h = 64 (peso de Hamming padrão)

        Valores recomendados de run.cpp:
        - Básico: logN=13, logQ=65, logp=30
        - Potências: logN=13, logQ=155, logp=30
        - Alta precisão: logN=15, logQ=618, logp=56
        - FFT: logN=13, logQ=100, logp=42

        Args:
            logN: log2(N) - log do grau do polinômio (N = 2^logN)
            logQ: log2(Q) - log do módulo base (Q = 2^logQ)
            logp: log2(Δ) - log do fator de escala (Δ = 2^logp)
            gaussian_noise_stddev: σ - desvio padrão para DG(σ²)
            hamming_weight: h - peso de Hamming para HWT(h)
            zero_one_density: ρ - densidade para ZO(ρ)
            total_levels: Número de níveis na cadeia (None = usa logQQ = 2*logQ)
        """
        # === PARÂMETROS ESTRUTURAIS (seguindo Context::init) ===
        self.logN = logN
        self.logQ = logQ
        self.logp = logp

        # N = 2^logN (polynomial degree)
        self.POLYNOMIAL_DEGREE = 1 << logN  # N

        # Nh = N/2
        self.Nh = self.POLYNOMIAL_DEGREE >> 1

        # M = 2N
        self.M = self.POLYNOMIAL_DEGREE << 1

        # Q = 2^logQ (base modulus)
        self.Q0 = 1 << logQ  # Q

        # logQQ = 2*logQ
        self.logQQ = logQ << 1

        # QQ = 2^(2*logQ) = Q^2 (special modulus for key switching)
        self.P = 1 << self.logQQ  # QQ = Q^2

        # === MÓDULOS DA CADEIA CKKS (seguindo qpowvec) ===
        # HEAAN usa qpowvec[i] = 2^i para i de 0 até logQQ+1
        # A cadeia de módulos é implícita: qualquer módulo é 2^logq onde logq varia
        if total_levels is None:
            # Por padrão, cria cadeia até logQQ (2*logQ) para key switching
            total_levels = self.logQQ

        # Cadeia de módulos: [2^0, 2^1, 2^2, ..., 2^total_levels]
        # Similar ao qpowvec do HEAAN
        self.MODULUS_CHAIN = [1 << i for i in range(total_levels + 1)]

        # === PARÂMETROS DE ESCALA ===
        # SCALING_FACTOR = 2^logp (DELTA)
        self.SCALING_FACTOR = 1 << logp  # Δ = 2^logp

        # === PARÂMETROS DE RUÍDO ===
        self.GAUSSIAN_NOISE_STDDEV = gaussian_noise_stddev  # σ

        # === PARÂMETROS PARA DISTRIBUIÇÕES DE CHAVE ===
        self.HAMMING_WEIGHT = hamming_weight  # h
        self.ZERO_ONE_DENSITY = zero_one_density  # ρ

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

    def get_modulus_at_level(self, logq: int):
        """
        Retorna o módulo no nível especificado (seguindo padrão HEAAN qpowvec).

        Args:
            logq: log do módulo desejado (q = 2^logq)

        Returns:
            int: Módulo 2^logq
        """
        if logq < 0 or logq >= len(self.MODULUS_CHAIN):
            return 1 << logq  # Calcula diretamente se fora da cadeia
        return self.MODULUS_CHAIN[logq]

    def get_q0(self):
        """
        Retorna Q0 (módulo base).

        Returns:
            int: Q0 = 2^logQ
        """
        return self.Q0

    def get_p(self):
        """
        Retorna P (módulo especial para key switching, QQ em HEAAN).

        Returns:
            int: P = QQ = 2^(2*logQ) = Q^2
        """
        return self.P

    @classmethod
    def basic_config(cls):
        """
        Configuração básica recomendada do HEAAN (run.cpp).

        Returns:
            CKKSCryptographicParameters: Parâmetros para operações básicas
        """
        return cls(logN=13, logQ=65, logp=30)

    @classmethod
    def power_config(cls):
        """
        Configuração para operações de potência recomendada do HEAAN (run.cpp).

        Returns:
            CKKSCryptographicParameters: Parâmetros para operações de potência
        """
        return cls(logN=13, logQ=155, logp=30)

    @classmethod
    def high_precision_config(cls):
        """
        Configuração de alta precisão recomendada do HEAAN (run.cpp).

        Returns:
            CKKSCryptographicParameters: Parâmetros para alta precisão
        """
        return cls(logN=15, logQ=618, logp=56)

    @classmethod
    def precision_multiply_config(cls):
        """
        Configuração para multiplicação homomórfica com precisão ~0.01.

        Projetada para operar em segurança com backends de inteiros de alta
        precisão (por exemplo, int128 ou ints Python), sem depender de limites
        específicos de int64.

        Returns:
            CKKSCryptographicParameters: Parâmetros para precisão ~0.01 em multiply
        """
        return cls(
            logN=9,  # N=512 (balanço ruído vs. tempo)
            logQ=16,
            logp=12,
            total_levels=29,  # top q=2^29; após rescale q=2^28 >> 14*2^23
            gaussian_noise_stddev=1.0,  # ruído menor para melhor precisão
        )

    @classmethod
    def fft_config(cls):
        """
        Configuração para operações FFT recomendada do HEAAN (run.cpp).

        Returns:
            CKKSCryptographicParameters: Parâmetros para operações FFT
        """
        return cls(logN=13, logQ=100, logp=42)

    @classmethod
    def inverse_config(cls):
        """
        Configuração para operações inversas recomendada do HEAAN (run.cpp).

        Returns:
            CKKSCryptographicParameters: Parâmetros para operações inversas
        """
        return cls(logN=14, logQ=255, logp=25)

    def print_parameters_summary(self):
        """
        Imprime um resumo dos parâmetros configurados.
        """
        print("=== PARÂMETROS CRIPTOGRÁFICOS CKKS (HEAAN) ===")
        print(f"logN: {self.logN} → N = 2^{self.logN} = {self.POLYNOMIAL_DEGREE}")
        print(f"logQ: {self.logQ} → Q = 2^{self.logQ} = {self.Q0}")
        print(f"logp: {self.logp} → Δ = 2^{self.logp} = {self.SCALING_FACTOR}")
        print(
            f"logQQ: {self.logQQ} → QQ = 2^{self.logQQ} = {self.P} (P para key switching)"
        )
        print(f"Desvio padrão do ruído (σ): {self.GAUSSIAN_NOISE_STDDEV}")
        print(f"Cadeia de módulos: {len(self.MODULUS_CHAIN)} níveis")
        print(
            f"  - Maior módulo: {self.MODULUS_CHAIN[-1]} "
            f"(~{self.MODULUS_CHAIN[-1].bit_length()} bits, 2^{int(np.log2(self.MODULUS_CHAIN[-1]))})"
        )
        print(
            f"  - Menor módulo: {self.MODULUS_CHAIN[0]} "
            f"(~{self.MODULUS_CHAIN[0].bit_length()} bits, 2^{int(np.log2(self.MODULUS_CHAIN[0]))})"
        )
        print(f"  - Módulo base (Q0): {self.Q0} (~{self.Q0.bit_length()} bits)")
        print(f"  - Módulo especial (P/QQ): {self.P} (~{self.P.bit_length()} bits)")
        print(f"Slots disponíveis: {self.POLYNOMIAL_DEGREE // 2} (Nh = N/2)")
        print(f"Peso de Hamming (h): {self.HAMMING_WEIGHT}")
        print(f"Densidade ZO (ρ): {self.ZERO_ONE_DENSITY}")
        print("=" * 50)

    # === FUNÇÕES AUXILIARES PARA OPERAÇÕES POLINOMIAIS ===
    @staticmethod
    def mod_centered(value, modulus):
        # Delegate to bigint-aware helper
        return bi_mod_centered(value, modulus)

    @staticmethod
    def poly_coeffs_mod_q(p_numpy, q_coeff):
        """Wrapper para a versão bigint em ckks.bigint_poly_utils."""
        return bi_poly_coeffs_mod_q(p_numpy, q_coeff)

    @staticmethod
    def poly_ring_mod(p_numpy, ring_poly_mod, q_coeff):
        """Wrapper para a versão bigint em ckks.bigint_poly_utils."""
        return bi_poly_ring_mod(p_numpy, ring_poly_mod, q_coeff)

    @staticmethod
    def poly_mul_mod(p1, p2, q, ring_poly_mod):
        """Wrapper para a versão bigint em ckks.bigint_poly_utils."""
        return bi_poly_mul_mod(p1, p2, q, ring_poly_mod)

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

        coeffs = np.round(np.random.normal(0, sigma_val, size=degree_n))
        return Polynomial(make_int_backend_array(coeffs))

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

        # Inicializa todos os coeficientes como zero (backend integer representation)
        coeffs = make_int_backend_array(np.zeros(n, dtype=int))

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

        # Vectorized implementation
        rand_vals = np.random.random(degree_n)
        coeffs = np.zeros(degree_n, dtype=int)
        coeffs[rand_vals < density / 2] = -1
        coeffs[(rand_vals >= density / 2) & (rand_vals < density)] = 1

        return Polynomial(make_int_backend_array(coeffs))

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

        return bi_generate_uniform_random_poly(degree_n, q_bound)


# Validação automática dos parâmetros
if __name__ == "__main__":
    try:
        print("=== CONFIGURAÇÃO PADRÃO (Básica) ===")
        params = CKKSCryptographicParameters()
        params.print_parameters_summary()
        print("\n=== CONFIGURAÇÃO PARA POTÊNCIAS ===")
        params_power = CKKSCryptographicParameters.power_config()
        params_power.print_parameters_summary()
        print("\n=== CONFIGURAÇÃO ALTA PRECISÃO ===")
        params_hp = CKKSCryptographicParameters.high_precision_config()
        params_hp.print_parameters_summary()
        print("\n✓ Todas as configurações são válidas!")
    except ValueError as e:
        print(f"✗ Erro na validação dos parâmetros: {e}")
