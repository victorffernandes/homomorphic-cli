"""
Constantes centralizadas para o esquema de criptografia homomórfica CKKS.

Esta classe organiza todos os parâmetros criptográficos de forma semântica
para facilitar manutenção e configuração do sistema.

Baseado nas recomendações do HEAAN (run.cpp):
- Básico: logN=13, logQ=65, logp=30
- Potências: logN=13, logQ=155, logp=30
- Alta precisão: logN=15, logQ=618, logp=56
- FFT: logN=13, logQ=100, logp=42
- Inverso: logN=14, logQ=255, logp=25

Seguindo implementação HEAAN (Context.cpp):
- N = 2^logN (polynomial degree)
- Q = 2^logQ (base modulus)
- QQ = 2^(2*logQ) = Q^2 (special modulus for key switching)
- Δ = 2^logp (scaling factor)
- σ = 3.2 (default gaussian noise stddev)
- h = 64 (default hamming weight)
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
        print(f"logQQ: {self.logQQ} → QQ = 2^{self.logQQ} = {self.P} (P para key switching)")
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

        Implementação baseada em HEEAN usando redução eficiente:
        - Explora X^N ≡ -1 mod (X^N + 1)
        - Para polinômio p: res[i] = (pp[i] - pp[i+N]) mod q

        Este método realiza duas reduções:
        1. Redução polinomial: p_numpy mod (X^N + 1) - reduz grau para < N
        2. Redução dos coeficientes: coeficientes mod q - reduz valores para ℤ_q

        Args:
            p_numpy: Polinômio numpy (grau pode ser até 2N-2)
            ring_poly_mod: Polinômio de módulo do anel (tipicamente X^N + 1)
            q_coeff: Coeficiente modular

        Returns:
            Polynomial: Polinômio reduzido no anel R_q
        """
        # Extrai o grau N do polinômio de módulo (X^N + 1)
        # ring_poly_mod deve ser X^N + 1, então o grau é N
        degree = len(ring_poly_mod.coef) - 1

        # Extrai coeficientes do polinômio de entrada
        coeffs = (
            np.array(p_numpy.coef, dtype=np.int64)
            if hasattr(p_numpy, "coef")
            else np.array(p_numpy, dtype=np.int64)
        )

        # Garante que temos pelo menos 2*degree coeficientes (preenche com zeros se necessário)
        # Isso corresponde a pp.SetLength(2 * degree) no HEEAN
        if len(coeffs) < 2 * degree:
            coeffs = np.pad(coeffs, (0, 2 * degree - len(coeffs)), mode="constant")
        elif len(coeffs) > 2 * degree:
            # Trunca para 2*degree (mantém apenas os primeiros 2*N coeficientes)
            coeffs = coeffs[: 2 * degree]

        # Aplica redução mod X^N + 1 usando padrão HEEAN:
        # Para cada i em [0, degree), res[i] = (pp[i] - pp[i+degree]) mod q
        # Isso funciona porque X^N ≡ -1, então termos em posição i+N contribuem como -X^i
        #
        # Implementação vetorizada seguindo padrão HEEAN:
        # 1. Reduz todos os coeficientes mod q (rem no HEEAN)
        # 2. Calcula diferença e reduz mod q (SubMod no HEEAN)
        pp_low = coeffs[:degree] % q_coeff
        pp_high = coeffs[degree : 2 * degree] % q_coeff

        # SubMod vetorizado: res[i] = (pp[i] - pp[i+degree]) mod q
        result_coeffs = (pp_low - pp_high) % q_coeff

        # Cria polinômio resultante e aplica mod centrado para CKKS
        # (HEEAN usa mod padrão, mas CKKS precisa de representação centrada)
        result_poly = Polynomial(result_coeffs.astype(np.int64))
        return CKKSCryptographicParameters.poly_coeffs_mod_q(result_poly, q_coeff)

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
