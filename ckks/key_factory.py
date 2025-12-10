"""
Fábrica para geração e gerenciamento de chaves CKKS seguindo a definição KeyGen formal.

Esta classe implementa a geração de chaves conforme especificado no esquema CKKS.
"""

import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Union, Tuple, Dict, Any

# Imports relativos para uso como módulo, absolutos para execução direta
try:
    from .constants import CKKSCryptographicParameters
except ImportError:
    # Fallback para execução direta
    from constants import CKKSCryptographicParameters


class CKKSKeyFactory:
    """
    Fábrica para geração e gerenciamento de chaves CKKS seguindo a definição KeyGen formal.

    Esta classe implementa a geração de chaves conforme especificado:

    KeyGen(1λ):
    - Escolhe parâmetros M, h, P, σ baseados no parâmetro de segurança λ
    - Sample s ← HWT(h): chave secreta com peso de Hamming h
    - Sample a ← RqL, e ← DG(σ²): elementos para chave pública
    - Set sk ← (1, s), pk ← (b, a) onde b ← −as + e (mod qL)
    - Sample a' ← RP·qL, e' ← DG(σ²): elementos para evaluation key
    - Set evk ← (b', a') onde b' ← −a's + e' + P s² (mod P · qL)

    Distribuições utilizadas:
    - DG(σ²): Gaussiana Discreta com variância σ²
    - HWT(h): Vetores binários {0, ±1}^N com peso de Hamming h
    - ZO(ρ): Distribuição zero-one com densidade ρ
    """

    def __init__(self, crypto_params: CKKSCryptographicParameters = None):
        """
        Inicializa a fábrica de chaves com parâmetros criptográficos.

        Args:
            crypto_params: Parâmetros criptográficos CKKS (usa padrão se None)
        """
        if crypto_params is None:
            crypto_params = CKKSCryptographicParameters()

        self.crypto_params = crypto_params

    def generate_secret_key(
        self, hamming_weight: int = None
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma chave secreta para o esquema CKKS seguindo a definição KeyGen.

        Conforme a definição KeyGen:
        - For a positive integer h, HWT(h) is the set of signed binary vectors
          in {0, ±1}^N whose Hamming weight is exactly h
        - Sample s ← HWT(h)
        - Set the secret key as sk ← (1, s)

        A chave secreta é um vetor de dois polinômios: sk = (1, s), onde:
        - O primeiro componente é o polinômio constante 1
        - O segundo componente s é amostrado de HWT(h) com exatamente h
          coeficientes não-zero em {-1, +1}, com os demais sendo 0

        Args:
            hamming_weight: Peso de Hamming h (usa padrão se None)

        Returns:
            Tuple[Polynomial, Polynomial]: Chave secreta sk = (1, s) onde
                - Primeiro elemento: Polynomial([1]) - polinômio constante 1
                - Segundo elemento: s ∈ HWT(h) ⊂ {0, ±1}^N
        """
        if hamming_weight is None:
            hamming_weight = self.crypto_params.HAMMING_WEIGHT

        # Sample s ← HWT(h): vetor binário assinado com peso de Hamming h
        s_coeffs = self.crypto_params.generate_hamming_weight(
            self.crypto_params.POLYNOMIAL_DEGREE, hamming_weight
        )
        s = Polynomial(s_coeffs)

        # Set the secret key as sk ← (1, s)
        one = Polynomial([1])  # Polinômio constante 1

        return (one, s)

    def generate_public_key(
        self, secret_key: Tuple[Polynomial, Polynomial], level: int = None
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma chave pública a partir da chave secreta seguindo a definição KeyGen.

        Conforme a definição:
        - Sample a ← RqL and e ← DG(σ²)
        - Set the public key as pk ← (b, a) ∈ R²qL where b ← −as + e (mod qL)

        Args:
            secret_key: Chave secreta sk = (1, s) onde s é usado para gerar pk
            level: Nível na cadeia de módulos (usa o maior se None)

        Returns:
            Tuple[Polynomial, Polynomial]: Tupla (b, a) da chave pública
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1

        # Extrai o componente s de sk = (1, s)
        _, s = secret_key

        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        q_mod = self.crypto_params.MODULUS_CHAIN[level]  # qL
        sigma = self.crypto_params.GAUSSIAN_NOISE_STDDEV

        # Sample a ← RqL (componente aleatório uniforme)
        a = self.crypto_params.generate_uniform_random_poly(n_degree, q_mod)

        # Sample e ← DG(σ²) (erro gaussiano)
        e = self.crypto_params.generate_gaussian_poly(n_degree, sigma)

        # Calcular b seguindo padrão HEEAN: b = e - a*s (mod qL)
        # HEEAN: mult(bx, sx, ax, QQ, N) -> bx = sx * ax
        #        sub(bx, ex, bx, QQ, N) -> bx = ex - bx = ex - sx*ax
        # Primeiro calcula s * a (seguindo ordem HEEAN: sx * ax)
        s_a = self.crypto_params.poly_mul_mod(s, a, q_mod, ring_poly_mod)

        # Depois calcula e - s*a (seguindo HEEAN: ex - bx)
        b = e - s_a

        # Aplica redução modular final
        b_final = self.crypto_params.poly_ring_mod(b, ring_poly_mod, q_mod)

        # Retorna pk ← (b, a)
        return (b_final, a)

    def generate_evaluation_key(
        self,
        secret_key: Tuple[Polynomial, Polynomial],
        level: int = None,
        P: int = None,
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma Evaluation Key (EVK) seguindo a definição KeyGen.

        Conforme a definição:
        - Sample a' ← RP·qL and e' ← DG(σ²)
        - Set the evaluation key as evk ← (b', a') ∈ R²P·qL
          where b' ← −a's + e' + P s² (mod P · qL)

        Args:
            secret_key: Chave secreta sk = (1, s) onde s é usado para gerar evk
            level: Nível na cadeia de módulos (usa o maior se None)
            P: Parâmetro P conforme definição (usa valor padrão se None)

        Returns:
            Tuple[Polynomial, Polynomial]: Tupla (b', a') da evaluation key
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1

        # Extrai o componente s de sk = (1, s)
        _, s = secret_key

        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        qL = self.crypto_params.MODULUS_CHAIN[level]  # qL

        # Usar P fornecido ou padrão
        if P is None:
            P = self.crypto_params.P

        sigma = self.crypto_params.GAUSSIAN_NOISE_STDDEV

        P_qL = P * qL  # P · qL (equivalente a QQ no HEEAN)

        # Sample a' ← RP·qL (componente aleatório uniforme em RP·qL)
        # HEEAN: sampleUniform2(ax, N, logQQ)
        a_prime = self.crypto_params.generate_uniform_random_poly(n_degree, P_qL)

        # Sample e' ← DG(σ²) (erro gaussiano)
        # HEEAN: sampleGauss(ex, N, sigma)
        e_prime = self.crypto_params.generate_gaussian_poly(n_degree, sigma)

        # Calcular s² (s ao quadrado) seguindo padrão HEEAN exato
        # HEEAN Step 1: mult(sxsx, sx, sx, Q, N) -> sxsx = sx^2 mod Q
        s_squared = self.crypto_params.poly_mul_mod(
            s, s, qL, ring_poly_mod
        )  # mod qL (Q in HEEAN)

        # HEEAN Step 2: leftShiftAndEqual(sxsx, logQ, QQ, N) -> sxsx = Q*sx^2 mod QQ
        # leftShift by logQ bits means multiply by Q (2^logQ = Q)
        # In our case: multiply by P (the shift factor)
        P_s_squared = P * s_squared

        # Apply reduction mod P*qL (like HEEAN reduces mod QQ after shift)
        P_s_squared = self.crypto_params.poly_ring_mod(P_s_squared, ring_poly_mod, P_qL)

        # HEEAN Step 3: addAndEqual(ex, sxsx, QQ, N) -> ex = ex + sxsx = e' + Q*sx^2
        # Add P*s^2 to error polynomial
        e_prime = e_prime + P_s_squared

        # Apply reduction mod P*qL (HEEAN uses mod QQ)
        e_prime = self.crypto_params.poly_ring_mod(e_prime, ring_poly_mod, P_qL)

        # HEEAN Step 4: mult(bx, sx, ax, QQ, N) -> bx = sx * ax mod QQ
        # Compute s * a_prime
        s_a_prime = self.crypto_params.poly_mul_mod(s, a_prime, P_qL, ring_poly_mod)

        # HEEAN Step 5: sub(bx, ex, bx, QQ, N) -> bx = ex - bx = (e' + Q*sx^2) - sx*ax
        # Final: b' = e' - s*a' = -s*a' + e' + P*s^2 (matching HEEAN pattern)
        b_prime = e_prime - s_a_prime

        # Apply final reduction mod P*qL
        b_prime_final = self.crypto_params.poly_ring_mod(b_prime, ring_poly_mod, P_qL)

        # Retorna evk ← (b', a') ∈ R²_{P·qL}
        return (b_prime_final, a_prime)

    def generate_key_switching_key(
        self,
        old_secret_key: Polynomial,
        new_secret_key: Polynomial,
        level: int = None,
        P: int = None,
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma Key Switching Key (KSK) para transformar ciphertexts de uma chave para outra.

        Conforme a definição de Key Switching:
        KSK_SK'(s) = ([-(a·s' + e) + P·s]_{P·q}, a)

        Onde:
        - s é a chave secreta original (old_secret_key)
        - s' é a nova chave secreta (new_secret_key)
        - a é um polinômio aleatório em R_{P·q}
        - e é um erro gaussiano pequeno
        - P é um fator de escala auxiliar

        Esta chave permite transformar um ciphertext que criptografa m sob SK = (1, s)
        em um ciphertext que criptografa a mesma mensagem m sob SK' = (1, s').

        Args:
            old_secret_key: Chave secreta original s
            new_secret_key: Nova chave secreta s'
            level: Nível na cadeia de módulos (usa o maior se None)
            P: Parâmetro de escala auxiliar (usa valor padrão se None)

        Returns:
            Tuple[Polynomial, Polynomial]: Tupla (ksk0, ksk1) onde:
                - ksk0 = [-(a·s' + e) + P·s]_{P·q}
                - ksk1 = a
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1

        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        q = self.crypto_params.MODULUS_CHAIN[level]
        sigma = self.crypto_params.GAUSSIAN_NOISE_STDDEV

        # Usar P padrão se não fornecido
        if P is None:
            P = self.crypto_params.P

        P_q = P * q  # P · q

        # Verificar overflow
        max_int64 = 2**63 - 1
        if P_q > max_int64:
            P = max_int64 // q
            P_q = P * q
            print(f"Aviso: P ajustado para {P} para evitar overflow")

        # Sample a ← R_{P·q} (componente aleatório uniforme em R_{P·q})
        a = self.crypto_params.generate_uniform_random_poly(n_degree, P_q)

        # Sample e ← DG(σ²) (erro gaussiano)
        e = self.crypto_params.generate_gaussian_poly(n_degree, sigma)

        # Calcular a · s' (mod P·q)
        a_s_prime = self.crypto_params.poly_mul_mod(
            a, new_secret_key, P_q, ring_poly_mod
        )

        # Calcular P · s (escalar vezes polinômio)
        P_s = P * old_secret_key

        # Calcular ksk0 = [-(a·s' + e) + P·s]_{P·q}
        # Primeiro: -(a·s' + e)
        neg_a_s_prime_plus_e = -a_s_prime - e

        # Depois: -(a·s' + e) + P·s
        ksk0 = neg_a_s_prime_plus_e + P_s

        # Aplicar redução modular
        ksk0_final = self.crypto_params.poly_coeffs_mod_q(ksk0, P_q)

        # ksk1 é simplesmente a
        ksk1 = a

        # Retornar KSK_SK'(s) = (ksk0, ksk1)
        return (ksk0_final, ksk1)

    def generate_keypair(
        self, level: int = None, hamming_weight: int = None
    ) -> Tuple[Tuple[Polynomial, Polynomial], Tuple[Polynomial, Polynomial]]:
        """
        Gera um par completo de chaves (secreta e pública) seguindo a definição KeyGen.

        Args:
            level: Nível na cadeia de módulos (usa o maior se None)
            hamming_weight: Peso de Hamming para chave secreta (usa padrão se None)

        Returns:
            Tuple: (secret_key, public_key) onde:
                - secret_key: sk = (1, s) onde s ← HWT(h)
                - public_key: (b, a) onde b ← −as + e (mod qL)
        """
        secret_key = self.generate_secret_key(hamming_weight)
        public_key = self.generate_public_key(secret_key, level)
        return secret_key, public_key

    def generate_full_keyset(
        self, level: int = None, hamming_weight: int = None
    ) -> Dict[str, Tuple[Polynomial, Polynomial]]:
        """
        Gera um conjunto completo de chaves para operações CKKS seguindo a definição KeyGen.

        Args:
            level: Nível na cadeia de módulos (usa o maior se None)
            hamming_weight: Peso de Hamming para chave secreta (usa padrão se None)

        Returns:
            Dict: Dicionário contendo todas as chaves:
                - 'secret_key': sk = (1, s) onde s ← HWT(h)
                - 'public_key': (b, a)
                - 'evaluation_key': (b', a')
        """
        secret_key = self.generate_secret_key(hamming_weight)
        public_key = self.generate_public_key(secret_key, level)
        evaluation_key = self.generate_evaluation_key(secret_key, level)

        return {
            "secret_key": secret_key,
            "public_key": public_key,
            "evaluation_key": evaluation_key,
        }

    def validate_keypair(
        self,
        secret_key: Polynomial,
        public_key: Tuple[Polynomial, Polynomial],
        level: int = None,
    ) -> bool:
        """
        Valida se um par de chaves é consistente.

        Args:
            secret_key: Chave secreta para validar
            public_key: Chave pública para validar
            level: Nível na cadeia de módulos (usa o maior se None)

        Returns:
            bool: True se as chaves são consistentes, False caso contrário
        """
        try:
            if level is None:
                level = len(self.crypto_params.MODULUS_CHAIN) - 1

            pk_b, pk_a = public_key
            ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
            q_mod = self.crypto_params.MODULUS_CHAIN[level]

            # Calcula pk_a * secret_key + pk_b
            product = self.crypto_params.poly_mul_mod(
                pk_a, secret_key, q_mod, ring_poly_mod
            )
            result = (product + pk_b) % ring_poly_mod

            # Aplica redução modular aos coeficientes
            result_coeffs = self.crypto_params.poly_coeffs_mod_q(result, q_mod)

            # Verifica se o resultado tem coeficientes pequenos (indicando ruído)
            # Para CKKS, o ruído deve ter magnitude muito menor que q_mod
            if hasattr(result_coeffs, "coef"):
                max_coeff = np.max(np.abs(result_coeffs.coef))
            else:
                max_coeff = np.max(np.abs(result_coeffs))

            # Tolerância ajustada para o ruído gaussiano CKKS
            # O ruído deve ser da ordem de sigma_err, muito menor que q_mod
            noise_threshold = min(
                q_mod // 1000, 1000 * self.crypto_params.GAUSSIAN_NOISE_STDDEV
            )

            return max_coeff < noise_threshold

        except Exception as e:
            # Em caso de erro, considera inválido
            print(f"Erro na validação: {e}")
            return False


# Função de conveniência para criar instância da fábrica de chaves
def create_key_factory(
    crypto_params: CKKSCryptographicParameters = None,
) -> CKKSKeyFactory:
    """
    Cria uma nova instância da fábrica de chaves CKKS.

    Args:
        crypto_params: Parâmetros criptográficos (usa padrão se None)

    Returns:
        CKKSKeyFactory: Nova instância da fábrica de chaves
    """
    return CKKSKeyFactory(crypto_params)
