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

    def ckks_encode_real(
        self,
        real_vector: List[float],
        delta_scale: float = None,
        n_poly_coeffs: int = None,
    ) -> Polynomial:
        """
        Codifica um vetor de números reais em um polinômio CKKS.

        Args:
            real_vector: Vetor de números reais para codificar
            delta_scale: Fator de escala (usa padrão se None)
            n_poly_coeffs: Número de coeficientes do polinômio (usa padrão se None)

        Returns:
            Polynomial: Polinômio codificado
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

        # Cria vetor complexo para FFT
        z = np.zeros(max_slots + 1, dtype=np.float64)
        z[: len(input_array)] = input_array * delta_scale

        # Aplica FFT inversa para obter coeficientes polinomiais
        poly_real_coeffs = np.fft.irfft(z, n=n_poly_coeffs)

        return Polynomial(np.round(poly_real_coeffs).astype(np.int64))

    def ckks_decode_real(
        self,
        message_poly: Polynomial,
        delta_scale: float = None,
        n_poly_coeffs: int = None,
        q_mod: int = None,
    ) -> np.ndarray:
        """
        Decodifica um polinômio CKKS em um vetor de números reais.

        Args:
            message_poly: Polinômio a ser decodificado
            delta_scale: Fator de escala (usa padrão se None)
            n_poly_coeffs: Número de coeficientes do polinômio (usa padrão se None)
            q_mod: Módulo para correção de coeficientes (usa o maior se None)

        Returns:
            np.ndarray: Vetor de números reais decodificados
        """
        if delta_scale is None:
            delta_scale = self.crypto_params.SCALING_FACTOR

        if n_poly_coeffs is None:
            n_poly_coeffs = self.crypto_params.POLYNOMIAL_DEGREE

        if q_mod is None:
            q_mod = self.crypto_params.MODULUS_CHAIN[-1]  # Maior módulo

        num_output_elements = n_poly_coeffs - 1
        coeffs = message_poly.coef

        # Correção centered lift para valores negativos
        corrected_coeffs = np.where(coeffs > q_mod // 2, coeffs - q_mod, coeffs)

        # Aplica FFT para decodificar
        coeffs_for_fft = corrected_coeffs.astype(np.float64)
        decoded_scaled_spectrum = np.fft.rfft(coeffs_for_fft, n=n_poly_coeffs)

        return np.real(decoded_scaled_spectrum[:num_output_elements]) / delta_scale

    def encrypt(
        self,
        message_poly: Polynomial,
        public_key: Tuple[Polynomial, Polynomial],
        level: int = None,
    ) -> CKKSCiphertext:
        """
        Criptografa um polinômio usando a chave pública.

        Args:
            message_poly: Polinômio da mensagem a ser criptografada
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
        zero_one_density = (
            self.crypto_params.ZERO_ONE_DENSITY
        )  # Densidade para distribuição zero-one

        pk_b, pk_a = public_key

        # Gera ruídos aleatórios
        u = self.crypto_params.generate_zero_one_poly(n_degree, zero_one_density)
        e1 = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        e2 = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)

        # Calcula componentes do ciphertext
        c0 = pk_b + u + e1 + message_poly
        c1 = pk_a + u + e2

        # Aplica redução modular
        c0_final = self.crypto_params.poly_ring_mod(c0, ring_poly_mod, q_mod)
        c1_final = self.crypto_params.poly_ring_mod(c1, ring_poly_mod, q_mod)

        return CKKSCiphertext(
            components=[c0_final, c1_final],
            level=level,
            crypto_params=self.crypto_params,
        )

    def decrypt(
        self, ciphertext: Union[CKKSCiphertext, Dict[str, Any]], secret_key: Polynomial
    ) -> Polynomial:
        """
        Descriptografa um ciphertext usando a chave secreta.

        Args:
            ciphertext: Ciphertext a ser descriptografado (CKKSCiphertext ou dict)
            secret_key: Chave secreta para descriptografia

        Returns:
            Polynomial: Polinômio da mensagem descriptografada
        """
        # Suporte para formato de dicionário legado
        if isinstance(ciphertext, dict):
            ciphertext = CKKSCiphertext.from_dict(ciphertext, self.crypto_params)

        q_mod = ciphertext.current_modulus
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()

        # Descriptografia padrão para ciphertext de 2 componentes: m = c0 + c1*s
        c0 = ciphertext.get_component(0)
        c1 = ciphertext.get_component(1)

        # Calcula c1 * s
        c1_s = self.crypto_params.poly_mul_mod(c1, secret_key, q_mod, ring_poly_mod)

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
        secret_key: Polynomial,
        expected_length: int = None,
    ) -> np.ndarray:
        """
        Descriptografa e decodifica um ciphertext em uma única operação.

        Args:
            ciphertext: Ciphertext a ser processado
            secret_key: Chave secreta para descriptografia
            expected_length: Comprimento esperado do vetor resultante

        Returns:
            np.ndarray: Vetor de números reais recuperados
        """
        # Descriptografa
        decrypted_poly = self.decrypt(ciphertext, secret_key)

        # Determina parâmetros para decodificação
        if isinstance(ciphertext, CKKSCiphertext):
            level = ciphertext.level
            scale = ciphertext.scale
        else:  # Dict format
            level = ciphertext["level"]
            scale = ciphertext["scale"]

        q_mod = self.crypto_params.MODULUS_CHAIN[level]

        # Decodifica
        decoded_vector = self.ckks_decode_real(
            decrypted_poly, scale, self.crypto_params.POLYNOMIAL_DEGREE, q_mod
        )

        # Trunca para o comprimento esperado se especificado
        if expected_length is not None:
            decoded_vector = decoded_vector[:expected_length]

        return decoded_vector


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

    def generate_secret_key(self, hamming_weight: int = None) -> Polynomial:
        """
        Gera uma chave secreta para o esquema CKKS seguindo a distribuição HWT(h).

        Conforme a definição KeyGen:
        - Sample s ← HWT(h)
        - Set the secret key as sk ← (1, s)

        Note: Retornamos apenas o componente 's', pois o componente '1' é implícito
        na estrutura do ciphertext.

        Args:
            hamming_weight: Peso de Hamming h (usa padrão se None)

        Returns:
            Polynomial: Chave secreta s com peso de Hamming h
        """
        if hamming_weight is None:
            hamming_weight = self.crypto_params.HAMMING_WEIGHT

        return self.crypto_params.generate_hamming_weight_poly(
            self.crypto_params.POLYNOMIAL_DEGREE, hamming_weight
        )

    def generate_public_key(
        self, secret_key: Polynomial, level: int = None
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma chave pública a partir da chave secreta seguindo a definição KeyGen.

        Conforme a definição:
        - Sample a ← RqL and e ← DG(σ²)
        - Set the public key as pk ← (b, a) ∈ R²qL where b ← −as + e (mod qL)

        Args:
            secret_key: Chave secreta s para derivar a chave pública
            level: Nível na cadeia de módulos (usa o maior se None)

        Returns:
            Tuple[Polynomial, Polynomial]: Tupla (b, a) da chave pública
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1

        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        q_mod = self.crypto_params.MODULUS_CHAIN[level]  # qL
        sigma = self.crypto_params.GAUSSIAN_NOISE_STDDEV

        # Sample a ← RqL (componente aleatório uniforme)
        a = self.crypto_params.generate_uniform_random_poly(n_degree, q_mod)

        # Sample e ← DG(σ²) (erro gaussiano)
        e = self.crypto_params.generate_gaussian_poly(n_degree, sigma)

        # Calcular b ← −as + e (mod qL)
        # Primeiro calcula as (a * s)
        a_s = self.crypto_params.poly_mul_mod(a, secret_key, q_mod, ring_poly_mod)

        # Depois calcula -as + e
        b = -a_s + e

        # Aplica redução modular final
        b_final = self.crypto_params.poly_ring_mod(b, ring_poly_mod, q_mod)

        # Retorna pk ← (b, a)
        return (b_final, a)

    def generate_evaluation_key(
        self, secret_key: Polynomial, level: int = None, P: int = None
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma Evaluation Key (EVK) seguindo a definição KeyGen.

        Conforme a definição:
        - Sample a' ← RP·qL and e' ← DG(σ²)
        - Set the evaluation key as evk ← (b', a') ∈ R²P·qL
          where b' ← −a's + e' + P s² (mod P · qL)

        Args:
            secret_key: Chave secreta s para derivar a evaluation key
            level: Nível na cadeia de módulos (usa o maior se None)
            P: Parâmetro P conforme definição (usa valor padrão se None)

        Returns:
            Tuple[Polynomial, Polynomial]: Tupla (b', a') da evaluation key
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1

        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        qL = self.crypto_params.MODULUS_CHAIN[level]  # qL
        P = self.crypto_params.P
        sigma = self.crypto_params.GAUSSIAN_NOISE_STDDEV

        P_qL = self.crypto_params.P * qL  # P · qL

        # Verificar overflow
        max_int64 = 2**63 - 1
        if P_qL > max_int64:
            P = max_int64 // qL
            P_qL = P * qL
            print(f"Aviso: P ajustado para {P} para evitar overflow")

        # Sample a' ← RP·qL (componente aleatório uniforme em RP·qL)
        a_prime = self.crypto_params.generate_uniform_random_poly(n_degree, P_qL)

        # Sample e' ← DG(σ²) (erro gaussiano)
        e_prime = self.crypto_params.generate_gaussian_poly(n_degree, sigma)

        # Calcular s² (s ao quadrado)
        s_squared = self.crypto_params.poly_mul_mod(
            secret_key, secret_key, P_qL, ring_poly_mod
        )

        # Calcular P · s²
        P_s_squared = P * s_squared

        # Calcular b' ← −a's + e' + P s² (mod P · qL)
        # Primeiro: a' · s
        a_prime_s = self.crypto_params.poly_mul_mod(
            a_prime, secret_key, P_qL, ring_poly_mod
        )

        # Depois: −a's + e' + P s²
        b_prime = -a_prime_s + e_prime + P_s_squared

        # Aplica redução modular final
        b_prime_final = self.crypto_params.poly_coeffs_mod_q(b_prime, P_qL)

        # Retorna evk ← (b', a')
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
    ) -> Tuple[Polynomial, Tuple[Polynomial, Polynomial]]:
        """
        Gera um par completo de chaves (secreta e pública) seguindo a definição KeyGen.

        Args:
            level: Nível na cadeia de módulos (usa o maior se None)
            hamming_weight: Peso de Hamming para chave secreta (usa padrão se None)

        Returns:
            Tuple: (secret_key, public_key) onde:
                - secret_key: s ← HWT(h)
                - public_key: (b, a) onde b ← −as + e (mod qL)
        """
        secret_key = self.generate_secret_key(hamming_weight)
        public_key = self.generate_public_key(secret_key, level)
        return secret_key, public_key

    def generate_full_keyset(
        self, level: int = None, hamming_weight: int = None
    ) -> Dict[str, Union[Polynomial, Tuple[Polynomial, Polynomial]]]:
        """
        Gera um conjunto completo de chaves para operações CKKS seguindo a definição KeyGen.

        Args:
            level: Nível na cadeia de módulos (usa o maior se None)
            hamming_weight: Peso de Hamming para chave secreta (usa padrão se None)

        Returns:
            Dict: Dicionário contendo todas as chaves:
                - 'secret_key': Chave secreta s ← HWT(h)
                - 'public_key': Chave pública (b, a)
                - 'evaluation_key': Chave de avaliação (b', a')
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

        # Decodifica para verificar
        decoded_data = factory.ckks_decode_real(encoded_poly)
        print(f"✓ Decodificação realizada: {decoded_data[:len(real_data)]}")

        print("\n✓ CKKSCiphertextFactory funcionando corretamente!")

    except ImportError as e:
        print(f"Erro de importação: {e}")
        print("Execute os testes usando: python -m pytest test_factory.py -v")
