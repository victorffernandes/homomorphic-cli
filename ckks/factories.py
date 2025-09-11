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

        # Cria vetor complexo para FFT
        z = np.zeros(n_poly_coeffs // 2 + 1, dtype=np.float64)
        z[: len(real_vector)] = np.array(real_vector, dtype=np.float64) * delta_scale

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

        pk_b, pk_a = public_key

        # Gera ruídos aleatórios
        u = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        e1 = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)
        e2 = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)

        # Calcula componentes do ciphertext
        c0 = (
            self.crypto_params.poly_mul_mod(pk_b, u, q_mod, ring_poly_mod)
            + e1
            + message_poly
        )
        c1 = self.crypto_params.poly_mul_mod(pk_a, u, q_mod, ring_poly_mod) + e2

        # Aplica redução modular
        c0_final = self.crypto_params.poly_ring_mod(c0, ring_poly_mod, q_mod)
        c1_final = self.crypto_params.poly_ring_mod(c1, ring_poly_mod, q_mod)

        return CKKSCiphertext(
            components=[c0_final, c1_final],
            level=level,
            scale=self.crypto_params.SCALING_FACTOR,
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

        level = ciphertext.level
        q_mod = self.crypto_params.MODULUS_CHAIN[level]
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()

        if ciphertext.size == 2:
            # Descriptografia padrão para ciphertext de 2 componentes: m = c0 + c1*s
            c0 = ciphertext.get_component(0)
            c1 = ciphertext.get_component(1)

            # Calcula c1 * s
            c1_s = self.crypto_params.poly_mul_mod(c1, secret_key, q_mod, ring_poly_mod)

            # Descriptografa: c0 + c1*s
            decrypted_poly = c0 + c1_s

        elif ciphertext.size == 3:
            # Descriptografia para ciphertext de 3 componentes seguindo a fórmula CKKS:
            # Fórmula: C'₁ + s·C'₂ + s²·C'₃
            # Aplicar poly_mul_mod apenas no final

            c0 = ciphertext.get_component(0)  # C'₁
            c1 = ciphertext.get_component(1)  # C'₂
            c2 = ciphertext.get_component(2)  # C'₃

            # Calcular s² (multiplicação simples, sem redução modular ainda)
            s_squared = self.crypto_params.poly_mul(secret_key, secret_key)

            # Calcular cada termo sem redução modular intermediária:
            # Termo 1: C'₁ (já está correto)
            # Termo 2: s · C'₂
            term2 = self.crypto_params.poly_mul(secret_key, c1)

            # Termo 3: s² · C'₃
            term3 = self.crypto_params.poly_mul(s_squared, c2)

            # Soma final: C'₁ + s·C'₂ + s²·C'₃
            decrypted_poly = c0 + term2 + term3

        else:
            raise ValueError(
                f"Descriptografia não suportada para ciphertext com {ciphertext.size} componentes"
            )

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
    Fábrica para geração e gerenciamento de chaves CKKS.

    Esta classe encapsula a geração de chaves secretas, públicas e de relinearização
    para o esquema CKKS, seguindo as melhores práticas de segurança.
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

    def generate_secret_key(self) -> Polynomial:
        """
        Gera uma chave secreta para o esquema CKKS.

        Returns:
            Polynomial: Chave secreta gerada com distribuição gaussiana
        """
        return self.crypto_params.generate_gaussian_poly(
            self.crypto_params.POLYNOMIAL_DEGREE,
            self.crypto_params.GAUSSIAN_NOISE_STDDEV,
        )

    def generate_public_key(
        self, secret_key: Polynomial, level: int = None
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma chave pública a partir da chave secreta.

        Args:
            secret_key: Chave secreta para derivar a chave pública
            level: Nível na cadeia de módulos (usa o maior se None)

        Returns:
            Tuple[Polynomial, Polynomial]: Tupla (pk_b, pk_a) da chave pública
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1

        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        q_mod = self.crypto_params.MODULUS_CHAIN[level]
        sigma_err = self.crypto_params.GAUSSIAN_NOISE_STDDEV

        # Gera componente aleatório pk_a
        pk_a = self.crypto_params.generate_uniform_random_poly(n_degree, q_mod)

        # Gera erro gaussiano
        e_err = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)

        # Calcula pk_b = -(pk_a * secret_key) + e_err
        neg_a_s = -self.crypto_params.poly_mul_mod(
            pk_a, secret_key, q_mod, ring_poly_mod
        )
        pk_b = (neg_a_s + e_err) % ring_poly_mod

        # Aplica redução modular
        pk_b_final = self.crypto_params.poly_ring_mod(pk_b, ring_poly_mod, q_mod)

        return (pk_b_final, pk_a)

    def generate_relinearization_key(
        self, secret_key: Polynomial, level: int = None
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma chave de relinearização para o esquema CKKS.

        Args:
            secret_key: Chave secreta para derivar a chave de relinearização
            level: Nível na cadeia de módulos (usa o maior se None)

        Returns:
            Tuple[Polynomial, Polynomial]: Tupla (rlk_b, rlk_a) da chave de relinearização
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1

        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        q_mod = self.crypto_params.MODULUS_CHAIN[level]
        sigma_err = self.crypto_params.GAUSSIAN_NOISE_STDDEV

        # Calcula secret_key^2
        sk_s_squared = self.crypto_params.poly_mul_mod(
            secret_key, secret_key, q_mod, ring_poly_mod
        )

        # Gera componente aleatório rlk_a
        rlk_a = self.crypto_params.generate_uniform_random_poly(n_degree, q_mod)

        # Gera erro gaussiano
        e_err = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)

        # Calcula rlk_b = -(rlk_a * secret_key) + e_err + secret_key^2
        neg_a_s = -self.crypto_params.poly_mul_mod(
            rlk_a, secret_key, q_mod, ring_poly_mod
        )
        rlk_b = (neg_a_s + e_err + sk_s_squared) % ring_poly_mod

        # Aplica redução modular
        rlk_b_final = self.crypto_params.poly_ring_mod(rlk_b, ring_poly_mod, q_mod)

        return (rlk_b_final, rlk_a)

    def generate_evaluation_key(
        self, secret_key: Polynomial, level: int = None, P: int = None
    ) -> Tuple[Polynomial, Polynomial]:
        """
        Gera uma Evaluation Key (EVK) para reduzir ciphertexts de 3 para 2 componentes.

        Fórmula: EVK = (EVK1, EVK2) = ([-(a·s + e) + P·s²]_{P·q}, a)
        onde P é um parâmetro inteiro (aproximadamente do tamanho de q para baixo ruído),
        a é um polinômio aleatório em R_{P·q}, e e é um polinômio pequeno aleatório.

        Args:
            secret_key: Chave secreta para derivar a evaluation key
            level: Nível na cadeia de módulos (usa o maior se None)
            P: Parâmetro P (usa q/1000 se None para evitar overflow)

        Returns:
            Tuple[Polynomial, Polynomial]: Tupla (evk1, evk2) da evaluation key
        """
        if level is None:
            level = len(self.crypto_params.MODULUS_CHAIN) - 1

        n_degree = self.crypto_params.POLYNOMIAL_DEGREE
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        q_mod = self.crypto_params.MODULUS_CHAIN[level]
        sigma_err = self.crypto_params.GAUSSIAN_NOISE_STDDEV

        # P deve ser menor para evitar overflow, mas ainda grande o suficiente
        if P is None:
            P = max(
                1000, min(q_mod // 100, 10000)
            )  # P muito menor para evitar problemas

        # Verificar se P*q não causa overflow em int64
        pq_mod = P * q_mod
        max_int64 = 2**63 - 1
        if pq_mod > max_int64:
            # Ajustar P para não causar overflow
            P = max_int64 // q_mod
            pq_mod = P * q_mod
            print(f"Aviso: P ajustado para {P} para evitar overflow")

        # Calcular s² (secret_key ao quadrado)
        s_squared = self.crypto_params.poly_mul_mod(
            secret_key, secret_key, pq_mod, ring_poly_mod
        )

        # Calcular P·s²
        p_s_squared = (P * s_squared) % ring_poly_mod

        # Gerar componente aleatório a em R_{P·q}
        evk_a = self.crypto_params.generate_uniform_random_poly(n_degree, pq_mod)

        # Gerar erro gaussiano pequeno
        e_err = self.crypto_params.generate_gaussian_poly(n_degree, sigma_err)

        # Calcular evk1 = [-(a·s + e) + P·s²]_{P·q}
        # Primeiro calcula a·s
        a_s = self.crypto_params.poly_mul_mod(evk_a, secret_key, pq_mod, ring_poly_mod)

        # Depois calcula -(a·s + e) + P·s²
        evk1 = (-a_s - e_err + p_s_squared) % ring_poly_mod

        # Aplica redução modular final
        evk1_final = self.crypto_params.poly_ring_mod(evk1, ring_poly_mod, pq_mod)

        return (evk1_final, evk_a)

    def generate_keypair(
        self, level: int = None
    ) -> Tuple[Polynomial, Tuple[Polynomial, Polynomial]]:
        """
        Gera um par completo de chaves (secreta e pública).

        Args:
            level: Nível na cadeia de módulos (usa o maior se None)

        Returns:
            Tuple: (secret_key, public_key) onde public_key = (pk_b, pk_a)
        """
        secret_key = self.generate_secret_key()
        public_key = self.generate_public_key(secret_key, level)
        return secret_key, public_key

    def generate_full_keyset(
        self, level: int = None
    ) -> Dict[str, Union[Polynomial, Tuple[Polynomial, Polynomial]]]:
        """
        Gera um conjunto completo de chaves para operações CKKS.

        Args:
            level: Nível na cadeia de módulos (usa o maior se None)

        Returns:
            Dict: Dicionário contendo todas as chaves:
                - 'secret_key': Chave secreta
                - 'public_key': Chave pública (pk_b, pk_a)
                - 'relinearization_key': Chave de relinearização (rlk_b, rlk_a)
                - 'evaluation_key': Chave de avaliação (evk1, evk2)
        """
        secret_key = self.generate_secret_key()
        public_key = self.generate_public_key(secret_key, level)
        relinearization_key = self.generate_relinearization_key(secret_key, level)
        evaluation_key = self.generate_evaluation_key(secret_key, level)

        return {
            "secret_key": secret_key,
            "public_key": public_key,
            "relinearization_key": relinearization_key,
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
