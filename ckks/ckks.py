"""
Classe para representar ciphertexts do esquema CKKS.

Esta classe encapsula todas as informações necessárias para um ciphertext CKKS,
incluindo componentes, nível, escala e metadados.
"""

import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Optional
from .constants import CKKSCryptographicParameters


class CKKSCiphertext:
    """
    Classe que representa um ciphertext do esquema CKKS.

    Esta classe encapsula todos os componentes e metadados necessários
    para operações de cri    @staticmethod
    def multiply_homomorphic(
        ct1: "CKKSCiphertext",
        ct2: "CKKSCiphertext",
        evaluation_key: tuple,
        auto_rescale: bool = True  # Habilitando rescale automático novamente
    ) -> "CKKSCiphertext":ia homomórfica CKKS.

    Attributes:
        components: Lista de componentes polinomiais do ciphertext
        level: Nível atual na cadeia de módulos
        scale: Fator de escala atual
        crypto_params: Instância dos parâmetros criptográficos
    """

    def __init__(
        self,
        components: List[Polynomial],
        level: int,
        crypto_params: Optional[CKKSCryptographicParameters] = None,
        scale: Optional[int] = None,
    ):
        """
        Inicializa um novo ciphertext CKKS.

        Args:
            components: Lista de polinômios que formam o ciphertext
            level: Nível atual na cadeia de módulos (0 = menor módulo)
            crypto_params: Parâmetros criptográficos (usa instância padrão se None)
            scale: Fator de escala para decodificação (usa SCALING_FACTOR se None)

        Raises:
            ValueError: Se os parâmetros estiverem inválidos
        """
        if crypto_params is None:
            crypto_params = CKKSCryptographicParameters()

        self.crypto_params = crypto_params

        # Validar parâmetros primeiro antes de calcular a escala
        self._validate_initialization_params(components, level)

        # Usar scale fornecido ou SCALING_FACTOR padrão
        if scale is None:
            self.scale = self.crypto_params.SCALING_FACTOR
        else:
            self.scale = scale

        self.components = components.copy()
        self.level = level

    def _validate_initialization_params(self, components: List[Polynomial], level: int):
        """Valida os parâmetros de inicialização."""
        if not components:
            raise ValueError("Lista de componentes não pode estar vazia")

        if not all(isinstance(comp, Polynomial) for comp in components):
            raise ValueError("Todos os componentes devem ser instâncias de Polynomial")

        max_level = len(self.crypto_params.MODULUS_CHAIN) - 1
        if level < 0 or level >= len(self.crypto_params.MODULUS_CHAIN):
            raise ValueError(f"Nível deve estar entre 0 e {max_level}")

    @property
    def current_modulus(self) -> int:
        """Retorna o módulo atual baseado no nível."""
        return self.crypto_params.MODULUS_CHAIN[self.level]

    @property
    def noise_budget(self) -> int:
        """Retorna uma estimativa do orçamento de ruído restante."""
        return self.level  # Simplificado - o nível indica níveis restantes

    @property
    def size(self) -> int:
        """Retorna o número de componentes do ciphertext."""
        return len(self.components)

    def is_fresh(self) -> bool:
        """
        Verifica se é um ciphertext recém-criptografado.

        Um ciphertext é considerado "fresh" se:
        - Tem exatamente 2 componentes (não foi multiplicado)
        - Tem scale igual ao SCALING_FACTOR (não foi rescalado)
        """
        return (
            self.size == 2
            and abs(self.scale - self.crypto_params.SCALING_FACTOR) < 1e-10
        )

    def can_add_with(self, other: "CKKSCiphertext") -> bool:
        """
        Verifica se é possível somar com outro ciphertext.

        Args:
            other: Outro ciphertext CKKS

        Returns:
            bool: True se a adição for possível
        """
        return (
            self.level == other.level
            and abs(self.scale - other.scale) < 1e-10
            and self.size == other.size
        )

    def can_multiply_with(self, other: "CKKSCiphertext") -> bool:
        """
        Verifica se é possível multiplicar com outro ciphertext.

        Args:
            other: Outro ciphertext CKKS

        Returns:
            bool: True se a multiplicação for possível
        """
        return (
            self.level == other.level
            and self.level > 0  # Precisa de pelo menos um nível para rescale
            and self.size <= 2
            and other.size <= 2
        )  # Suporte apenas para ciphertexts de tamanho 2

    def copy(self) -> "CKKSCiphertext":
        """Cria uma cópia profunda do ciphertext."""
        return CKKSCiphertext(
            components=[Polynomial(comp.coef.copy()) for comp in self.components],
            level=self.level,
            crypto_params=self.crypto_params,
        )

    def get_component(self, index: int) -> Polynomial:
        """
        Retorna um componente específico do ciphertext.

        Args:
            index: Índice do componente (0, 1, 2, ...)

        Returns:
            Polynomial: O componente solicitado

        Raises:
            IndexError: Se o índice estiver fora do alcance
        """
        if index < 0 or index >= len(self.components):
            raise IndexError(
                f"Índice {index} fora do alcance. Ciphertext tem {len(self.components)} componentes."
            )
        return self.components[index]

    def update_after_rescale(self, new_level: int, new_scale: float):
        """
        Atualiza o ciphertext após operação de rescale.

        Args:
            new_level: Novo nível na cadeia de módulos
            new_scale: Nova escala após rescale
        """
        if new_level < 0 or new_level >= self.level:
            raise ValueError(
                "Novo nível deve ser menor que o nível atual e não negativo"
            )

        self.level = new_level
        self.scale = new_scale

    def print_summary(self):
        """Imprime um resumo detalhado do ciphertext."""
        print("=== RESUMO DO CIPHERTEXT CKKS ===")
        print(f"Número de componentes: {self.size}")
        print(
            f"Nível atual: {self.level} (de {len(self.crypto_params.MODULUS_CHAIN)-1})"
        )
        print(
            f"Módulo atual: {self.current_modulus} (~{self.current_modulus.bit_length()} bits)"
        )
        print(f"Escala: {self.scale:.2e}")
        print(f"Orçamento de ruído: {self.noise_budget} níveis restantes")
        print(f"Status: {'Fresh' if self.is_fresh() else 'Processado'}")

        for i, comp in enumerate(self.components):
            coeffs = np.array(comp.coef, dtype=np.int64)
            print(
                f"Componente {i}: {len(coeffs)} coeficientes, "
                f"max={np.max(np.abs(coeffs)):.2e}"
            )
        print("=" * 35)

    @staticmethod
    def add_homomorphic(
        ct1: "CKKSCiphertext", ct2: "CKKSCiphertext"
    ) -> "CKKSCiphertext":
        """
        Realiza adição homomórfica entre dois ciphertexts CKKS.

        Args:
            ct1: Primeiro ciphertext CKKS
            ct2: Segundo ciphertext CKKS

        Returns:
            CKKSCiphertext: Resultado da adição homomórfica

        Raises:
            ValueError: Se os ciphertexts não são compatíveis para adição
        """
        # Validação de compatibilidade
        if not ct1.can_add_with(ct2):
            raise ValueError(
                f"Ciphertexts não são compatíveis para adição. "
                f"ct1: level={ct1.level}, scale={ct1.scale}, size={ct1.size}; "
                f"ct2: level={ct2.level}, scale={ct2.scale}, size={ct2.size}"
            )

        # Obter parâmetros necessários
        level = ct1.level
        q_mod = ct1.current_modulus
        crypto_params = ct1.crypto_params
        ring_poly_mod = crypto_params.get_polynomial_modulus_ring()

        # Realizar adição componente por componente
        result_components = []
        for i in range(ct1.size):
            comp_add = ct1.components[i] + ct2.components[i]
            comp_mod = crypto_params.poly_ring_mod(comp_add, ring_poly_mod, q_mod)
            result_components.append(comp_mod)

        return CKKSCiphertext(
            components=result_components,
            level=level,
            crypto_params=crypto_params,
        )

    def rescale(self, target_level: int = None) -> "CKKSCiphertext":
        """
        Realiza operação de rescale conforme definição do paper CKKS.

        RS_{ℓ→ℓ'}(c): Para um ciphertext c ∈ R_q^k no nível ℓ e nível inferior ℓ' < ℓ,
        produz c' ← ⌊(q_{ℓ'}/q_ℓ) * c⌉ em R_{q_{ℓ'}}^k.

        Args:
            ct: Ciphertext CKKS para rescalonar
            target_level: Nível alvo (usa level-1 se None)

        Returns:
            CKKSCiphertext: Novo ciphertext rescalonado

        Raises:
            ValueError: Se não há mais níveis para rescalonar
        """
        if self.level == 0:
            raise ValueError("Não há mais níveis para rescalonar.")

        # Determina o nível alvo
        if target_level is None:
            target_level = self.level - 1

        if target_level < 0 or target_level >= self.level:
            raise ValueError(f"Nível alvo inválido: {target_level}")

        # Obter módulos da cadeia
        q_current = self.crypto_params.MODULUS_CHAIN[self.level]  # q_ℓ (módulo atual)
        q_target = self.crypto_params.MODULUS_CHAIN[
            target_level
        ]  # q_{ℓ-1} (módulo alvo)
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()

        # Calcula o fator de escala entre os módulos: q_{ℓ-1}/q_ℓ
        # Este fator é menor que 1, pois q_{ℓ-1} < q_ℓ no CKKS
        modulus_ratio = q_target / q_current

        rescaled_components = []
        for comp in self.components:
            # Multiplica por q_{ℓ-1}/q_ℓ e arredonda para o inteiro mais próximo
            scaled_coeffs = (comp.coef * modulus_ratio + 0.5).astype(np.int64)
            scaled_poly = Polynomial(scaled_coeffs)

            # Reduz para o anel R^n_{q_{ℓ-1}}
            comp_mod = self.crypto_params.poly_ring_mod(
                scaled_poly, ring_poly_mod, q_target
            )
            rescaled_components.append(comp_mod)

        # A nova escala é ajustada pelo mesmo fator q_{ℓ-1}/q_ℓ
        # Se p = q_ℓ / q_{ℓ-1}, então q_{ℓ-1}/q_ℓ = 1/p
        # Logo: new_scale = old_scale * (q_{ℓ-1}/q_ℓ) = old_scale / p
        new_scale = self.scale * modulus_ratio

        return CKKSCiphertext(
            components=rescaled_components,
            level=target_level,
            crypto_params=self.crypto_params,
            scale=new_scale,
        )

    @staticmethod
    def raw_multiply_homomorphic(
        ct1: "CKKSCiphertext", ct2: "CKKSCiphertext"
    ) -> "CKKSCiphertext":
        """
        Realiza multiplicação homomórfica raw conforme definição do paper CKKS.

        Para c1 = (b1, a1), c2 = (b2, a2) ∈ R²_{q_ℓ}, calcula:
        (d0, d1, d2) = (b1*b2, a1*b2 + a2*b1, a1*a2) (mod q_ℓ)

        Args:
            ct1: Primeiro ciphertext CKKS (deve ter exatamente 2 componentes)
            ct2: Segundo ciphertext CKKS (deve ter exatamente 2 componentes)

        Returns:
            CKKSCiphertext: Ciphertext resultante com 3 componentes

        Raises:
            ValueError: Se os ciphertexts não são compatíveis para multiplicação
        """
        # Validação de compatibilidade
        if not ct1.can_multiply_with(ct2):
            raise ValueError(
                f"Ciphertexts não são compatíveis para multiplicação. "
                f"ct1: level={ct1.level}, size={ct1.size}; "
                f"ct2: level={ct2.level}, size={ct2.size}"
            )

        if ct1.size != 2 or ct2.size != 2:
            raise ValueError(
                f"raw_multiply_homomorphic requer ciphertexts com exatamente 2 componentes. "
                f"ct1.size={ct1.size}, ct2.size={ct2.size}"
            )

        # Obter parâmetros
        level = ct1.level
        crypto_params = ct1.crypto_params
        P_RING_MOD = crypto_params.get_polynomial_modulus_ring()

        # Extrair componentes: c1 = (b1, a1), c2 = (b2, a2)
        b1 = ct1.get_component(0)  # c0 do primeiro ciphertext
        a1 = ct1.get_component(1)  # c1 do primeiro ciphertext
        b2 = ct2.get_component(0)  # c0 do segundo ciphertext
        a2 = ct2.get_component(1)  # c1 do segundo ciphertext

        # Calcular multiplicação conforme paper CKKS:
        # d0 = b1 * b2 (mod q_ℓ)
        d0 = b1 * b2

        # d1 = a1 * b2 + a2 * b1 (mod q_ℓ)
        a1_b2 = a1 * b2
        a2_b1 = a2 * b1
        d1 = a1_b2 + a2_b1

        # d2 = a1 * a2 (mod q_ℓ)
        d2 = a1 * a2

        # rescaling_factor = (
        #     crypto_params.MODULUS_CHAIN[ct1.level - 1]
        #     / crypto_params.MODULUS_CHAIN[ct1.level]
        # )
        # components = [
        #     d0 * rescaling_factor,
        #     d1 * rescaling_factor,
        #     d2 * rescaling_factor,
        # ]

        components = [
            d0,
            d1,
            d2,
        ]

        # components = [
        #     Polynomial(np.round(comp.coef).astype(np.int64)) for comp in components
        # ]
        components = [
            crypto_params.poly_ring_mod(
                comp,
                P_RING_MOD,
                ct1.current_modulus,
            )
            for comp in components
        ]

        # A nova escala é o produto das escalas dos ciphertexts de entrada
        # conforme paper CKKS: scale_mult = scale1 * scale2
        new_scale = ct1.scale * ct2.scale

        return CKKSCiphertext(
            components=components,
            level=level,
            crypto_params=crypto_params,
            scale=new_scale,
        )

    @staticmethod
    def relinearize(
        ciphertext: "CKKSCiphertext", evaluation_key: tuple
    ) -> "CKKSCiphertext":
        """
        Relineariza um ciphertext de 3 componentes para 2 componentes usando a Evaluation Key.

        Conforme paper CKKS, para ct = (d0, d1, d2) e EVK = (evk0, evk1) ∈ R²_{P·q_L}:

        Relin_{EVK}(ct) = (d0, d1) + ⌊(q_ℓ/P) · d2 · EVK⌋_{q_ℓ}
                        = (d0 + ⌊(q_ℓ/P)·d2·evk0⌋, d1 + ⌊(q_ℓ/P)·d2·evk1⌋)

        Processo:
        1. Calcular d2 · EVK em R_{P·q_ℓ}
        2. Multiplicar por (q_ℓ/P) e arredondar
        3. Adicionar a (d0, d1)
        4. Reduzir módulo q_ℓ

        onde:
        - EVK está em R_{P·q_L} (módulo P·q_L)
        - d2 está em R_{q_ℓ} (módulo q_ℓ)
        - O resultado final está em R_{q_ℓ}
        - ⌊·⌋ denota arredondamento para o inteiro mais próximo

        Args:
            ciphertext: Ciphertext de 3 componentes para relinearizar
            evaluation_key: Tupla (evk0, evk1) da Evaluation Key em R_{P·q_L}

        Returns:
            CKKSCiphertext: Novo ciphertext com 2 componentes

        Raises:
            ValueError: Se o ciphertext não tiver exatamente 3 componentes
        """
        if ciphertext.size != 3:
            raise ValueError(
                f"Relinearização requer ciphertext com exatamente 3 componentes. "
                f"Recebido: {ciphertext.size} componentes"
            )

        if len(evaluation_key) != 2:
            raise ValueError(
                f"Evaluation Key deve ter exatamente 2 componentes. "
                f"Recebido: {len(evaluation_key)} componentes"
            )

        # Extrair componentes do ciphertext
        d0 = ciphertext.components[0]  # Termo constante
        d1 = ciphertext.components[1]  # Termo linear
        d2 = ciphertext.components[2]  # Termo quadrático

        # Extrair componentes da Evaluation Key
        evk0, evk1 = evaluation_key

        # Obter parâmetros criptográficos
        crypto_params = ciphertext.crypto_params
        level = ciphertext.level
        q_ell = crypto_params.MODULUS_CHAIN[level]  # q_ℓ (módulo atual)
        P = crypto_params.P  # Fator auxiliar
        q0 = crypto_params.Q0
        ring_poly_mod = crypto_params.get_polynomial_modulus_ring()

        d2_ql_Pq = d2 * crypto_params.MODULUS_CHAIN[level] / (P * q0)

        # ETAPA 1: Calcular d2 · EVK em R_{P·q_ℓ}
        # EVK está em R_{P·q_L}, d2 está em R_{q_ℓ}
        # Multiplicamos usando módulo P·q_ℓ como aproximação
        d2_evk0 = crypto_params.poly_mul_mod(d2_ql_Pq, evk0, q_ell, ring_poly_mod)
        d2_evk1 = crypto_params.poly_mul_mod(d2_ql_Pq, evk1, q_ell, ring_poly_mod)

        d2_evk0_scaled = np.round(d2_evk0.coef).astype(np.int64)
        d2_evk1_scaled = np.round(d2_evk1.coef).astype(np.int64)

        d2_evk0_poly = Polynomial(d2_evk0_scaled)
        d2_evk1_poly = Polynomial(d2_evk1_scaled)

        # ETAPA 3: Adicionar a (d0, d1)
        # (c0', c1') = (d0, d1) + ⌊(q_ℓ/P)·d2·EVK⌋
        c0_prime = d0 + d2_evk0_poly
        c1_prime = d1 + d2_evk1_poly

        # ETAPA 4: Aplicar redução modular em R_{q_ℓ}
        c0_final = crypto_params.poly_ring_mod(c0_prime, ring_poly_mod, q_ell)
        c1_final = crypto_params.poly_ring_mod(c1_prime, ring_poly_mod, q_ell)

        # A relinearização preserva a escala do ciphertext de entrada
        return CKKSCiphertext(
            components=[c0_final, c1_final],
            level=level,
            crypto_params=crypto_params,
            scale=ciphertext.scale,
        )

    @staticmethod
    def multiply_homomorphic(
        ct1: "CKKSCiphertext",
        ct2: "CKKSCiphertext",
        evaluation_key: tuple,
    ) -> "CKKSCiphertext":
        """
        Multiplica dois ciphertexts homomorficamente seguindo o paper CKKS.

        Processo completo conforme CKKS paper:
        1. Multiplicação raw (resulta em 3 componentes)
        2. Relinearização usando EVK (reduz para 2 componentes)
        3. Rescale (reduz nível e normaliza escala)

        Args:
            ct1: Primeiro ciphertext (deve ter 2 componentes)
            ct2: Segundo ciphertext (deve ter 2 componentes)
            evaluation_key: Tupla (evk0, evk1) da Evaluation Key
            auto_rescale: Se True, aplica rescale automático após multiplicação

        Returns:
            CKKSCiphertext: Resultado da multiplicação completa

        Raises:
            ValueError: Se os ciphertexts não forem compatíveis ou EVK inválida
        """
        # Validações de entrada
        if ct1.size != 2:
            raise ValueError(
                f"ct1 deve ter exatamente 2 componentes. Recebido: {ct1.size}"
            )

        if ct2.size != 2:
            raise ValueError(
                f"ct2 deve ter exatamente 2 componentes. Recebido: {ct2.size}"
            )

        if len(evaluation_key) != 2:
            raise ValueError(
                f"Evaluation Key deve ter exatamente 2 componentes. "
                f"Recebido: {len(evaluation_key)}"
            )

        # Verificar compatibilidade dos ciphertexts
        if ct1.level != ct2.level:
            raise ValueError(
                f"Ciphertexts devem estar no mesmo nível. "
                f"ct1: level={ct1.level}, ct2: level={ct2.level}"
            )

        if ct1.crypto_params != ct2.crypto_params:
            raise ValueError(
                "Ciphertexts devem usar os mesmos parâmetros criptográficos"
            )

        # Etapa 1: Multiplicação raw (2 componentes → 3 componentes)
        ct_mult_raw = CKKSCiphertext.raw_multiply_homomorphic(ct1, ct2)

        # Etapa 2: Relinearização (3 componentes → 2 componentes)
        ct_result = CKKSCiphertext.relinearize(ct_mult_raw, evaluation_key)

        return ct_result

    @staticmethod
    def multiply_homomorphic_without_relin(
        ct1: "CKKSCiphertext", ct2: "CKKSCiphertext"
    ) -> "CKKSCiphertext":
        """
        Multiplica dois ciphertexts homomorficamente sem relinearização.

        Este é um alias conveniente para raw_multiply_homomorphic,
        fornecendo uma interface mais clara quando a relinearização
        não é desejada imediatamente.

        Args:
            ct1: Primeiro ciphertext (deve ter 2 componentes)
            ct2: Segundo ciphertext (deve ter 2 componentes)

        Returns:
            CKKSCiphertext: Resultado da multiplicação (3 componentes)
        """
        return CKKSCiphertext.raw_multiply_homomorphic(ct1, ct2)

    @staticmethod
    def key_switching(
        ciphertext: "CKKSCiphertext", key_switching_key: tuple, P: int = None
    ) -> "CKKSCiphertext":
        """
        Realiza key switching conforme definição do paper CKKS.

        Transforma um ciphertext C = (C₁, C₂) que criptografa plaintext m sob
        a chave secreta SK = (1, s) em um ciphertext C' que criptografa o mesmo
        plaintext m sob uma nova chave secreta SK' = (1, s').

        Key Switching Key (KSK_SK'(s)):
            KSK_SK'(s) = ([-(a·s' + e) + P·s]_{P·q}, a)

        onde:
            - a é um polinômio aleatório em R_{P·q}
            - e é um polinômio pequeno de erro
            - P é um fator de escala auxiliar
            - s é a chave secreta original
            - s' é a nova chave secreta

        A operação de key switching é:
            C' = (C₁, 0) + ⌊(KSK_SK'(s) · q_ℓ) / (P · q)⌋ · C₂

        Args:
            ciphertext: Ciphertext para realizar key switching (deve ter 2 componentes)
            key_switching_key: Tupla (ksk0, ksk1) onde:
                - ksk0 = [-(a·s' + e) + P·s]_{P·q}
                - ksk1 = a
            P: Fator de escala auxiliar (usa parâmetro padrão se None)

        Returns:
            CKKSCiphertext: Novo ciphertext criptografado sob a nova chave

        Raises:
            ValueError: Se o ciphertext não tiver exatamente 2 componentes
        """
        if ciphertext.size != 2:
            raise ValueError(
                f"Key switching requer ciphertext com exatamente 2 componentes. "
                f"Recebido: {ciphertext.size} componentes"
            )

        if len(key_switching_key) != 2:
            raise ValueError(
                f"Key Switching Key deve ter exatamente 2 componentes. "
                f"Recebido: {len(key_switching_key)} componentes"
            )

        # Extrair componentes do ciphertext: C = (C₁, C₂)
        C1 = ciphertext.components[0]
        C2 = ciphertext.components[1]

        # Extrair componentes da Key Switching Key
        ksk0, ksk1 = key_switching_key

        # Obter parâmetros criptográficos
        crypto_params = ciphertext.crypto_params
        level = ciphertext.level
        q_ell = crypto_params.MODULUS_CHAIN[level]
        ring_poly_mod = crypto_params.get_polynomial_modulus_ring()

        # Usar P padrão se não fornecido
        if P is None:
            P = crypto_params.P

        # Calcular KSK_SK'(s) · C₂ = (ksk0 · C₂, ksk1 · C₂)
        # Note: KSK está em R_{P·q}, então os produtos estão em escala P·q
        P_q = P * q_ell
        ksk0_C2 = crypto_params.poly_mul_mod(ksk0, C2, P_q, ring_poly_mod)
        ksk1_C2 = crypto_params.poly_mul_mod(ksk1, C2, P_q, ring_poly_mod)

        # Aplicar escalonamento e arredondamento: ⌊(KSK · C₂ · q_ℓ) / (P · q_ℓ)⌋
        # Simplifica para: ⌊(KSK · C₂) / P⌋
        # Dividir por P e arredondar
        ksk0_C2_scaled_coeffs = np.round(ksk0_C2.coef.astype(np.float64) / P).astype(
            np.int64
        )
        ksk1_C2_scaled_coeffs = np.round(ksk1_C2.coef.astype(np.float64) / P).astype(
            np.int64
        )

        # Criar polinômios com coeficientes arredondados
        ksk0_C2_scaled = Polynomial(ksk0_C2_scaled_coeffs)
        ksk1_C2_scaled = Polynomial(ksk1_C2_scaled_coeffs)

        # Calcular C' = (C₁, 0) + ⌊KSK · q_ℓ / (P·q)⌋ · C₂
        # Componente 0: C₁ + ksk0_C2_scaled
        C0_prime = C1 + ksk0_C2_scaled

        # Componente 1: 0 + ksk1_C2_scaled = ksk1_C2_scaled
        C1_prime = ksk1_C2_scaled

        # Aplicar redução modular
        C0_final = crypto_params.poly_ring_mod(C0_prime, ring_poly_mod, q_ell)
        C1_final = crypto_params.poly_ring_mod(C1_prime, ring_poly_mod, q_ell)

        # Retornar novo ciphertext (mantém nível original)
        return CKKSCiphertext(
            components=[C0_final, C1_final],
            level=level,
            crypto_params=crypto_params,
        )


if __name__ == "__main__":
    # Exemplo de uso
    from constants import CKKSCryptographicParameters

    crypto_params = CKKSCryptographicParameters()

    # Criar componentes de exemplo
    c0 = crypto_params.generate_uniform_random_poly()
    c1 = crypto_params.generate_uniform_random_poly()

    # Criar ciphertext
    ct = CKKSCiphertext(
        components=[c0, c1], level=2, scale=crypto_params.SCALING_FACTOR
    )

    # Demonstrar funcionalidades
    print("Exemplo de uso da classe CKKSCiphertext:")
    ct.print_summary()

    # Teste de conversão para dicionário
    dict_format = ct.to_dict()
    print(f"\nFormato dicionário: {list(dict_format.keys())}")

    # Teste de criação a partir de dicionário
    ct2 = CKKSCiphertext.from_dict(dict_format)
    print(f"Ciphertext recriado: {ct2}")

    print("\n✓ Classe CKKSCiphertext funcionando corretamente!")
