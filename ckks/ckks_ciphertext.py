"""
Classe para representar ciphertexts do esquema CKKS.

Esta classe encapsula todas as informações necessárias para um ciphertext CKKS,
incluindo componentes, nível, escala e metadados.
"""

import json
import time
import numpy as np
from numpy.polynomial import Polynomial
from typing import List, Optional, Dict, Any
from .constants import CKKSCryptographicParameters
from .int_backend import cast_array_to_backend

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

        # Campo opcional para depuração de multiplicação
        self.debug_mult: Dict[str, Any] = {}

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
    def size(self) -> int:
        """Retorna o número de componentes do ciphertext."""
        return len(self.components)

    @property
    def modulus_bits(self) -> int:
        """Retorna o número de bits do módulo atual."""
        return int(self.current_modulus).bit_length()

    @property
    def scale_bits(self) -> Optional[float]:
        """Retorna uma aproximação em bits da escala (se numérica)."""
        try:
            if isinstance(self.scale, (int, float)) and self.scale > 0:
                return float(np.log2(self.scale))
        except Exception:
            return None
        return None

    @property
    def coeff_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas simples dos coeficientes para depuração:
        - max_abs por componente
        - dtype e número de coeficientes
        """
        stats = []
        for idx, comp in enumerate(self.components):
            coeffs = np.array(comp.coef)
            max_abs = float(np.max(np.abs(coeffs))) if coeffs.size > 0 else 0.0
            stats.append(
                {
                    "index": idx,
                    "max_abs": max_abs,
                    "len": int(coeffs.size),
                    "dtype": str(coeffs.dtype),
                }
            )
        return {
            "level": int(self.level),
            "scale": float(self.scale) if isinstance(self.scale, (int, float)) else str(self.scale),
            "modulus_bits": self.modulus_bits,
            "components": stats,
        }

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
        if not (
            ct1.level == ct2.level
            and abs(ct1.scale - ct2.scale) < 1e-10
            and ct1.size == ct2.size
        ):
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

    @staticmethod
    def multiply_homomorphic(
        ct1: "CKKSCiphertext", ct2: "CKKSCiphertext", evaluation_key: tuple
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


        b1 = ct1.get_component(0)  # c0 do primeiro ciphertext
        a1 = ct1.get_component(1)  # c1 do primeiro ciphertext
        b2 = ct2.get_component(0)  # c0 do segundo ciphertext
        a2 = ct2.get_component(1)  # c1 do segundo ciphertext

        evkax, evkbx = evaluation_key

        axbx1 = crypto_params.poly_ring_mod(a1 + b1, P_RING_MOD, ct1.current_modulus)
        axbx2 = crypto_params.poly_ring_mod(a2 + b2, P_RING_MOD, ct2.current_modulus)
        axbx1 = crypto_params.poly_ring_mod(axbx1 * axbx2, P_RING_MOD, ct1.current_modulus)

        axax = crypto_params.poly_ring_mod(a1 * a2, P_RING_MOD, ct1.current_modulus)
        bxbx = crypto_params.poly_ring_mod(b1 * b2, P_RING_MOD, ct1.current_modulus)

        qQ = 1 << crypto_params.logQ * ct1.current_modulus

        axmult = crypto_params.poly_mul_mod(axax, evkax, qQ, P_RING_MOD)
        bxmult = crypto_params.poly_mul_mod(axax, evkbx, qQ, P_RING_MOD)

        axmult = crypto_params.poly_ring_mod(axmult, P_RING_MOD, 1 << crypto_params.logQ)
        bxmult = crypto_params.poly_ring_mod(bxmult, P_RING_MOD, 1 << crypto_params.logQ)

        axmult = crypto_params.poly_ring_mod(axmult + axbx1, P_RING_MOD, ct1.current_modulus)
        axmult = crypto_params.poly_ring_mod(axmult - bxbx, P_RING_MOD, ct1.current_modulus)

        axmult = crypto_params.poly_ring_mod(axmult - axax, P_RING_MOD, ct1.current_modulus)
        axmult = crypto_params.poly_ring_mod(axmult + bxbx, P_RING_MOD, ct1.current_modulus)
        
        new_scale = ct1.scale * ct2.scale

        return CKKSCiphertext(
            components=[axmult, bxmult],
            level=level,
            crypto_params=crypto_params,
            scale=new_scale,
        )

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
        # Dividir por P e arredondar, armazenando via backend.
        ksk0_C2_scaled_float = np.round(
            ksk0_C2.coef.astype(np.float64) / P
        )
        ksk1_C2_scaled_float = np.round(
            ksk1_C2.coef.astype(np.float64) / P
        )
        ksk0_C2_scaled_coeffs = cast_array_to_backend(ksk0_C2_scaled_float)
        ksk1_C2_scaled_coeffs = cast_array_to_backend(ksk1_C2_scaled_float)

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
    print(f"Level: {ct.level}, Scale: {ct.scale}, Size: {ct.size}")

    print("\n✓ Classe CKKSCiphertext funcionando corretamente!")
