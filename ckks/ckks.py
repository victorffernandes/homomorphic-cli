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
    para operações de criptografia homomórfica CKKS.

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
        scale: float,
        crypto_params: Optional[CKKSCryptographicParameters] = None,
    ):
        """
        Inicializa um novo ciphertext CKKS.

        Args:
            components: Lista de polinômios que formam o ciphertext
            level: Nível atual na cadeia de módulos (0 = menor módulo)
            scale: Fator de escala para decodificação
            crypto_params: Parâmetros criptográficos (usa instância padrão se None)

        Raises:
            ValueError: Se os parâmetros estiverem inválidos
        """
        if crypto_params is None:
            crypto_params = CKKSCryptographicParameters()

        self.crypto_params = crypto_params
        self._validate_initialization_params(components, level, scale)

        self.components = components.copy()
        self.level = level
        self.scale = scale

    def _validate_initialization_params(
        self, components: List[Polynomial], level: int, scale: float
    ):
        """Valida os parâmetros de inicialização."""
        if not components:
            raise ValueError("Lista de componentes não pode estar vazia")

        if not all(isinstance(comp, Polynomial) for comp in components):
            raise ValueError("Todos os componentes devem ser instâncias de Polynomial")

        max_level = len(self.crypto_params.MODULUS_CHAIN) - 1
        if level < 0 or level >= len(self.crypto_params.MODULUS_CHAIN):
            raise ValueError(f"Nível deve estar entre 0 e {max_level}")

        if scale <= 0:
            raise ValueError("Escala deve ser positiva")

    @classmethod
    def from_dict(
        cls, data: dict, crypto_params: Optional[CKKSCryptographicParameters] = None
    ):
        """
        Cria um CKKSCiphertext a partir de um dicionário.

        Args:
            data: Dicionário contendo 'c0', 'c1', 'level', 'scale'
            crypto_params: Parâmetros criptográficos

        Returns:
            CKKSCiphertext: Nova instância criada a partir do dicionário
        """
        if crypto_params is None:
            crypto_params = CKKSCryptographicParameters()

        components = []

        # Suporte para formato atual {'c0': poly, 'c1': poly, ...}
        if "c0" in data:
            components.append(data["c0"])
        if "c1" in data:
            components.append(data["c1"])
        if "c2" in data:  # Para casos de multiplicação antes da relinearização
            components.append(data["c2"])

        # Suporte para outros formatos possíveis
        if "components" in data:
            components = data["components"]

        return cls(
            components=components,
            level=data["level"],
            scale=data["scale"],
            crypto_params=crypto_params,
        )

    def to_dict(self) -> dict:
        """
        Converte o ciphertext para o formato de dicionário legado.

        Returns:
            dict: Dicionário no formato {'c0': poly, 'c1': poly, 'level': int, 'scale': float}
        """
        result = {"level": self.level, "scale": self.scale}

        # Mantém compatibilidade com código existente
        if len(self.components) >= 1:
            result["c0"] = self.components[0]
        if len(self.components) >= 2:
            result["c1"] = self.components[1]
        if len(self.components) >= 3:
            result["c2"] = self.components[2]

        return result

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
        """Verifica se é um ciphertext recém-criptografado."""
        return (
            self.level == len(self.crypto_params.MODULUS_CHAIN) - 1
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
            scale=self.scale,
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

    def __str__(self) -> str:
        """Representação string do ciphertext."""
        return (
            f"CKKSCiphertext(size={self.size}, level={self.level}, "
            f"scale={self.scale:.2e}, modulus={self.current_modulus})"
        )

    def __repr__(self) -> str:
        """Representação detalhada do ciphertext."""
        return (
            f"CKKSCiphertext(components={len(self.components)}, "
            f"level={self.level}/{len(self.crypto_params.MODULUS_CHAIN)-1}, "
            f"scale={self.scale}, modulus_bits={self.current_modulus.bit_length()})"
        )

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
            scale=ct1.scale,
            crypto_params=crypto_params,
        )


# Funções de conveniência para compatibilidade com código existente
def create_ciphertext_from_dict(data: dict) -> CKKSCiphertext:
    """Função de conveniência para criar ciphertext a partir de dicionário."""
    return CKKSCiphertext.from_dict(data)


def ciphertext_to_dict(ct: CKKSCiphertext) -> dict:
    """Função de conveniência para converter ciphertext para dicionário."""
    return ct.to_dict()


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
