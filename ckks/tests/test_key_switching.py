"""
Testes para o método de Key Switching do esquema CKKS.
"""

import numpy as np
import pytest
from ckks.ciphertext_factory import CKKSCiphertextFactory
from ckks.key_factory import CKKSKeyFactory
from ckks.ckks import CKKSCiphertext
from ckks.constants import CKKSCryptographicParameters


class TestKeySwitching:
    """Testes para a operação de Key Switching"""

    def setup_method(self):
        """Configuração executada antes de cada teste"""
        self.crypto_params = CKKSCryptographicParameters()
        self.key_factory = CKKSKeyFactory(self.crypto_params)
        self.ciphertext_factory = CKKSCiphertextFactory(self.crypto_params)

    def test_key_switching_basic(self):
        """Teste básico de key switching"""
        print("\n" + "=" * 70)
        print("TESTE BÁSICO DE KEY SWITCHING")
        print("=" * 70)

        # Gerar primeira chave secreta e chaves públicas
        secret_key_1 = self.key_factory.generate_secret_key()
        public_key_1 = self.key_factory.generate_public_key(secret_key_1)

        # Gerar segunda chave secreta
        secret_key_2 = self.key_factory.generate_secret_key()

        # Gerar Key Switching Key: KSK_SK2(s1)
        # Esta chave permite transformar ciphertexts de SK1 para SK2
        key_switching_key = self.key_factory.generate_key_switching_key(
            secret_key_1, secret_key_2
        )

        # Criar mensagem de teste
        message = np.array([1.5, 2.5, 3.5, 4.5] + [0.0] * 508)
        print(f"Mensagem original: {message[:4]}")

        # Criptografar com a primeira chave
        ct1 = self.ciphertext_factory.encode_and_encrypt(message, public_key_1)
        print(f"Ciphertext criptografado com SK1: {ct1}")

        # Descriptografar com a primeira chave (deve funcionar)
        decrypted_1 = self.ciphertext_factory.decrypt_and_decode(ct1, secret_key_1, 4)
        print(f"Descriptografado com SK1: {decrypted_1}")

        error_1 = np.max(np.abs(decrypted_1 - message[:4]))
        print(f"Erro com SK1: {error_1:.6f}")

        # Aplicar key switching
        ct2 = CKKSCiphertext.key_switching(ct1, key_switching_key)
        print(f"Ciphertext após key switching: {ct2}")

        # Descriptografar com a segunda chave (deve funcionar após key switching)
        decrypted_2 = self.ciphertext_factory.decrypt_and_decode(ct2, secret_key_2, 4)
        print(f"Descriptografado com SK2: {decrypted_2}")

        error_2 = np.max(np.abs(decrypted_2 - message[:4]))
        print(f"Erro com SK2: {error_2:.6f}")

        # Verificar que o key switching não introduziu erro excessivo adicional
        # O erro após key switching deve ser comparável ao erro inicial
        # (dentro de um fator de tolerância)
        error_increase = error_2 / error_1 if error_1 > 0 else float("inf")
        print(f"Aumento do erro: {error_increase:.2f}x")

        # O key switching pode introduzir algum erro adicional, mas não deve ser excessivo
        # Aceitamos até 2x o erro original como tolerável
        assert error_increase < 2.0, (
            f"Key switching introduziu muito erro adicional: "
            f"{error_1:.6f} -> {error_2:.6f} ({error_increase:.2f}x)"
        )

        print("✅ Key switching básico funcionou!")

    def test_key_switching_validation(self):
        """Teste de validação de parâmetros do key switching"""
        # Gerar chaves
        secret_key_1 = self.key_factory.generate_secret_key()
        public_key_1 = self.key_factory.generate_public_key(secret_key_1)
        secret_key_2 = self.key_factory.generate_secret_key()

        # Criar ciphertext de 3 componentes (não suportado)
        c0 = self.crypto_params.generate_uniform_random_poly()
        c1 = self.crypto_params.generate_uniform_random_poly()
        c2 = self.crypto_params.generate_uniform_random_poly()

        ct_3comp = CKKSCiphertext(
            components=[c0, c1, c2],
            level=2,
            crypto_params=self.crypto_params,
        )

        # Key switching key inválida (apenas 1 componente)
        invalid_ksk = (c0,)

        # Deve falhar com ciphertext de 3 componentes
        with pytest.raises(ValueError, match="exatamente 2 componentes"):
            CKKSCiphertext.key_switching(ct_3comp, (c0, c1))

        # Deve falhar com KSK inválida
        message = np.array([1.0, 2.0] + [0.0] * 510)
        ct_valid = self.ciphertext_factory.encode_and_encrypt(message, public_key_1)

        with pytest.raises(ValueError, match="exatamente 2 componentes"):
            CKKSCiphertext.key_switching(ct_valid, invalid_ksk)

        print("✅ Validações de key switching funcionaram!")


if __name__ == "__main__":
    # Executar testes manualmente
    test = TestKeySwitching()
    test.setup_method()

    try:
        test.test_key_switching_basic()
    except Exception as e:
        print(f"⚠️ Teste básico falhou: {e}")
        print("Nota: Precisa implementar generate_key_switching_key no KeyFactory")

    try:
        test.test_key_switching_validation()
    except Exception as e:
        print(f"⚠️ Teste de validação falhou: {e}")
