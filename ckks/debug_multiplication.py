#!/usr/bin/env python3
"""
Script de debug para investigar problemas na multiplicação homomórfica CKKS.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ckks.ckks import CKKSCiphertext
from ckks.constants import CKKSCryptographicParameters
from ckks.factories import CKKSCiphertextFactory, CKKSKeyFactory


def debug_multiplication():
    print("=== DEBUG: MULTIPLICAÇÃO HOMOMÓRFICA CKKS ===")

    # Criar parâmetros e factories
    crypto_params = CKKSCryptographicParameters()
    ciphertext_factory = CKKSCiphertextFactory(crypto_params)
    key_factory = CKKSKeyFactory(crypto_params)

    print(f"Parâmetros CKKS:")
    print(f"  N = {crypto_params.POLYNOMIAL_DEGREE}")
    print(f"  DELTA = {crypto_params.SCALING_FACTOR:.2e}")
    print(f"  Q_CHAIN = {[q.bit_length() for q in crypto_params.MODULUS_CHAIN]}")

    # Gerar chaves
    keyset = key_factory.generate_full_keyset()
    sk = keyset["secret_key"]
    pk = keyset["public_key"]
    evk = keyset["evaluation_key"]

    # Teste com valores simples
    print(f"\n--- TESTE 1: Valores simples ---")
    m1 = np.array([2.0] + [0.0] * 511)
    m2 = np.array([3.0] + [0.0] * 511)

    print(f"m1[0] = {m1[0]}")
    print(f"m2[0] = {m2[0]}")
    print(f"Esperado: m1 * m2 = {m1[0] * m2[0]}")

    # Criptografar
    ct1 = ciphertext_factory.encode_and_encrypt(m1, pk)
    ct2 = ciphertext_factory.encode_and_encrypt(m2, pk)

    print(f"\nCiphertexts criados:")
    print(f"  ct1: level={ct1.level}, scale={ct1.scale:.2e}")
    print(f"  ct2: level={ct2.level}, scale={ct2.scale:.2e}")

    # Verificar descriptografia individual
    dec1 = ciphertext_factory.decrypt_and_decode(ct1, sk, 1)
    dec2 = ciphertext_factory.decrypt_and_decode(ct2, sk, 1)

    print(f"\nDescriptografia individual:")
    print(f"  Dec(ct1)[0] = {dec1[0]:.6f} (erro: {abs(dec1[0] - m1[0]):.2e})")
    print(f"  Dec(ct2)[0] = {dec2[0]:.6f} (erro: {abs(dec2[0] - m2[0]):.2e})")

    # Multiplicação passo a passo
    print(f"\n--- MULTIPLICAÇÃO PASSO A PASSO ---")

    # Passo 1: Raw multiply
    print(f"1. Raw multiply...")
    ct_raw = CKKSCiphertext.raw_multiply_homomorphic(ct1, ct2)
    print(
        f"   Resultado: level={ct_raw.level}, scale={ct_raw.scale:.2e}, size={ct_raw.size}"
    )

    # Tentar descriptografar (vai dar errado, mas vamos ver)
    try:
        dec_raw = ciphertext_factory.decrypt_and_decode(ct_raw, sk, 1)
        print(f"   Dec(raw)[0] = {dec_raw[0]:.6f}")
    except Exception as e:
        print(f"   Erro na descriptografia raw (esperado): {str(e)[:50]}...")

    # Passo 2: Relinearização
    print(f"2. Relinearização...")
    ct_relin = CKKSCiphertext.relinearize(ct_raw, evk)
    print(
        f"   Resultado: level={ct_relin.level}, scale={ct_relin.scale:.2e}, size={ct_relin.size}"
    )

    # Descriptografar após relinearização
    dec_relin = ciphertext_factory.decrypt_and_decode(ct_relin, sk, 1)
    print(f"   Dec(relin)[0] = {dec_relin[0]:.6f}")
    print(f"   Escala esperada: {ct1.scale * ct2.scale:.2e}")
    print(
        f"   Fator de ajuste necessário: {(ct1.scale * ct2.scale) / crypto_params.SCALING_FACTOR:.2e}"
    )

    # Passo 3: Rescale
    print(f"3. Rescale...")
    ct_rescaled = CKKSCiphertext.rescale(ct_relin)
    print(
        f"   Resultado: level={ct_rescaled.level}, scale={ct_rescaled.scale:.2e}, size={ct_rescaled.size}"
    )

    # Descriptografar após rescale
    dec_rescaled = ciphertext_factory.decrypt_and_decode(ct_rescaled, sk, 1)
    print(f"   Dec(rescaled)[0] = {dec_rescaled[0]:.6f}")
    print(f"   Erro vs esperado: {abs(dec_rescaled[0] - (m1[0] * m2[0])):.2e}")

    # Teste com método completo
    print(f"\n--- MÉTODO COMPLETO ---")
    ct_complete = CKKSCiphertext.multiply_homomorphic(ct1, ct2, evk, auto_rescale=True)
    dec_complete = ciphertext_factory.decrypt_and_decode(ct_complete, sk, 1)
    print(f"Resultado completo: {dec_complete[0]:.6f}")
    print(f"Erro vs esperado: {abs(dec_complete[0] - (m1[0] * m2[0])):.2e}")

    # Teste com método sem rescale
    print(f"\n--- MÉTODO SEM RESCALE ---")
    ct_no_rescale = CKKSCiphertext.multiply_homomorphic(
        ct1, ct2, evk, auto_rescale=False
    )
    dec_no_rescale = ciphertext_factory.decrypt_and_decode(ct_no_rescale, sk, 1)
    print(f"Resultado sem rescale: {dec_no_rescale[0]:.6f}")
    print(f"Erro vs esperado: {abs(dec_no_rescale[0] - (m1[0] * m2[0])):.2e}")

    # Análise do problema de escala
    print(f"\n--- ANÁLISE DE ESCALA ---")
    expected_scale_after_mult = ct1.scale * ct2.scale
    actual_scale_after_relin = ct_relin.scale
    print(f"Escala esperada após mult: {expected_scale_after_mult:.2e}")
    print(f"Escala real após relin: {actual_scale_after_relin:.2e}")
    print(f"Diferença: {abs(expected_scale_after_mult - actual_scale_after_relin):.2e}")

    # Tentar ajustar manualmente a escala
    manual_adjustment = dec_relin[0] * (
        crypto_params.SCALING_FACTOR / expected_scale_after_mult
    )
    print(f"Ajuste manual: {manual_adjustment:.6f}")
    print(f"Erro com ajuste manual: {abs(manual_adjustment - (m1[0] * m2[0])):.2e}")


if __name__ == "__main__":
    debug_multiplication()
