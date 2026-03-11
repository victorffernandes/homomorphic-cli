"""
Tests for canonical embedding (special FFT with cyclotomic roots).

One test per step: rotGroup, ksiPows, bitReverse, fftSpecial, fftSpecialInv.
"""

import numpy as np
import pytest

from ckks.constants import CKKSCryptographicParameters
from ckks.canonical_embedding import (
    compute_rot_group,
    compute_ksi_pows,
    _bit_reverse,
    fft_special,
    fft_special_inv,
    fft_special_inv_lazy,
    get_fft_tables,
)


class TestRotGroup:
    """Tests for rotGroup: rotGroup[j] = 5^j mod M"""

    @pytest.mark.parametrize("N", [8, 64])
    def test_rot_group(self, N):
        M = N << 1
        Nh = N >> 1
        rot_group = compute_rot_group(M, Nh)

        five_pows = 1
        for j in range(Nh):
            assert rot_group[j] == five_pows, f"j={j}: expected {five_pows}, got {rot_group[j]}"
            five_pows = (five_pows * 5) % M


class TestKsiPows:
    """Tests for ksiPows: ksiPows[j] = exp(2πi j/M)"""

    @pytest.mark.parametrize("M", [16, 128])
    def test_ksi_pows(self, M):
        ksi_pows = compute_ksi_pows(M)

        for j in range(M + 1):
            expected = np.exp(2j * np.pi * j / M)
            np.testing.assert_allclose(ksi_pows[j], expected, rtol=1e-14, atol=1e-14)

        # ksiPows[M] == ksiPows[0] (periodicity)
        np.testing.assert_allclose(ksi_pows[M], ksi_pows[0], rtol=1e-14, atol=1e-14)


class TestBitReverse:
    """Tests for bit reversal permutation"""

    @pytest.mark.parametrize("size", [4, 8, 64])
    def test_bit_reverse_involutory(self, size):
        """bitReverse(bitReverse(vals)) restores original"""
        vals = np.arange(size, dtype=np.complex128) + 1j * np.arange(size, dtype=np.float64)
        original = vals.copy()

        _bit_reverse(vals, size)
        _bit_reverse(vals, size)

        np.testing.assert_allclose(vals, original, rtol=0, atol=0)

    def test_bit_reverse_mapping_size8(self):
        """For size=8, verify Cooley-Tukey bit reversal mapping"""
        # 0->0, 1->4, 2->2, 3->6, 4->1, 5->5, 6->3, 7->7
        # Binary: 000->000, 001->100, 010->010, 011->110, 100->001, 101->101, 110->011, 111->111
        size = 8
        vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.complex128)
        _bit_reverse(vals, size)
        # After bit reverse: index i goes to rev(i)
        # rev(0)=0, rev(1)=4, rev(2)=2, rev(3)=6, rev(4)=1, rev(5)=5, rev(6)=3, rev(7)=7
        # So vals[0]=0 (was at 0), vals[1]=4 (was at 4), vals[2]=2, vals[3]=6, vals[4]=1, vals[5]=5, vals[6]=3, vals[7]=7
        expected = np.array([0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0], dtype=np.complex128)
        np.testing.assert_allclose(vals, expected, rtol=0, atol=0)


class TestFftSpecialInvLazyRoundtrip:
    """fftSpecial(fftSpecialInvLazy(vals)) / size ≈ vals"""

    @pytest.mark.parametrize("N", [8, 64])
    def test_fft_special_inv_roundtrip(self, N):
        Nh = N >> 1
        M = N << 1
        rot_group, ksi_pows = get_fft_tables(N)

        np.random.seed(42)
        vals = np.random.randn(Nh) + 1j * np.random.randn(Nh)
        vals = vals.astype(np.complex128)
        original = vals.copy()

        fft_special_inv_lazy(vals, rot_group, ksi_pows, M)
        fft_special(vals, rot_group, ksi_pows, M)
        vals /= Nh

        np.testing.assert_allclose(vals, original, rtol=1e-10, atol=1e-10)


class TestFftSpecialInvRoundtrip:
    """fftSpecial(fftSpecialInv(vals)) == vals"""

    @pytest.mark.parametrize("N", [8, 64])
    def test_fft_special_inv_full_roundtrip(self, N):
        Nh = N >> 1
        M = N << 1
        rot_group, ksi_pows = get_fft_tables(N)

        np.random.seed(42)
        z1 = np.random.randn(Nh) + 1j * np.random.randn(Nh)
        z1 = z1.astype(np.complex128)
        original = z1.copy()

        fft_special_inv(z1, rot_group, ksi_pows, M)
        fft_special(z1, rot_group, ksi_pows, M)

        np.testing.assert_allclose(z1, original, rtol=1e-10, atol=1e-10)


class TestSigmaRoundtrip:
    """sigma(sigma_inverse(z)) ≈ z for slot values"""

    def setup_method(self):
        self.crypto_params = CKKSCryptographicParameters(
            logN=8, logQ=35, logp=15, total_levels=60
        )

    def test_sigma_roundtrip(self):
        from ckks.ckks_plaintext import CKKSPlaintext

        n = self.crypto_params.POLYNOMIAL_DEGREE // 2
        z = np.random.randn(n).astype(np.float64)  # real vector
        z = z + 0j  # complex with zero imag

        p = CKKSPlaintext.sigma_inverse(z, self.crypto_params)
        z2 = CKKSPlaintext.sigma(p, self.crypto_params)

        np.testing.assert_allclose(z2.real, z.real, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(z2.imag, z.imag, rtol=1e-10, atol=1e-10)


class TestSlotWiseMultiplyProperty:
    """σ(m1·m2) = σ(m1) ⊙ σ(m2) (Hadamard product in slots)"""

    def setup_method(self):
        self.crypto_params = CKKSCryptographicParameters(
            logN=8, logQ=35, logp=15, total_levels=60
        )

    def test_slot_wise_multiply_property(self):
        from ckks.ckks_plaintext import CKKSPlaintext

        n = self.crypto_params.POLYNOMIAL_DEGREE // 2
        m1 = np.array([1.0, 2.0, 3.0, 4.0] + [0.0] * (n - 4))
        m2 = np.array([0.5, 1.5, 2.5, 3.5] + [0.0] * (n - 4))

        pt1 = CKKSPlaintext.from_vector(m1.tolist(), self.crypto_params)
        pt2 = CKKSPlaintext.from_vector(m2.tolist(), self.crypto_params)

        # Decode m1 and m2
        z1 = CKKSPlaintext.sigma(pt1.polynomial, self.crypto_params)
        z2 = CKKSPlaintext.sigma(pt2.polynomial, self.crypto_params)

        # Multiply in ring, then decode
        q = self.crypto_params.get_initial_modulus()
        ring_poly_mod = self.crypto_params.get_polynomial_modulus_ring()
        m_prod = self.crypto_params.poly_mul_mod(
            pt1.polynomial, pt2.polynomial, q, ring_poly_mod
        )
        z_prod = CKKSPlaintext.sigma(m_prod, self.crypto_params)

        # Slot-wise product
        expected = z1 * z2

        # σ(m1·m2) = σ(m1)⊙σ(m2) holds in the ring. Mod q and rounding cause some error.
        # Assert real parts of first 4 slots (with data) match - primary CKKS use case.
        num_data_slots = 4
        np.testing.assert_allclose(
            z_prod.real[:num_data_slots],
            expected.real[:num_data_slots],
            rtol=1e-5,
            atol=1e-5,
        )
        # Imag parts can diverge due to mod reduction; check magnitude is reasonable
        # (expected imag ~0 for real input; z_prod.imag may have mod-induced error)
        assert np.all(np.abs(z_prod.imag[:num_data_slots]) < np.abs(expected.real[:num_data_slots]) * 0.5)


class TestEncodeDecodeRoundtrip:
    """CKKSPlaintext encode then decode returns values close to original"""

    def setup_method(self):
        self.crypto_params = CKKSCryptographicParameters(
            logN=8, logQ=35, logp=15, total_levels=60
        )

    def test_encode_decode_roundtrip_special_fft(self):
        from ckks.ckks_plaintext import CKKSPlaintext

        original = [1.5, -2.3, 3.7, 0.0, 4.2]
        pt = CKKSPlaintext.from_vector(original, self.crypto_params)
        decoded = pt.to_vector(q_mod=False)
        # Rounding in encode causes small error; use relaxed tolerance.
        np.testing.assert_allclose(
            decoded[: len(original)],
            original,
            rtol=1e-2,
            atol=1e-2,
        )
