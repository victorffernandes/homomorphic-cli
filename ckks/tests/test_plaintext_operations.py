"""
Tests for CKKS plaintext operations (add, sub, multiply, rescale, negate, multiply_by_const).

HEAAN-aligned operations. Validates encode → op → decode roundtrips.
"""

import numpy as np
import pytest

from ckks.constants import CKKSCryptographicParameters
from ckks.ckks_plaintext import CKKSPlaintext


class TestCKKSPlaintextOperations:
    """Tests for CKKSPlaintext operations"""

    PRECISION_TOLERANCE = 1e-1

    def setup_method(self):
        """Setup executed before each test.

        Uses custom config: modulus must fit in int64 (q < 2^63) for
        poly_ring_mod. logN=8, logQ=35, logp=15, total_levels=60 gives
        N=256, modulus up to 2^60 (fits int64), q > N*scale^2 for multiply.
        """
        self.crypto_params = CKKSCryptographicParameters(
            logN=8, logQ=35, logp=15, total_levels=60
        )

    def _num_slots(self):
        return self.crypto_params.POLYNOMIAL_DEGREE // 2

    def test_add(self):
        """from_vector m1, m2 → add → to_vector → assert close to m1 + m2"""
        n = self._num_slots()
        m1 = np.array([1.0, 2.0, 3.0, 4.0] + [0.0] * (n - 4))
        m2 = np.array([0.5, 1.5, 2.5, 3.5] + [0.0] * (n - 4))

        pt1 = CKKSPlaintext.from_vector(m1.tolist(), self.crypto_params)
        pt2 = CKKSPlaintext.from_vector(m2.tolist(), self.crypto_params)

        pt_sum = CKKSPlaintext.add(pt1, pt2)
        result = pt_sum.to_vector(q_mod=False)

        expected = m1 + m2
        np.testing.assert_allclose(
            result[: len(expected)],
            expected,
            rtol=self.PRECISION_TOLERANCE,
            atol=self.PRECISION_TOLERANCE,
        )

    def test_sub(self):
        """from_vector m1, m2 → sub → to_vector → assert close to m1 - m2"""
        n = self._num_slots()
        m1 = np.array([1.0, 2.0, 3.0, 4.0] + [0.0] * (n - 4))
        m2 = np.array([0.5, 1.5, 2.5, 3.5] + [0.0] * (n - 4))

        pt1 = CKKSPlaintext.from_vector(m1.tolist(), self.crypto_params)
        pt2 = CKKSPlaintext.from_vector(m2.tolist(), self.crypto_params)

        pt_diff = CKKSPlaintext.sub(pt1, pt2)
        result = pt_diff.to_vector(q_mod=False)

        expected = m1 - m2
        np.testing.assert_allclose(
            result[: len(expected)],
            expected,
            rtol=self.PRECISION_TOLERANCE,
            atol=self.PRECISION_TOLERANCE,
        )

    def test_multiply(self):
        """from_vector m1, m2 → multiply → rescale(logp) → to_vector → assert close to m1 * m2."""
        n = self._num_slots()
        m1 = np.array([1.0, 2.0, 3.0, 4.0] + [0.0] * (n - 4))
        m2 = np.array([0.5, 1.5, 2.5, 3.5] + [0.0] * (n - 4))

        pt1 = CKKSPlaintext.from_vector(m1.tolist(), self.crypto_params)
        pt2 = CKKSPlaintext.from_vector(m2.tolist(), self.crypto_params)

        pt_prod = CKKSPlaintext.multiply(pt1, pt2)
        bits_down = self.crypto_params.logp
        pt_rescaled = pt_prod.rescale(bits_down)
        result = pt_rescaled.to_vector(
            q_mod=True, q_mod_value=self.crypto_params.get_initial_modulus()
        )

        expected = m1 * m2
        # Special FFT (canonical embedding) enables correct slot-wise multiply
        np.testing.assert_allclose(
            result[: len(expected)],
            expected,
            rtol=self.PRECISION_TOLERANCE,
            atol=self.PRECISION_TOLERANCE,
        )

    def test_rescale(self):
        """from_vector m → multiply(m,m) → rescale(bits_down) → to_vector → assert close to m²."""
        n = self._num_slots()
        m = np.array([2.0, 3.0, 4.0, 5.0] + [0.0] * (n - 4))

        pt = CKKSPlaintext.from_vector(m.tolist(), self.crypto_params)
        pt_sq = CKKSPlaintext.multiply(pt, pt)
        bits_down = self.crypto_params.logp
        pt_rescaled = pt_sq.rescale(bits_down)
        result = pt_rescaled.to_vector(
            q_mod=True, q_mod_value=self.crypto_params.get_initial_modulus()
        )

        expected = m * m
        # Special FFT (canonical embedding) enables correct slot-wise multiply
        np.testing.assert_allclose(
            result[: len(expected)],
            expected,
            rtol=self.PRECISION_TOLERANCE,
            atol=self.PRECISION_TOLERANCE,
        )

    def test_negate(self):
        """from_vector m → negate → to_vector → assert close to -m"""
        n = self._num_slots()
        m = np.array([1.5, -2.3, 3.7, 0.0] + [0.0] * (n - 4))

        pt = CKKSPlaintext.from_vector(m.tolist(), self.crypto_params)
        pt_neg = pt.negate()
        result = pt_neg.to_vector(q_mod=False)

        expected = -m
        np.testing.assert_allclose(
            result[: len(expected)],
            expected,
            rtol=self.PRECISION_TOLERANCE,
            atol=self.PRECISION_TOLERANCE,
        )

    def test_multiply_by_const(self):
        """from_vector m → multiply_by_const(m, c, logp) → rescale → to_vector → assert close to c * m"""
        n = self._num_slots()
        m = np.array([1.0, 2.0, 3.0, 4.0] + [0.0] * (n - 4))
        c = 2.5

        pt = CKKSPlaintext.from_vector(m.tolist(), self.crypto_params)
        pt_scaled = CKKSPlaintext.multiply_by_const(pt, c, self.crypto_params.logp)
        pt_rescaled = pt_scaled.rescale(self.crypto_params.logp)
        result = pt_rescaled.to_vector(q_mod=False)

        expected = c * m
        np.testing.assert_allclose(
            result[: len(expected)],
            expected,
            rtol=self.PRECISION_TOLERANCE,
            atol=self.PRECISION_TOLERANCE,
        )

    def test_add_mismatched_scales_raises(self):
        """add with mismatched scales raises ValueError"""
        n = self._num_slots()
        m1 = np.array([1.0, 2.0] + [0.0] * (n - 2))
        m2 = np.array([0.5, 1.5] + [0.0] * (n - 2))

        pt1 = CKKSPlaintext.from_vector(m1.tolist(), self.crypto_params)
        pt2 = CKKSPlaintext.from_vector(
            m2.tolist(), self.crypto_params, scale=2 * self.crypto_params.SCALING_FACTOR
        )

        with pytest.raises(ValueError, match="same scale"):
            CKKSPlaintext.add(pt1, pt2)

    def test_sub_mismatched_scales_raises(self):
        """sub with mismatched scales raises ValueError"""
        n = self._num_slots()
        m1 = np.array([1.0, 2.0] + [0.0] * (n - 2))
        m2 = np.array([0.5, 1.5] + [0.0] * (n - 2))

        pt1 = CKKSPlaintext.from_vector(m1.tolist(), self.crypto_params)
        pt2 = CKKSPlaintext.from_vector(
            m2.tolist(), self.crypto_params, scale=2 * self.crypto_params.SCALING_FACTOR
        )

        with pytest.raises(ValueError, match="same scale"):
            CKKSPlaintext.sub(pt1, pt2)

    def test_from_vector_to_vector_roundtrip(self):
        """CKKSPlaintext.from_vector → to_vector preserves values"""
        original = [1.5, -2.3, 3.7, 0.0, 4.2]
        pt = CKKSPlaintext.from_vector(original, self.crypto_params)
        decoded = pt.to_vector(q_mod=False)
        np.testing.assert_allclose(
            decoded[: len(original)],
            original,
            rtol=self.PRECISION_TOLERANCE,
            atol=self.PRECISION_TOLERANCE,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
