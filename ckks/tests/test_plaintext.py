import numpy as np
from numpy.polynomial import Polynomial
import pytest

from ckks.constants import CKKSCryptographicParameters
from ckks.ckks_plaintext import CKKSPlaintext


class TestCKKSPlaintext:
    """Tests for CKKSPlaintext encode and decode methods"""

    PRECISION_TOLERANCE = 1e-1
    CUSTOM_SCALE_TOLERANCE = 5e-2

    def setup_method(self):
        """Setup executed before each test"""
        self.crypto_params = CKKSCryptographicParameters()

    def test_encode_decode_roundtrip(self):
        """Test basic encode → decode roundtrip preserves values"""
        original_data = [1.5, -2.3, 3.7, 0.0]

        encoded_poly = CKKSPlaintext.encode(original_data, self.crypto_params)
        assert isinstance(encoded_poly, Polynomial)

        decoded_data = CKKSPlaintext.decode(
            encoded_poly, self.crypto_params, q_mod=False
        )

        assert len(decoded_data) >= len(original_data)
        np.testing.assert_allclose(
            decoded_data[: len(original_data)],
            original_data,
            rtol=self.PRECISION_TOLERANCE,
            atol=self.PRECISION_TOLERANCE,
        )

    def test_encode_decode_with_custom_scale(self):
        """Test encode/decode with explicit custom scale parameter"""
        original_data = [2.5, -1.8, 4.0]
        custom_scale = 1000.0

        encoded_poly = CKKSPlaintext.encode(
            original_data, self.crypto_params, scale=custom_scale
        )
        assert isinstance(encoded_poly, Polynomial)
        assert len(encoded_poly.coef) > 0

        decoded_data = CKKSPlaintext.decode(
            encoded_poly, self.crypto_params, scale=custom_scale, q_mod=False
        )

        assert isinstance(decoded_data, np.ndarray)
        assert len(decoded_data) >= len(original_data)
        np.testing.assert_allclose(
            decoded_data[: len(original_data)],
            original_data,
            rtol=self.CUSTOM_SCALE_TOLERANCE,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
