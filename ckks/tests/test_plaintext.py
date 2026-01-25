import numpy as np
from numpy.polynomial import Polynomial

from ckks.ciphertext_factory import create_ckks_factory


class TestCKKSPlaintextEncoding:

    def test_encode_decode_cycle(self):
        factory = create_ckks_factory()

        original_data = [1.5, -2.3, 3.7, 0.0]

        encoded_poly = factory.ckks_encode_real(original_data)
        assert isinstance(encoded_poly, Polynomial)

        decoded_data = factory.ckks_decode_real(encoded_poly, q_mod=False)

        assert len(decoded_data) >= len(
            original_data
        ), f"Decoded data length {len(decoded_data)} < original {len(original_data)}"

        np.testing.assert_allclose(
            decoded_data[: len(original_data)],
            original_data,
            rtol=1e-1,
            atol=1e-1,
        )

    def test_encode_with_custom_params(self):
        factory = create_ckks_factory()

        original_data = [2.5, -1.8]
        custom_scale = 1000.0

        assert len(original_data) > 0
        assert custom_scale > 0

        encoded_poly = factory.ckks_encode_real(original_data, delta_scale=custom_scale)

        assert isinstance(encoded_poly, Polynomial)
        assert len(encoded_poly.coef) > 0

        decoded_data = factory.ckks_decode_real(
            encoded_poly, delta_scale=custom_scale, q_mod=False
        )

        assert isinstance(decoded_data, np.ndarray)
        assert len(decoded_data) >= len(original_data)

        np.testing.assert_allclose(
            decoded_data[: len(original_data)],
            original_data,
            rtol=5e-2,
        )

    def test_empty_vector_encoding(self):
        factory = create_ckks_factory()

        empty_data = []
        encoded_poly = factory.ckks_encode_real(empty_data)
        assert isinstance(encoded_poly, Polynomial)

        decoded_data = factory.ckks_decode_real(encoded_poly, q_mod=False)
        assert len(decoded_data) > 0

        max_slots = factory.crypto_params.POLYNOMIAL_DEGREE // 2
        expected_zeros = [0] * min(max_slots, len(decoded_data))
        np.testing.assert_allclose(
            decoded_data[: len(expected_zeros)], expected_zeros, atol=1e-1
        )

    def test_single_element_encoding(self):
        factory = create_ckks_factory()

        single_value = [42.7]
        encoded_poly = factory.ckks_encode_real(single_value)
        decoded_data = factory.ckks_decode_real(encoded_poly, q_mod=False)

        np.testing.assert_allclose([decoded_data[0]], single_value, rtol=1e-2)

    def test_large_vector_encoding(self):
        factory = create_ckks_factory()

        max_elements = factory.crypto_params.POLYNOMIAL_DEGREE // 2
        large_data = np.random.uniform(-10, 10, max_elements // 2).tolist()

        assert len(large_data) > 0
        assert all(isinstance(x, float) for x in large_data)

        encoded_poly = factory.ckks_encode_real(large_data)
        assert isinstance(encoded_poly, Polynomial)
        assert len(encoded_poly.coef) > 0

        decoded_data = factory.ckks_decode_real(encoded_poly, q_mod=False)
        assert isinstance(decoded_data, np.ndarray)
        assert len(decoded_data) >= len(large_data)

        np.testing.assert_allclose(
            decoded_data[: len(large_data)],
            large_data,
            rtol=1e-1,
        )
