"""Plaintext Householder QR decomposition (float matrix)."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


Matrix = List[List[float]]


def householder_qr(a: Sequence[Sequence[float]]) -> Tuple[Matrix, Matrix]:
	"""
	Compute QR using Householder reflections.

	Uses v = x + sign(x0) * ||x|| * e1 to avoid catastrophic cancellation,
	and applies reflectors as A <- A - 2 v (v^T A) without forming H.
	"""
	m = len(a)
	n = len(a[0]) if m else 0
	r: Matrix = [list(map(float, row)) for row in a]
	q: Matrix = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

	for k in range(min(m, n)):
		x = [r[i][k] for i in range(k, m)]
		norm_x = math.sqrt(sum(xi * xi for xi in x))

		if norm_x == 0.0:
			continue

		sgn = 1.0 if x[0] >= 0.0 else -1.0
		v = x[:]
		v[0] += sgn * norm_x
 
		norm_v = math.sqrt(sum(vi * vi for vi in v))
		if norm_v == 0.0:
			continue
		v = [vi / norm_v for vi in v]

		# Update R: R[k:, k:] -= 2 v (v^T R[k:, k:])
		for j in range(k, n):
			dot = sum(v[i] * r[k + i][j] for i in range(len(v)))
			for i in range(len(v)):
				r[k + i][j] -= 2.0 * v[i] * dot

    
		# Accumulate Q on the right: Q <- Q * H
		for row in range(m):
			dot = sum(v[i] * q[row][k + i] for i in range(len(v)))
			for i in range(len(v)):
				q[row][k + i] -= 2.0 * v[i] * dot

	return q, r


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
	rows, cols, inner = len(a), len(b[0]), len(b)
	return [[sum(a[i][t] * b[t][j] for t in range(inner)) for j in range(cols)] for i in range(rows)]


def fro_norm(a: Sequence[Sequence[float]]) -> float:
	return math.sqrt(sum(v * v for row in a for v in row))


def sub(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
	return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


if __name__ == "__main__":
	import time

	A = [
		[12.0, -51.0, 4.0],
		[6.0, 167.0, -68.0],
		[-4.0, 24.0, -41.0],
	]

	t0 = time.perf_counter()
	Q, R = householder_qr(A)
	elapsed = time.perf_counter() - t0

	A_recon = matmul(Q, R)
	err = fro_norm(sub(A, A_recon))
	norm_A = fro_norm(A)
	rel_err = err / norm_A if norm_A > 0 else 0.0

	print(f"3x3 plaintext: {elapsed:.4f}s")
	print(f"  ||A - QR||_F / ||A||_F = {rel_err:.2e}")
	print(f"  PASS: {rel_err < 1e-10}")

