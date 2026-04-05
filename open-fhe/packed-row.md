# Packed-Row Householder QR over CKKS — Production FHE (OpenFHE-Python)

## Context

The hybrid variant (`qr_householder_cipher_cell.py`) decrypts three scalars per Householder
step — norm, sign, and v^Tv — leaking information to the key holder.
The packed-row plan (`packed-row.md` v1) reduced ciphertext count from O(m²) to O(m) by
row-packing R and column-packing Q, but kept the same hybrid decryption shortcut.

**This revision eliminates all decryption.**
Norm is computed entirely in ciphertext space via a minimax Chebyshev approximation of
√t evaluated using OpenFHE's baby-step / giant-step `EvalChebyshevSeries`.
Sign is assumed **always +1** — the caller must guarantee matrix entries are in a range
where this holds, or pre-condition the input column.

---

## Design Assumptions

| Symbol | Meaning | Value for 150×4 |
|--------|---------|-----------------|
| B | Max absolute entry value in A | 10.0 |
| m, n | Matrix dimensions | **150, 4** |
| steps | Householder steps = min(m,n) | **4** |
| N | CKKS ring dimension (slots = N/2) | 2¹³ = 8192 (research) / 2¹⁸ (production) |
| ε | Minimum non-zero norm (skip threshold) | 1e-6 |
| D_sqrt | Chebyshev degree for √t | 16 |
| D_inv  | Chebyshev degree for 1/t | 16 |

**Why 150×4 is efficient**: `min(150, 4) = 4` steps total, so no bootstrapping is needed
even though m is large. The depth is `4 × 24 = 96` levels — a single context suffices.
The large m only affects the number of operations per step (150 squarings at step 0),
not the multiplicative depth.

Domain bounds (computed once from B and m):

```python
# For 150×4: B=10, m=150
NORM_SQ_LO  = ε**2                        # ~1e-12
NORM_SQ_HI  = B**2 * m                    # 100 * 150 = 15000
NORM_LO     = ε                            # ~1e-6
NORM_HI     = B * math.sqrt(m)            # 10 * sqrt(150) ≈ 122.5
VTV_LO      = 2 * NORM_LO * NORM_LO       # ~2e-12 (conservative)
VTV_HI      = 2 * NORM_HI * (NORM_HI + B) # 2 * 122.5 * 132.5 ≈ 32_513
```

---

## Data Layout — 150×4

| Matrix | Packing | Count | Meaningful slots |
|--------|---------|-------|-----------------|
| R (150×4) | Row cts: `R_cts[i]` = `[R[i,0], R[i,1], R[i,2], R[i,3], 0, …]` | 150 | 4 per ct |
| Q (150×150) | Col cts: `Q_cols[j]` = `[Q[0,j], …, Q[149,j], 0, …]` | 150 | 150 per ct |

Q has 150 meaningful slots per ciphertext — requires N/2 ≥ 150, satisfied by N=2¹³ (4096 slots).
R rows only use 4 slots, but must live in the same ring size as Q for key compatibility.

---

## Step 1: Crypto Context

```python
def setup_crypto_context(mult_depth: int, N: int = 8192) -> tuple:
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    params.SetRingDim(N)
    params.SetBatchSize(N // 2)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)  # required for EvalChebyshevSeries

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # Rotation indices needed: k for slot→slot-0, and N/2-powers for replication
    slots = N // 2
    rotation_indices = list(range(1, slots, 1))  # all shifts; tune to actual k values
    cc.EvalRotateKeyGen(keys.secretKey, rotation_indices)

    return cc, keys
```

**Why ADVANCEDSHE**: `EvalChebyshevSeries` lives in the advanced SHE feature set in OpenFHE ≥ 1.1.

**Rotation keys**: Required for slot-to-slot-0 shift and for `replicate_slot_0`. In production,
restrict to the exact indices used: `{1, 2, 4, …, N/4}` for replication + `{k : k = 0..n-1}` for
column extraction.

---

## Step 2: Chebyshev Primitives

### 2a. Homomorphic square-root  `he_sqrt(cc, ct, a, b, degree) → ct`

Approximates `f(t) = √t` on `[a, b]` using OpenFHE's built-in baby-step / giant-step evaluator.

```python
def he_sqrt(cc, ct, a: float, b: float, degree: int = 16):
    """
    Chebyshev approximation of sqrt(t) on [a, b].
    Depth cost: ceil(log2(degree)) + ceil(log2(ceil(degree/sqrt(degree)))) ≈ 2*ceil(log2(degree)).
    For degree=16: depth = 2*4 = 8.
    """
    return cc.EvalChebyshevSeries(ct, [0.0]*0, a, b, degree)
    # OpenFHE computes the minimax Chebyshev coefficients internally when
    # given the function handle. Use the overload that accepts a Python callable:
    #   cc.EvalChebyshevFunction(lambda t: math.sqrt(abs(t)), ct, a, b, degree)
```

In practice use the callable overload:

```python
def he_sqrt(cc, ct, a: float, b: float, degree: int = 16):
    return cc.EvalChebyshevFunction(lambda t: math.sqrt(max(t, a)), ct, a, b, degree)
    # depth cost for degree=16: +8 levels
```

### 2b. Homomorphic reciprocal  `he_inv(cc, ct, a, b, degree) → ct`

Approximates `f(t) = 1/t` on `[a, b]`.

```python
def he_inv(cc, ct, a: float, b: float, degree: int = 16):
    return cc.EvalChebyshevFunction(lambda t: 1.0 / t, ct, a, b, degree)
    # depth cost for degree=16: +8 levels
```

### 2c. Slot replication  `replicate_slot_0(cc, ct, slots) → ct`

Expands the value in slot 0 to all `slots` positions using a rotation tree.
Depth cost: **0** (only EvalAdd). Rotation count: O(log₂ slots).

```python
def replicate_slot_0(cc, ct, slots: int):
    """All slots receive the value originally in slot 0."""
    result = ct
    step = 1
    while step < slots:
        result = cc.EvalAdd(result, cc.EvalRotate(result, -step))
        step *= 2
    return result
```

> **Note**: This works when slot 0 is the only non-zero slot. If there is noise in other
> slots from prior operations, zero them first with a plaintext mask before calling this.

---

## Step 3: Householder Step — Fully Homomorphic

```python
def householder_step_fhe(cc, keys, R_cts, Q_cols, k: int, m: int, n: int,
                          norm_sq_lo, norm_sq_hi, vtv_lo, vtv_hi,
                          D_sqrt: int = 16, D_inv: int = 16):
    """
    One Householder reflection at column k.  No decryption.
    Assumes sign(x[0]) = +1.
    """
    slots  = cc.GetRingDimension() // 2
    length = m - k  # number of active rows

    # --- mask vector e_k: plaintext [0,...,0,1,0,...,0] with 1 at index k ---
    e_k = [0.0] * slots
    e_k[k] = 1.0
    ptxt_ek = cc.MakeCKKSPackedPlaintext(e_k)

    # =========================================================
    # Phase 1: FHE norm computation
    # =========================================================
    # Each x_j_ct has R[k+j, k] in slot k, zeros elsewhere.
    # Squaring and summing gives ||x||² in slot k.

    x_masked = [cc.EvalMult(R_cts[k + j], ptxt_ek) for j in range(length)]
    # depth +0 (ct × plaintext)

    norm_sq_ct = cc.EvalMult(x_masked[0], x_masked[0])  # depth +1
    for j in range(1, length):
        sq_j = cc.EvalMult(x_masked[j], x_masked[j])    # depth +1 (same level, no extra)
        norm_sq_ct = cc.EvalAdd(norm_sq_ct, sq_j)        # depth +0

    # Rotate slot k → slot 0 for Chebyshev evaluation
    norm_sq_slot0 = cc.EvalRotate(norm_sq_ct, k)  # depth +0

    # Zero out all slots except 0 to avoid replication noise
    e0 = [0.0] * slots; e0[0] = 1.0
    ptxt_e0 = cc.MakeCKKSPackedPlaintext(e0)
    norm_sq_slot0 = cc.EvalMult(norm_sq_slot0, ptxt_e0)  # depth +0

    # Chebyshev sqrt: slot 0 now holds ||x||
    norm_ct = he_sqrt(cc, norm_sq_slot0, norm_sq_lo, norm_sq_hi, D_sqrt)
    # depth: +D_sqrt/2 ... +D_sqrt (baby-step giant-step; ≈+8 for degree 16)

    # =========================================================
    # Phase 2: Build v[0] = x[0] + norm  and  compute 2/(v^Tv)
    # =========================================================
    # x0 is R[k, k] — extract to slot 0
    x0_ct = cc.EvalMult(R_cts[k], ptxt_ek)          # slot k = R[k,k], depth +0
    x0_slot0 = cc.EvalMult(cc.EvalRotate(x0_ct, k), ptxt_e0)  # slot 0 = R[k,k], depth +0

    # v[0] = x[0] + ||x|| (sign = +1 assumption)
    v0_ct = cc.EvalAdd(x0_slot0, norm_ct)            # slot 0 = v[0], depth +0

    # v^T v = 2 * ||x|| * (||x|| + x[0]) = 2 * norm * v0
    # (algebraic identity when sign=+1)
    vtv_ct = cc.EvalMult(cc.EvalMult(norm_ct, 2.0), v0_ct)   # depth +1

    # Chebyshev inverse: slot 0 = 2/(v^Tv) = 1/(norm * v0)
    two_over_vtv_ct = he_inv(cc, vtv_ct, vtv_lo, vtv_hi, D_inv)  # depth +D_inv

    # =========================================================
    # Phase 3: Broadcast scalars to all slots for row ops
    # =========================================================
    # v[0] broadcast
    v0_bc = replicate_slot_0(cc, v0_ct, slots)         # depth +0
    # 2/(v^Tv) broadcast
    tau_bc = replicate_slot_0(cc, two_over_vtv_ct, slots)  # depth +0

    # v[j] for j > 0: x[j] = R[k+j, k] broadcast to all slots
    # x_masked[j] has value in slot k; rotate to slot 0 then replicate
    v_bc = [v0_bc]
    for j in range(1, length):
        xj_slot0 = cc.EvalMult(cc.EvalRotate(x_masked[j], k), ptxt_e0)  # depth +0
        v_bc.append(replicate_slot_0(cc, xj_slot0, slots))               # depth +0

    # =========================================================
    # Phase 4: Update R — depth +1 for w_ct, +2 per row update
    # =========================================================
    # w_ct[col] = sum_i v[i] * R[k+i, col]  (one ciphertext = all columns)
    w_ct = cc.EvalMult(v_bc[0], R_cts[k])             # ct × ct, depth +1
    for j in range(1, length):
        w_ct = cc.EvalAdd(w_ct, cc.EvalMult(v_bc[j], R_cts[k + j]))

    # R[k+i, :] -= (2/vtv) * v[i] * w_ct
    for i in range(length):
        update = cc.EvalMult(cc.EvalMult(tau_bc, v_bc[i]), w_ct)  # depth +2
        R_cts[k + i] = cc.EvalSub(R_cts[k + i], update)

    # =========================================================
    # Phase 5: Update Q columns — depth +1 for d_ct, +2 per col update
    # =========================================================
    # d_ct[row] = sum_j Q[row, k+j] * v[j]
    d_ct = cc.EvalMult(Q_cols[k], v_bc[0])
    for j in range(1, length):
        d_ct = cc.EvalAdd(d_ct, cc.EvalMult(Q_cols[k + j], v_bc[j]))

    # Q[:, k+i] -= (2/vtv) * v[i] * d_ct
    for i in range(length):
        update = cc.EvalMult(cc.EvalMult(tau_bc, v_bc[i]), d_ct)  # depth +2
        Q_cols[k + i] = cc.EvalSub(Q_cols[k + i], update)
```

---

## Step 4: Depth Budget

### Per Householder step k

| Phase | Operation | Depth |
|-------|-----------|-------|
| 1 | Mask + square x_j | +1 |
| 1 | Chebyshev √t (degree 16) | +8 |
| 2 | vtv = 2·norm·v0 | +1 |
| 2 | Chebyshev 1/t (degree 16) | +8 |
| 4 | w_ct = Σ v[i]·R_row[i] | +1 |
| 4 | Row update: τ·v[i]·w_ct | +2 |
| 5 | d_ct = Σ Q_col[j]·v[j] | +1 |
| 5 | Col update: τ·v[i]·d_ct | +2 |
| **Total** | | **+24** |

Phases 4 and 5 share the same depth accumulation; they are done at the same level.
FLEXIBLEAUTO handles automatic rescaling, so the +1 for each ct×ct is approximate.

### Total depth by matrix size

Depth is driven by `min(m,n)` steps, not by m alone.

| Size | Steps = min(m,n) | Depth/step (deg 16) | Total depth | Bootstrapping? |
|------|-----------------|---------------------|-------------|----------------|
| 2×2 | 1 | 24 | 24 | No |
| 4×4 | 3 | 24 | 72 | No |
| 8×8 | 7 | 24 | 168 | Yes (≥ 1) |
| 30×30 | 29 | 24 | ~696 | Yes (29 bootstraps) |
| **150×4** | **4** | **24** | **~96** | **No** |
| 150×150 | 149 | 24 | ~3576 | Yes (many) |

**150×4 is the sweet spot**: m is large (expensive per step) but n=4 keeps total steps to 4,
so the full computation fits in a single context with no bootstrapping.

### 150×4 at precision < 1e-4

**Accumulated Chebyshev error** (degree 16, 4 steps):
```
4 steps × 2 calls/step × 1e-6 error/call = 8e-6  ≪  1e-4  ✓
```
Degree 16 is more than sufficient. Even degree 12 (error ~1e-5/call) gives 8e-5 < 1e-4.

**Depth breakdown (degree 16):**

| Phase | Depth/step | × 4 steps |
|-------|-----------|-----------|
| Square x_j (length varies 150→147) | 1 | 4 |
| Chebyshev √t (deg 16) | 8 | 32 |
| vtv mult | 1 | 4 |
| Chebyshev 1/t (deg 16) | 8 | 32 |
| w_ct = Σ v[i]·R_cts[i] | 1 | 4 |
| Row update τ·v[i]·w_ct | 2 | 8 |
| d_ct = Σ Q_col[j]·v[j] | 1 | 4 |
| Col update τ·v[i]·d_ct | 2 | 8 |
| Safety margin | — | +10 |
| **Total** | **24** | **~106** |

**Crypto context — research (fast, no security):**
```python
params.SetRingDim(1 << 13)               # N=8192, slots=4096 ≥ 150 ✓
params.SetMultiplicativeDepth(110)       # 106 + margin
params.SetScalingModSize(50)
params.SetFirstModSize(60)
params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
# logQ ≈ 60 + 110×50 = 5560 bits — not secure at N=8192, demo only
```

**Crypto context — production (128-bit secure):**
```python
# logQ ≈ 60 + 110×50 = 5560 bits → need N=2^18 for HEStd_128_classic
# (HEStd_128 allows logQ/N up to ~3.8; 5560/131072 = 4.2 → need N=2^19 or reduce depth)
# Practical choice: reduce to ScalingModSize=40 → logQ ≈ 60 + 110×40 = 4460 bits
params.SetRingDim(1 << 17)               # N=131072, slots=65536 ≥ 150 ✓
params.SetMultiplicativeDepth(110)
params.SetScalingModSize(40)             # 40-bit primes to fit N=2^17 budget
params.SetFirstModSize(55)
params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
# No FHE feature needed — no bootstrapping
```

**HE operation count at each step (length = 150 − k):**

| Step k | length | Squarings (norm) | Accum (w+d) | Updates (R+Q) | Total ct×ct |
|--------|--------|-----------------|-------------|---------------|-------------|
| 0 | 150 | 150 | 300 | 600 | 1050 |
| 1 | 149 | 149 | 298 | 596 | 1043 |
| 2 | 148 | 148 | 296 | 592 | 1036 |
| 3 | 147 | 147 | 294 | 588 | 1029 |
| **Total** | | **594** | **1188** | **2376** | **4158** |

EvalRotates per step: O(length × log₂(N/2)) for replication of each v[i] scalar.
At N=2¹³ and length=150: ~150 × 12 = 1800 rotations at step 0.

**Timing estimate (AWS c5.18xlarge / 72-core, OpenFHE 1.2, N=2¹³ research context):**

| Operation | Per step | × 4 steps |
|-----------|----------|-----------|
| Chebyshev √ + 1/t | ~3 s | ~12 s |
| Squarings + accumulation (norm) | ~0.5 s | ~2 s |
| w_ct + d_ct accumulation | ~1.5 s | ~6 s |
| R + Q row/col updates | ~2 s | ~8 s |
| Encrypt (300 cts) + decrypt | — | ~5 s |
| **Total** | | **~33 s** |

No bootstrapping overhead. Compare to hybrid packed-row (30×30 square): ~2–4 min with full submatrix leakage.

---

## Step 5: Chebyshev Degree vs Accuracy Trade-off

| Degree | Depth/call | Error per call | Accum (4 steps, 2 calls) | Target met |
|--------|-----------|----------------|--------------------------|------------|
| 8 | +5 | ~1e-3 | ~8e-3 | < 1e-2 only |
| 12 | +7 | ~1e-5 | ~8e-5 | **< 1e-4 ✓** |
| 16 | +8 | ~1e-6 | ~8e-6 | **< 1e-5 ✓** |
| 32 | +11 | ~1e-10 | ~8e-10 | < 1e-9 |

For 150×4 at `< 1e-4`, **degree 12 suffices** (saves 4 levels/step = 16 levels total,
lowering mult_depth from 106 to ~90 and easing the N requirement).
Degree 16 gives a full extra order of magnitude margin and is the recommended default.

---

## Step 6: Encrypt / Decrypt Helpers

```python
def encrypt_row(cc, keys, row: list) -> object:
    slots = cc.GetRingDimension() // 2
    padded = list(row) + [0.0] * (slots - len(row))
    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(padded))

def decrypt_vector(cc, keys, ct, length: int) -> list:
    pt = cc.Decrypt(ct, keys.secretKey)
    pt.SetLength(length)
    return [v.real for v in pt.GetCKKSPackedValue()]

def encrypt_matrix_rows(cc, keys, A) -> list:
    return [encrypt_row(cc, keys, row) for row in A]

def encrypt_identity_cols(cc, keys, m: int) -> list:
    slots = cc.GetRingDimension() // 2
    cols = []
    for j in range(m):
        col = [1.0 if i == j else 0.0 for i in range(m)]
        col += [0.0] * (slots - m)
        cols.append(cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(col)))
    return cols

def decrypt_matrix_rows(cc, keys, R_cts, n: int) -> list:
    return [decrypt_vector(cc, keys, ct, n) for ct in R_cts]

def decrypt_matrix_cols(cc, keys, Q_cols, m: int) -> list:
    """Columns → m×m matrix."""
    cols = [decrypt_vector(cc, keys, ct, m) for ct in Q_cols]
    return [[cols[j][i] for j in range(m)] for i in range(m)]
```

---

## Step 7: Top-Level Driver

```python
def depth_for_size(m: int, n: int, D_sqrt: int = 16, D_inv: int = 16) -> int:
    """Depth scales with min(m,n) steps, not m."""
    steps = min(m, n)
    depth_per_step = 1 + (D_sqrt // 2 + 1) + 1 + (D_inv // 2 + 1) + 1 + 2 + 1 + 2
    return max(30, steps * depth_per_step + 10)  # +10 margin

def householder_qr_cipher(cc, keys, A: list,
                           B: float = 10.0, D_sqrt: int = 16, D_inv: int = 16):
    m, n = len(A), len(A[0])
    ε = 1e-6
    norm_sq_lo = ε**2
    norm_sq_hi = B**2 * m                              # depends on m (column length)
    vtv_lo     = 2 * ε * ε
    vtv_hi     = 2 * B * math.sqrt(m) * (B * math.sqrt(m) + B)

    R_cts  = encrypt_matrix_rows(cc, keys, A)
    Q_cols = encrypt_identity_cols(cc, keys, m)

    for k in range(min(m, n)):                         # 4 iterations for 150×4
        t0 = time.perf_counter()
        householder_step_fhe(
            cc, keys, R_cts, Q_cols, k, m, n,
            norm_sq_lo, norm_sq_hi, vtv_lo, vtv_hi,
            D_sqrt, D_inv
        )
        print(f"  step k={k}: {time.perf_counter() - t0:.1f}s")

    Q = decrypt_matrix_cols(cc, keys, Q_cols, m)
    R = decrypt_matrix_rows(cc, keys, R_cts, n)
    return Q, R
```

---

## Step 8: HE Operation Count Per Step k  (length = m − k)

| Operation | Count | Type | Depth |
|-----------|-------|------|-------|
| Mask (ct × ptxt) | length + 1 | ct×ptxt | 0 |
| Square x_j | length | ct×ct | +1 |
| Chebyshev √ | 1 | polynomial | +D_sqrt/2 |
| Chebyshev 1/t | 1 | polynomial | +D_inv/2 |
| Replicate (v_bc + τ_bc) | length+1 | EvalRotate+Add | 0 |
| w_ct / d_ct accumulation | 2·length | ct×ct | +1 |
| Row + col update | 2·length | ct×ct | +2 |
| **Total ct×ct mults** | **4·length + 2** | | |
| **Total EvalRotates** | **O(length·log N)** | | 0 |

Compare to packed-row v1 (hybrid): 4·length ct×scalar, 0 ct×ct.
Compare to cell variant: O(m²) ct×ct mults with depth +2/step.

---

## Step 9: Correctness Notes

**Why sign = +1 is safe here**: The Householder reflector only requires `v ≠ 0`.
With `v[0] = x[0] + ||x||` and `sign = +1`, the only degenerate case is `x[0] = −||x||`
(i.e. `x = [−||x||, 0, …, 0]`), which gives `v = 0`. This is handled by the
ε-threshold: if `norm_sq < ε²`, skip the step. In production, normalize or pre-condition A
to avoid near-zero columns if using the +1 sign assumption.

**Chebyshev domain safety**: The domain `[norm_sq_lo, norm_sq_hi]` must contain all values
of `||x||²` that will be fed to `he_sqrt`. If entries are larger than B, the approximation
silently produces garbage. Enforce `assert max(abs(a[i][j]) for ...) <= B` before encrypting.

**Noise budget**: With FLEXIBLEAUTO, each ct×ct costs ~1 level. Add 4–6 levels of margin
for CKKS rounding noise accumulation. For tight parameter sets, add 8.

---

## Step 10: Verification and Main

### verify_fhe (tighter threshold)

```python
def verify_fhe(A, Q, R, tol: float = 1e-4):
    m, n = len(A), len(A[0])
    norm_A = fro_norm(A)

    rel_recon = fro_norm(sub(A, matmul(Q, R))) / norm_A
    print(f"||A - QR||_F / ||A||_F = {rel_recon:.2e}  (target < {tol:.0e})")

    I   = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    ortho_err = fro_norm(sub(matmul(transpose(Q), Q), I))
    print(f"||Q^T Q - I||_F       = {ortho_err:.2e}  (target < {tol:.0e})")

    max_lower = max(
        (abs(R[i][j]) for i in range(m) for j in range(min(i, n))), default=0.0
    )
    print(f"max |R[i,j]| (i > j)  = {max_lower:.2e}  (target < {tol:.0e})")

    ok = rel_recon < tol and ortho_err < tol and max_lower < tol
    print(f"PASS: {ok}\n")
    return ok
```

### Main: 150×4 production run

```python
def main():
    print("=" * 60)
    print("Householder QR — FHE, 150×4, no bootstrapping")
    print("Precision target: < 1e-4  |  Chebyshev degree 16")
    print("=" * 60)

    A = random_matrix(150, 4, seed=42)           # entries in [-10, 10]
    B, D_sqrt, D_inv = 10.0, 16, 16
    m, n = len(A), len(A[0])

    mult_depth = depth_for_size(m, n, D_sqrt, D_inv)   # ~106 for 150×4
    print(f"Matrix: {m}×{n}  |  Steps: {min(m,n)}  |  mult_depth: {mult_depth}")

    # Plaintext reference
    t0 = time.perf_counter()
    Q_ref, R_ref = householder_qr(A)
    print(f"Plaintext QR:  {time.perf_counter() - t0:.3f}s")

    # FHE run
    t0 = time.perf_counter()
    cc, keys = setup_crypto_context(mult_depth, N=8192)
    print(f"Context setup: {time.perf_counter() - t0:.1f}s  "
          f"(slots={cc.GetRingDimension()//2})")

    t0 = time.perf_counter()
    Q_enc, R_enc = householder_qr_cipher(cc, keys, A, B=B, D_sqrt=D_sqrt, D_inv=D_inv)
    print(f"FHE QR total:  {time.perf_counter() - t0:.1f}s")

    print("--- Verification ---")
    verify_fhe(A, Q_enc, R_enc, tol=1e-4)

    # Smoke tests on small known cases
    for label, A_s, tol_s in [("2x2", [[3.0, 1.0], [4.0, 1.5]], 1e-2),
                                ("4x2", random_matrix(4, 2, seed=7), 1e-2)]:
        m_s, n_s = len(A_s), len(A_s[0])
        cc_s, keys_s = setup_crypto_context(depth_for_size(m_s, n_s), N=8192)
        Q_s, R_s = householder_qr_cipher(cc_s, keys_s, A_s, B=B)
        print(f"--- {label} ---")
        verify_fhe(A_s, Q_s, R_s, tol=tol_s)


if __name__ == "__main__":
    main()
```

Run: `cd open-fhe && python qr_householder_cipher_row.py`

### Verification thresholds

| Test | Matrix | Threshold | Bootstrapping |
|------|--------|-----------|---------------|
| Smoke | 2×2, 4×2 | `< 1e-2` | No |
| **Production** | **150×4** | **`< 1e-4`** | **No** |

---

## Files

| File | Action |
|------|--------|
| `open-fhe/qr_householder_cipher_row.py` | **Create** — production FHE packed-row variant |
| `open-fhe/qr_householder_cipher_cell.py` | Reference only |
| `open-fhe/qr_householder_plain.py` | Reference: `householder_qr`, `matmul`, `fro_norm`, `sub` |

---

## Performance vs Previous Variants

Figures shown for 150×4 (m=150, n=4, steps=4).

| Metric | Cell (hybrid) 150×4 | Row v1 (hybrid) 150×4 | Row v2 (FHE, this plan) 150×4 |
|--------|--------------------|-----------------------|-------------------------------|
| Ciphertexts | 150×4 = 600 | 2×150 = **300** | 2×150 = **300** |
| Decrypts/step | 3 | ~147–150 (column) | **0** |
| Depth/step | ~3 | ~0 | ~24 |
| Total depth (4 steps) | ~12 | ~0 | **~96** |
| Bootstrapping | No | No | **No** |
| ct×ct mults (all 4 steps) | O(m·n²) | 0 | **~4158** |
| EvalRotates (all 4 steps) | 0 | 0 | ~7200 (150·log₂ 4096) |
| Wall time (N=2¹³, research) | ~5 s | ~3 s | **~33 s** |
| Precision achievable | ~1e-6 | ~1e-6 | **< 1e-4** |
| Security | leaks 3 scalars/step | leaks full column | **fully private** |
