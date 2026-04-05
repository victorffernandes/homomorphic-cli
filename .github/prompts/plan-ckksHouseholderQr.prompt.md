## Exploration Summary: OpenFHE-Python CKKS Householder QR Prototype

I've explored the workspace and OpenFHE Python bindings to identify implementation patterns for encrypting Householder QR decomposition. Here are the findings:

---

### **1. EXISTING RELEVANT FILES**

[qr-householder-plain.py](open-fhe/qr-householder-plain.py) — Reference plaintext implementation (~100 LOC)
- Householder reflections using: `v = x + sign(x[0])||x|| e₁` 
- Updates: `R` and `Q` via rank-2 updates `A ← A - 2v(v^T A)`
- Helper: `matmul()`, `fro_norm()` for verification

[simple-real-numbers.py](open-fhe/openfhe-python/examples/pke/simple-real-numbers.py) — Basic CKKS scaffold
- Shows key setup, encrypt/decrypt flow, rotations pattern

[polynomial-evaluation.py](open-fhe/openfhe-python/examples/pke/polynomial-evaluation.py) — `EvalPoly()` pattern
- Polynomial coefficients directly applied to ciphertext

[function-evaluation.py](open-fhe/openfhe-python/examples/pke/function-evaluation.py) — **Critical: sqrt & Chebyshev**
- `cc.EvalChebyshevFunction(math.sqrt, ct, lower, upper, poly_degree=50)` with domain bounds
- `cc.EvalLogistic()` for sigmoid (shows Chebyshev complexity)

[gauss.py](open-fhe/gauss.py) — Gaussian elimination solver (shows project's linear algebra maturity)

---

### **2. EXACT OPENFHE PYTHON API METHODS**

#### **Encrypt/Decrypt Core**
```python
cc = GenCryptoContext(CCParamsCKKSRNS())
keys = cc.KeyGen()
plaintext = cc.MakeCKKSPackedPlaintext(vector)
ciphertext = cc.Encrypt(keys.publicKey, plaintext)
result = cc.Decrypt(ciphertext, keys.secretKey)
decrypted = result.GetCKKSPackedValue()
```

#### **Key Material (Must enable features first)**
```python
cc.Enable(PKESchemeFeature.PKE)          # Basic encryption
cc.Enable(PKESchemeFeature.KEYSWITCH)    # Required for mults
cc.Enable(PKESchemeFeature.LEVELEDSHE)   # Leveled homomorphic ops
cc.Enable(PKESchemeFeature.ADVANCEDSHE)  # Required for EvalChebyshev
cc.EvalMultKeyGen(keys.secretKey)        # Unlocks multiplication
cc.EvalRotateKeyGen(keys.secretKey, [rotation_indices])
```

#### **Arithmetic Operations**
```python
c_add = cc.EvalAdd(c1, c2)
c_sub = cc.EvalSub(c1, c2)
c_mult = cc.EvalMult(c1, c2)             # Consumes 1 depth level
c_scalar = cc.EvalMult(c, 3.5)           # Scalar (no depth cost)
c_rot = cc.EvalRotate(c, 1)              # Rotate slots left by 1
```

#### **HOUSEHOLDER-CRITICAL: Function Approximation**
```python
# Chebyshev series approximation of arbitrary functions
result = cc.EvalChebyshevFunction(
    func,              # Python callable: math.sqrt or lambda x: np.sign(x)
    ciphertext,        # Input ciphertext
    lower_bound,       # Domain constraint (must know bounds!)
    upper_bound,
    poly_degree        # 20-50 typical; affects depth & accuracy
)

# Direct polynomial evaluation (cheaper than Chebyshev)
result = cc.EvalPoly(ciphertext, [c0, c1, c2, ...])  # P(x) = c0 + c1*x + ...

# Logistic sigmoid (uses Chebyshev internally, fixed domain [-bound, bound])
result = cc.EvalLogistic(ciphertext, lower, upper, poly_degree=16)

# Manual Chebyshev series (coefficients pre-computed)
result = cc.EvalChebyshevSeries(ciphertext, coefficients, a, b)

# Rescaling (manual depth management)
c_rescaled = cc.Rescale(c_after_mult)
```

---

### **3. KNOWN CONSTRAINTS FROM EXAMPLES**

| Parameter | Typical Values | Notes |
|-----------|---|---|
| **Ring Dimension** | 1024 (1<<10) | Defines slot count = N/2 = 512 real values. Larger = slower but allows bigger matrices. |
| **Multiplicative Depth** | 1–10 | Tight budget! Each mult costs +1. Chebyshev sqrt ≈ +6–7. Full HQR on 3×3 ≈ 8–9. |
| **Scaling Mod Size** | 50 bits | Standard. Sqrt examples use 59–78 for larger ops. |
| **First Mod Size** | 60 bits | Slightly larger than scaling mods. |
| **Batch Size** | 8–16 | Plaintext vector length (packed slots). |
| **Poly Degree (Chebyshev)** | 16–50 | Accuracy vs. depth tradeoff. sqrt.py uses 50; logistic uses 16. |

#### **Depth Budget Profile (from examples)**
- **Logistic threshold**: depth ≥ 6, poly_degree=16
- **Square root**: depth ≥ 7, poly_degree=50
- **Bootstrapping (Chebyshev)**: depth ≥ 10
- **Householder estimate**: depth 7–10 depending on matrix size

---

### **4. BLOCKERS & WORKAROUNDS FOR HOUSEHOLDER NONLINEARS**

| Blocker | Issue | Workaround |
|---------|-------|-----------|
| **Sign function** | No built-in `cc.EvalSign()` | Use `cc.EvalChebyshevFunction(lambda x: np.sign(x), ct, -1, 1, poly_degree=20)` or rewrite as `sign(x) = x / (|x| + ε)` |
| **Square root** | Input domain-dependent; can't compute on 0 | `cc.EvalChebyshevFunction(math.sqrt, ct, ε, upper_bound, poly_degree=50)`. Add small epsilon to inputs. |
| **If x == 0 checks** | No encrypted conditionals | Skip check entirely (assume non-zero for random matrices), OR always execute both branches, OR decrypt locally & prompt |
| **Depth explosion** | k iterations × sqrt(+6) × 2 dot products (+2 each) | Limit to 2×2 matrices (depth ~8) or 3×3 (depth ~10). Use reduced poly_degree for approximations. |
| **Packed slot encoding** | Matrix flattens across N/2 slots; permutation overhead | For 2×2: use 4 ciphertexts (one per element) OR flatten & manage index mapping carefully. |

---

### **5. SUGGESTED MODULE STRUCTURE FOR MINIMAL PROTOTYPE**

```python
# qr-householder-cipher.py

# === CONSTANTS ===
MATRIX_SIZE = 2  # Start here
RING_DIM = 1 << 10
MULT_DEPTH = 8
SQRT_POLY_DEGREE = 30  # Reduced from 50 to save depth
SIGN_POLY_DEGREE = 15

# === UTILITIES ===
def create_ckks_context():
    """Setup crypto context with Householder-friendly params."""
    params = CCParamsCKKSRNS()
    params.SetRingDim(RING_DIM)
    params.SetMultiplicativeDepth(MULT_DEPTH)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)  # Needed
    
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys

def encrypt_matrix(A, cc, keys):
    """Flatten m×n matrix to ciphertexts."""
    m, n = len(A), len(A[0])
    ct_matrix = []
    for i in range(m):
        for j in range(n):
            pt = cc.MakeCKKSPackedPlaintext([A[i][j]] + [0]*7)  # Pad to batch_size
            ct = cc.Encrypt(keys.publicKey, pt)
            ct_matrix.append((i, j, ct))
    return ct_matrix

def decrypt_matrix(ct_matrix, cc, keys, m, n):
    """Reconstruct decrypted matrix."""
    result = [[0.0]*n for _ in range(m)]
    for i, j, ct in ct_matrix:
        pt = cc.Decrypt(ct, keys.secretKey)
        result[i][j] = pt.GetCKKSPackedValue()[0]
    return result

# === CORE HOUSEHOLDER ===
def encrypt_norm_of_vector(ct_x_list, cc, lower_bound=0, upper_bound=1):
    """
    Compute ||x||² = sum(x_i²) then sqrt it (Chebyshev).
    ct_x_list: list of ciphertexts for x[0], x[1], ...
    """
    # First: sum of squares
    ct_sum_sq = cc.EvalMult(ct_x_list[0], ct_x_list[0])  # x[0]²
    for i in range(1, len(ct_x_list)):
        ct_xi_sq = cc.EvalMult(ct_x_list[i], ct_x_list[i])
        ct_sum_sq = cc.EvalAdd(ct_sum_sq, ct_xi_sq)  # Depth: +len(x)
    
    # Second: sqrt via Chebyshev
    # CAREFUL: lower_bound should be > 0 to avoid sqrt(0) issues
    ct_norm = cc.EvalChebyshevFunction(
        math.sqrt, 
        ct_sum_sq, 
        lower_bound,
        upper_bound,
        SQRT_POLY_DEGREE
    )  # Depth: +6–7
    return ct_norm

def encrypt_sign_approx(ct_x, cc):
    """Approximate sign(x) using Chebyshev series."""
    # sign(x) ≈ x / (|x| + 0.01) on [-1, 1]
    # OR use direct Chebyshev for [-1, 1] range
    ct_sign = cc.EvalChebyshevFunction(
        lambda t: 1 if t > 0 else (-1 if t < 0 else 0),
        ct_x,
        -1, 1,
        SIGN_POLY_DEGREE
    )  # Depth: +3–4
    return ct_sign

def householder_reflection_step(ct_x_list, cc, keys, k):
    """
    Compute Householder reflection vector v for x = A[k:m, k].
    Returns: ct_v_list (encrypted v vector)
    
    Logic:
      norm_x = sqrt(sum(x[i]²))
      sign_x0 = sign(x[0])
      v[0] = x[0] + sign_x0 * norm_x
      v[i] = x[i] for i > 0
      v = v / ||v||
    """
    ct_norm = encrypt_norm_of_vector(ct_x_list)  # Depth: +len(x)+6
    ct_sign = encrypt_sign_approx(ct_x_list[0], cc)  # Depth: +3–4
    
    # v[0] = x[0] + sign(x[0]) * norm_x
    ct_mult_sign_norm = cc.EvalMult(ct_sign, ct_norm)  # Depth: +1
    ct_v0 = cc.EvalAdd(ct_x_list[0], ct_mult_sign_norm)  # Depth: +1
    
    ct_v_list = [ct_v0] + ct_x_list[1:]
    
    # Normalize v by ||v||
    ct_v_norm = encrypt_norm_of_vector(ct_v_list)
    ct_v_normalized = [cc.EvalMult(ct_vi, cc.EvalMult(ct_v_norm, -1.0)) 
                       for ct_vi in ct_v_list]  # ~1/||v|| trick
    
    return ct_v_normalized

# === ENTRY POINT ===
def householder_ckks_qr(A, cc, keys):
    """
    Full Householder QR over CKKS.
    A: plaintext m×n matrix (float)
    Returns: Q, R (decrypted)
    """
    m, n = len(A), len(A[0])
    
    # Initialize: encrypt A into ct_A
    ct_A = encrypt_matrix(A, cc, keys)
    ct_Q = encrypt_matrix([[1 if i == j else 0 for j in range(m)] 
                           for i in range(m)], cc, keys)  # Identity
    
    # QR iterations
    for k in range(min(m, n)):
        # Extract A[k:, k] (encrypted)
        ct_x = [ct for (i, j, ct) in ct_A if i >= k and j == k]
        
        # Compute Householder reflection
        ct_v = householder_reflection_step(ct_x, cc, keys, k)  # Depth: ~8–9
        
        # Apply to A[k:, k:] and Q
        # ... (matrix update logic)
        
    # Decrypt results
    Q = decrypt_matrix(ct_Q, cc, keys, m, m)
    R = decrypt_matrix(ct_A, cc, keys, m, n)
    return Q, R

# === MAIN ===
if __name__ == "__main__":
    A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]  # 3×3 test
    cc, keys = create_ckks_context()
    Q, R = householder_ckks_qr(A, cc, keys)
    print("Q:", Q)
    print("R:", R)
```

---

### **6. RESEARCH SUMMARY: Key Gaps**

1. **Max practical matrix size?** 
   - Depth budget ≈ 8–10 for current examples  
   - 2×2: ~8 depth (safe)  
   - 3×3: ~10 depth (tight)  
   - 4×4+: likely exceeds budget without bootstrapping

2. **Does Open-FHE have sign approximation built-in?**
   - No. Must use `EvalChebyshevFunction` or hybrid (decrypt + re-encrypt)

3. **Epsilon-based division safety?**
   - `1 / (|x| + ε)` Chebyshev approximated, but ε tuning critical to avoid ~0 inputs

4. **Packed vs. unpacked encoding?**
   - Packed (one ct per matrix) more efficient but requires careful slot alignment  
   - Unpacked (one ct per element) simpler for prototyping but uses more ciphertexts

5. **Depth optimization tricks from literature?**
   - Iterative refinement (Newton) likely cheaper than direct Chebyshev for sqrt
   - Merged evaluation (combine sign & norm into single circuit) might save ops

---

### **IMMEDIATE NEXT STEPS**

1. Copy [simple-real-numbers.py](open-fhe/openfhe-python/examples/pke/simple-real-numbers.py) → `qr-householder-cipher.py`
2. Add sqrt via `EvalChebyshevFunction(math.sqrt, ...)` (from function-evaluation.py)
3. Prototype `encrypt_compute_norm()` on a 2-element vector
4. Build single Householder reflection step for 2×2 input
5. **Profile depth at each step**—abort if > 10
6. Add sign approximation last (most complex)
7. Extend to full QR if budget permits

**Success metric**: Encrypt a 2×2 matrix, compute one Householder step, decrypt, verify against plaintext reference with relative error < 10⁻⁴.
