import numpy as np

k = 1
N = 4

alpha = 0.005
secret = np.random.randint(2, size=(N))
rng = np.random.default_rng()

xN_1 = np.poly1d([1] + [0] * (N - 1) + [1])

def error():
    return rng.normal(scale=alpha, size=(N)) % 1

def tlwe(s):
    a = np.array(rng.uniform(low=0, high=1, size=(k,N))) % 1
    b = np.array([0])
    for ax in a:
        product = np.polymul(np.poly1d(ax), np.poly1d(s))
        b = np.polyadd(b, product)
    b = np.polyadd(b,error())
    b = mod_xN_1(b)
    
    z = np.append(a,[b])
    z.shape = (k+1,N)
    
    return z


def trivial_tlwe(m):
    a = np.array(np.zeros(shape=(k,N)))
    b = np.array([m])
    
    a = np.append(a,[b])
    a.shape = (k+1,N)
    
    return a

def sum_tlwe(s1, s2):
    a = np.array(np.zeros(shape=(k+1,N)))
    for i in range(k):
        a[i] = mod_xN_1(np.polyadd(np.poly1d(s1[i]), np.poly1d(s2[i]))).coeffs

    # print("sum tlwe")
    # print(np.poly1d(a[0]), sum_b)
    
    return a

def phase_s(sample,s):
    sa = np.array([0])
    a = sample[:-1]
    b = sample[-1]
    for ax in a:
        product = np.polymul(np.poly1d(ax), np.poly1d(s))
        sa = np.polyadd(sa, product)
    r = np.polysub(b, sa)
    r = mod_xN_1(r)
    return r.coeffs

def mod_xN_1(P):
    z = np.poly1d(P.coeffs % 1)
    _, resto = np.polydiv(z, xN_1)
    return np.poly1d(resto)

def decompose(a, bg, l): # a kxN
    a_ = []
    for a_i in a:
        a_t = a_i
        m = 1 / bg ** l
        a_t = np.around(a_i * bg ** l) * m
        a_.append(a_t)

    r = []
    for a_i in a_:
        a__ = []
        for a_ij in a_i:
            a_ijp = []
            residual = a_ij
            for p in range(l):
                z = round(residual * bg**p)
                a_ijp.append(z)
                residual = residual - (z / bg**p)
            a__.append(a_ijp)
        r.append(a__)
    u = [] * (k+1)
    for i in range(k+1):
        u.append([])
        for p in range(l):
            f = []
            for j in range(N):
                f.append(r[i][j][p])
            u[i].append(f)
    return u

def norm(m, l):
    r = []
    for i in range(k+1):
        z = []
        for p in range(l):
            z.append(np.linalg.vector_norm(m[i][p]))
        r.append(z)
    print(z)

def H(l, bg):
    l_ = (k+1)*l
    h = np.zeros(shape=(l_,k+1))
    for i in range(k+1):
        for j in range(l):
            h[i*l + j][i] = 1 / bg**j
    return h

def tgsw(m, l, H):
    l_ = (k+1)*l
    
    tgsw_s = []
    tlwe_s = [] # (k+1)*l, k+1
    H_m = H * m # (k+1)*l, k+1
    
    for i in range(l_):
        tlwe_s.append(tlwe(secret))
        tgsw_s.append(sum_tlwe(tlwe_s[i], H_m[i]))
        #tgsw_s.append(tlwe_s[i])

    print(tlwe_s)
    print(H_m)
    print(tgsw_s)
    return tgsw_s


def phase_tgsw(tgsw_s, secret, l):
    msg = []
    for i in range((k+1)*l):
         msg.append(phase_s(tgsw_s[i], secret))
    return msg

# 35094913 35094911 35094700 35094900 36021438 
a = tlwe(secret)
print("tlwe :", phase_s(a, secret))

l = 4
print(H(l, 2))
s = tgsw(0, l, H(l, 2))
print(phase_tgsw(s, secret, l))
#dec = decompose(a, 2, l)
