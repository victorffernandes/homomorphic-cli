import numpy as np
from numpy.typing import NDArray
from numpy import poly1d

class Base():
    def __init__(self, k: int, N: int):
        self.params: dict[str, int] = {
            'k': k,
            'N': N,
        }
        self.mod_polynomial = np.poly1d([1] + [0] * (self.params['N'] - 1) + [1])

    def mod(self, P: poly1d):
        z = np.poly1d(P.coeffs % 1)
        _, resto = np.polydiv(z, self.mod_polynomial)
        return np.poly1d(resto)


class TLWE(Base):
    def __init__(self, k: int, N: int, values: NDArray[np.int64]):
        super().__init__(k, N)
        self.values = values
        
    def __str__(self):
        return self.values.__str__()
    
    def asPoly(self) -> NDArray[np.poly1d]:
        r = []
        
        for ax in range(self.params['k']+1):
            r.append(np.poly1d(self.values[ax]))
        return r
        
    def phase(self, s: NDArray[np.int64]):
        sa = np.array([0])
        a = self.values[:-1]
        b = self.values[-1]
        for ax in a:
            product = np.polymul(np.poly1d(ax), np.poly1d(s))
            sa = np.polyadd(sa, product)
        r = np.polysub(b, sa)
        r = super().mod(r)
        return r.coeffs
    
    def decompose(self, bg, l): # a kxN
        a = self.values
        k = self.params['k']
        N = self.params['N']
        
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
                u[i].append(np.poly1d(f))
        return u
    
    def __add__(self, other: 'TLWE'):
        a = np.array(np.zeros(shape=(self.params['k']+1,self.params['N'])))
        for i in range(self.params['k']+1):
            a[i] = super().mod(np.polyadd(np.poly1d(self.values[i]), np.poly1d(other.values[i]))).coeffs
        
        return TLWE(self.params['k'], self.params['N'], a)


class TLWE_Factory(Base):
    def __init__(self, k: int, N: int, alpha: int = 0.005):
        super().__init__(k, N)
        self.params['alpha'] = alpha
        
        self._rng = None
    
    def key_gen(self) -> NDArray[np.int64]:
        return np.random.randint(2, size=(self.params['N']))

    @property
    def rng(self):
        return np.random.default_rng()
    
    @staticmethod
    def error_distribution(self):
        return self.rng.normal(scale=self.params['alpha'], size=(self.params['N'])) % 1
    
    def trivial(self, message):
        a = np.array(np.zeros(shape=(self.params['k']+1,self.params['N'])))
        a[self.params['k'], 0] = message
        
        return TLWE(self.params['k'], self.params['N'], a)
    
    def fromPoly(self, a: NDArray[np.poly1d]):
        r = []
        for ax in a:
            r.append(ax.coeffs)
        return TLWE(self.params['k']+1, self.params['N'], r)
    
    def fresh(self, s: NDArray[np.int64]):
        a = np.array(self.rng.uniform(low=0, high=1, size=(self.params['k'],self.params['N']))) % 1
        b = np.array([0])
        for ax in a:
            product = np.polymul(np.poly1d(ax), np.poly1d(s))
            b = np.polyadd(b, product)
        b = np.polyadd(b,self.error_distribution(self))
        b = self.mod(b)
        
        z = np.append(a,[b])
        z.shape = (self.params['k']+1,self.params['N'])  
        return TLWE(self.params['k'], self.params['N'], z)
    

def example():
    factory: TLWE_Factory = TLWE_Factory(1, 2)
    key = factory.key_gen()
    sample1 = factory.fresh(key)
    sample2 = factory.trivial(0.5)
    sample3 = sample1 + sample2
    decomp = sample1.decompose(2, 4)
    print("sample1: ",sample1)
    print("sample2: ", sample2)
    print("sample3: ", sample3)
    print("decomp sample1: ", decomp)
    print("phase: ",sample3.phase(key))

if __name__ == '__main__':
    example()
