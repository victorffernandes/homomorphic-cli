import tlwe
from tlwe import TLWE_Factory
import numpy as np
from numpy.typing import NDArray

class TGSW(tlwe.Base):
    def __init__(self, k: int, N: int, l: int, values: list[tlwe.TLWE]):
        super().__init__(k, N)
        self.params['l'] = l
        self.values = values
    
    def phase(self, s: NDArray[np.int64]):
        msg = []
        for i in range((self.params['k']+1)*self.params['l']):
            msg.append(self.values[i].phase(s))
        return msg

class TGSW_Factory(tlwe.Base):
    def __init__(self, k: int, N: int, alpha: int = 0.005):
        super().__init__(k, N)
        self.tlwe_factory = TLWE_Factory(k, N, alpha)
    
    def key_gen(self) -> NDArray[np.int64]:
        return np.random.randint(2, size=(self.params['N']))
    
    def gen_H(self, l, bg):
        l_ = (self.params['k']+1)*l
        h = np.zeros(shape=(l_,self.params['k']+1), dtype=np.poly1d)
        for i in range(self.params['k']+1):
            for j in range(l):
                h[i*l + j][i] = 1 / bg**j
        
        return list(map(lambda x: list(map(lambda y: np.poly1d(y), x)), h))
    
    def fresh(self, l, bg, s, m):
        l_ = (self.params['k']+1)*l
        message = np.poly1d(m)
        H = self.gen_H(l, bg)

        tgsw_s = []
        tlwe_s = [] # (k+1)*l, k+1
        
        H_m = [] # (k+1)*l, k+1
        
        for j in range(l_):
            temp = []
            for i in range(self.params['k']+1):
                temp.append(np.polymul(H[j][i], message))
            H_m.append(temp)

        for j in range(l_):
            tlwe_s.append(self.tlwe_factory.fresh(s).asPoly())
            temp = []
            for i in range(self.params['k']+1):
                sum = np.polyadd(tlwe_s[j][i], H_m[j][i])
                temp.append(sum)
            tgsw_s.append(self.tlwe_factory.fromPoly(temp))
            print("phase: ", tgsw_s[j].phase(s))
            #
        return tgsw_s

def example():
    factory_TGSW: TGSW_Factory = TGSW_Factory(1, 2)
    key = factory_TGSW.key_gen()
    H = factory_TGSW.fresh(4, 2, 0, 0.4)
    print("sample1: ",H)

if __name__ == '__main__':
    example()