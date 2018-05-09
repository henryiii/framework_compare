from gen import Gen
import numpy as np

class NumpyPsuedoInverse(Gen):
    def prepare(self):
        self.d = self.d[:,np.newaxis]
    
    def run(self):
        return (np.linalg.pinv(self.X) @ self.d).squeeze()

if __name__ == '__main__':
    gen = NumpyPsuedoInverse()
    gen.simple_time()
