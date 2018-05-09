from gen import Gen
import numpy as np

class NumpyInverse(Gen):
    def prepare(self):
        self.d = self.d[:,np.newaxis]
    
    def run(self):
        return (np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.d).squeeze()

if __name__ == '__main__':
    gen = NumpyInverse()
    gen.simple_time()
