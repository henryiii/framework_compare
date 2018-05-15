from gen import Gen
import numpy as np
from numpy import float32

class PureNumpy(Gen):

    def run(self):
        f = float32(2 / self.N)

        y = np.empty(self.N, dtype=float32)
        err = np.zeros(self.N, dtype=float32)
        w = np.zeros(2, dtype=float32)
        grad = np.empty(2, dtype=float32)

        for _ in range(self.N_epochs):
            np.subtract(self.d, y, out=err)
            grad[:] = f * err.sum(), f * (err @ self.x)
            w += self.mu * grad
            y = w[0] + w[1] * self.x

        return w


if __name__ == '__main__':
    gen = PureNumpy()
    gen.simple_time()
