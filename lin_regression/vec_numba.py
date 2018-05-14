from gen import Gen
import numpy as np
from numba import jit, float32, int32

@jit(float32[:](float32[:], float32[:], float32, int32, int32), nopython=True)
def numba_loop(x, d, mu, N, epochs):
    f = 2 / N

    y = np.zeros(N, dtype=np.float32)
    w = np.zeros(2, dtype=np.float32)
    grad = np.zeros(2, dtype=np.float32)

    for _ in range(epochs):
        err = d - y
        grad[:] = f * np.sum(err), f * (err @ x)
        w += mu * grad
        y[:] = w[0] + w[1] * x

    return w

class VectorNumba(Gen):
    def run(self):
        return numba_loop(self.x, self.d, self.mu, self.N, self.N_epochs)

if __name__ == '__main__':
    gen = VectorNumba()
    gen.simple_time()
