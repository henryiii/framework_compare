from gen import Gen
import numpy as np

# Using numpy for storing large arrays
# Python's `array` could be used instead

from numba import jit, float32, int32

type_dict = {a:float32 for a in 'f grad0 grad1 w0 w1'.split()}

@jit(float32[:](float32[:], float32[:], float32, int32, int32),
        locals=type_dict,
        nopython=True)
def python_loop(x, d, mu, N, epochs):
    f = 2 / N

    y = np.zeros(N, dtype=np.float32)

    w0, w1 = 0., 0.

    for _ in range(epochs):
        grad0, grad1 = 0., 0.
        for i in range(N):
            err = f * (d[i] - y[i])
            grad0 += err
            grad1 += err * x[i]

        w0 += mu * grad0
        w1 += mu * grad1

        for i in range(N):
            y[i] = w0 + w1 * x[i]

    return np.array([w0, w1], dtype=np.float32)


class PureNumba(Gen):
    def run(self):
        return python_loop(self.x, self.d, self.mu, self.N, self.N_epochs)

if __name__ == '__main__':
    gen = PureNumba()
    gen.simple_time()
