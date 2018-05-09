from gen import Gen
import itertools
import numpy as np

class PureNumpy(Gen):

    def run(self):
        f = 2 / self.N

        y = np.zeros(self.N)
        err = np.zeros(self.N)
        w = np.zeros(2)
        grad = np.empty(2)

        for _ in itertools.repeat(None, self.N_epochs):
            np.subtract(self.d, y, out=err)
            grad[:] = f * np.sum(err), f * (err @ self.x)
            w += self.mu * grad
            y = w[0] + w[1] * self.x
        return w


if __name__ == '__main__':
    gen = PureNumpy()
    gen.simple_time()
