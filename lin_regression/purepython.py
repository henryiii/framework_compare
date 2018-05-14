from gen import Gen

class PurePython(Gen):
    def prepare(self):
        self.x = self.x.tolist()
        self.d = self.d.tolist()

    def run(self):
        f = 2 / self.N

        # "Empty" predictions, errors, weights, gradients.
        y = [0] * self.N
        w = [0, 0]
        grad = [0, 0]

        for _ in range(self.N_epochs):
            # Can't use a generator because we need to
            # access its elements twice.
            err = tuple(i - j for i, j in zip(self.d, y))
            grad[0] = f * sum(err)
            grad[1] = f * sum(i * j for i, j in zip(err, self.x))
            w = [i + self.mu * j for i, j in zip(w, grad)]
            y = (w[0] + w[1] * i for i in self.x)

        return w


if __name__ == '__main__':
    gen = PurePython()
    gen.simple_time()
