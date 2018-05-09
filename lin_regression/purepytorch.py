from gen import Gen
import torch
import numpy as np

class PurePyTorch(Gen):

    def prepare(self):
        self.X = torch.from_numpy(self.X)
        self.d = torch.from_numpy(self.d[:,None])

    def run(self):
        f = 2 / self.N

        w = torch.zeros(2, 1, dtype=torch.float)
        
        for epoch in range(self.N_epochs):
            y = torch.mm(self.X, w)
            e = y - self.d
            grad = f * torch.matmul(torch.t(self.X), e)

            w = w - self.mu * grad

        return w.numpy().squeeze()


if __name__ == '__main__':
    gen = PurePyTorch()
    gen.simple_time()
