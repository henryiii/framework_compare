from gen import Gen

import pyximport; pyximport.install()
from cython_loop import cython_loop

class PureCython(Gen):
    def run(self):
        return cython_loop(self.x, self.d, self.mu, self.N, self.N_epochs)

if __name__ == '__main__':
    gen = PureCython()
    gen.simple_time()
