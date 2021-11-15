import numpy as np
import abc
import time

class Gen(abc.ABC):
    mu = 0.001
    N_epochs = 10000
         
    def __init__(self, N = 10000):
        self.N = N
        np.random.seed(444)
        sigma = 0.1
        noise = sigma * np.random.randn(N)
        self.x = np.linspace(0, 2, N, dtype=np.float32)
        self.d = np.array(3 + 2 * self.x + noise, dtype=np.float32)
        
        # We need to prepend a column vector of 1s to `x`.
        self.X = np.column_stack((np.ones(N, dtype=self.x.dtype), self.x))
        
        self.prepare()

    def prepare(self):
        'Default prepare method. Override if you need new vectors. Not included in timing measurements.'

    @abc.abstractmethod
    def run(self):
        'All classes must implement this.'

    def simple_time(self):
        t0 = time.perf_counter()
        ws = self.run()
        t1 = time.perf_counter()
        print('Running {}'.format(self.__class__.__name__))
        print('  Solve time: {:.4f} seconds'.format(t1 - t0))
        print('  Answer: w_0={:.4f}, w_1={:.4f}'.format(*ws))
