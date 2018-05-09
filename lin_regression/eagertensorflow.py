from gen import Gen
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

class EagerTensorFlow(Gen):

    def prepare(self):
        self.d = self.d[:,None]
        tf.enable_eager_execution()

    def run(self):
        f = 2 / self.N

        w = tf.zeros((2, 1), dtype=tf.float32)
        for epoch in range(self.N_epochs):
            y = tf.matmul(self.X, w)
            e = y - self.d
            grad = f * tf.matmul(tf.transpose(self.X), e)

            w -= self.mu * grad

        return w.numpy().squeeze()


if __name__ == '__main__':
    gen = EagerTensorFlow()
    gen.simple_time()
