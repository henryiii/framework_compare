from gen import Gen
import tensorflow as tf

class GraphTensorFlow(Gen):

    def prepare(self):
        self.X_tf = tf.constant(self.X, dtype=tf.float32, name="X_tf")
        self.d_tf = tf.constant(self.d[:,None], dtype=tf.float32, name="d_tf")

    def run(self):
        f = 2 / self.N

        w = tf.Variable(tf.zeros((2, 1)), name="w_tf")
        y = tf.matmul(self.X_tf, w, name="y_tf")
        e = y - self.d_tf
        grad = f * tf.matmul(tf.transpose(self.X_tf), e)

        training_op = tf.assign(w, w - self.mu * grad)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            init.run()
            for epoch in range(self.N_epochs):
                sess.run(training_op)
            opt = w.eval()
        return opt.squeeze()


if __name__ == '__main__':
    gen = GraphTensorFlow()
    gen.simple_time()
