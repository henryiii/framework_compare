#!python
#cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, cdivision=True

# Compile with cython -a cython_loop.pyx to see the interactions with Python (should be only outside the looping part)

import numpy as np

def cython_loop(float[:] x, float[:] d, float mu, int N, int epochs):
    y_holder = np.zeros(N, dtype=np.float32)
    cdef float[:] y = y_holder

    cdef float f = 2.0 / N

    cdef float[2] w = (0.,0.)
    
    cdef float[2] grad
    cdef float err

    cdef int i

    for _ in range(epochs):
        grad = (0., 0.)

        for i in range(N):
            err = f * (d[i] - y[i])
            grad[0] += err
            grad[1] += err * x[i]

        w[0] += mu * grad[0]
        w[1] += mu * grad[1]

        for i in range(N):
            y[i] = w[0] + w[1] * x[i]

    return np.array(w, dtype=np.float32)
