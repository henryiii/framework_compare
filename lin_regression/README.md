This was taken from https://realpython.com/numpy-tensorflow-performance/

The testing class is in `gen.py`. 

To run a test, use `python testname.py`.

To run all tests, run `./runall.sh`.

My results on my MacBook:

```
Running NumpyInverse
  Solve time: 0.03 seconds
  Answer: w_0=2.9954, w_1=2.0029
Running NumpyPsuedoInverse
  Solve time: 0.03 seconds
  Answer: w_0=2.9954, w_1=2.0029
Running PurePython
  Solve time: 34.03 seconds
  Answer: w_0=2.9599, w_1=2.0330
Running PureNumpy
  Solve time: 0.38 seconds
  Answer: w_0=2.9598, w_1=2.0330
Running VectorNumba
  Solve time: 0.37 seconds
  Answer: w_0=2.9599, w_1=2.0330
Running PureNumba
  Solve time: 0.16 seconds
  Answer: w_0=2.9599, w_1=2.0330
Running PureCython
  Solve time: 0.15 seconds
  Answer: w_0=2.9599, w_1=2.0330
Running GraphTensorFlow
  Solve time: 1.64 seconds
  Answer: w_0=2.9599, w_1=2.0330
Running EagerTensorFlow
  Solve time: 4.96 seconds
  Answer: w_0=2.9599, w_1=2.0330
Running PurePyTorch
  Solve time: 1.07 seconds
  Answer: w_0=2.9599, w_1=2.0330
```

PyPy:

```
Running PurePython
  Solve time: 15.78 seconds
  Answer: w_0=2.9599, w_1=2.0330
Running PureNumpy
  Solve time: 1.72 seconds
  Answer: w_0=2.9599, w_1=2.0330
```

C++11:
```
Running C++ example
  Solve time: 0.113623
  Answer: w_0=3.01296, w_1=2.0307
```


Nim example (contributed):

```
(w0: 2.968954757075724, w1: 2.02593328759163)
Nim time: 0.139302 seconds
```

