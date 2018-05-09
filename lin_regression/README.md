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
  Solve time: 0.57 seconds
  Answer: w_0=2.9599, w_1=2.0330
Running PureNumba
  Solve time: 0.37 seconds
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

