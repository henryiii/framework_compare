#!/usr/bin/env bash

echo "Python:"

python np_inv.py
python np_pinv.py
python purepython.py
python purenumpy.py
python purenumba.py
python graphtensorflow.py
python eagertensorflow.py
python purepytorch.py

echo "PyPy:"

pypy3 purepython.py
pypy3 purenumpy.py

echo "Nim:"

nim -c -d:release example.nim
./example
