#!/usr/bin/env bash

echo "Python:"

python np_inv.py
python np_pinv.py
python purepython.py
python purenumpy.py
python vectornumba.py
python purenumba.py
python purecython.py
python graphtensorflow.py
python eagertensorflow.py
python purepytorch.py

echo "PyPy:"

pypy3 purepython.py
pypy3 purenumpy.py

echo "Nim:"

nim c -d:release example.nim
./example

echo "C++:"
$CXX -std=c++11 compile.cpp -O3
./a.out
