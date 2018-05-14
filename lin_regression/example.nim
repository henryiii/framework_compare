# From:
# https://github.com/narimiran/narimiran.github.io/blob/master/code/python-numpy-nim/gradDesc.nim
#
# Compile with:
# nim c -d:release filename.nim
# #
# (On a mac, `brew install nim` will install nim)

import random, times, math

randomize(444)

const
  N = 10_000
  sigma = 0.1
  f = 2 / N
  mu = 0.001
  nEpochs = 10_000


proc randomNormal(mean = 0.0, std = 1.0): float =
  # https://github.com/mratsim/Arraymancer/blob/master/src/tensor/init_cpu.nim#L209-L222
  let
    x = rand(1.0)
    y = rand(1.0)
    rho = sqrt(-2.0 * ln(1.0 - y))
  return rho * cos(2.0 * PI * x) * std + mean


var x, d: array[N, float]
for i in 0 ..< N:
  x[i] = f * i.float
  d[i] = 3.0 + 2.0 * x[i] + sigma * randomNormal()


proc gradientDescent(x, d: array[N, float], mu: float, nEpochs: int):
    tuple[w0, w1: float] =
  var
    y: array[N, float]
    err: float
    w0, w1: float

  for n in 1 .. nEpochs:
    var grad0, grad1: float

    for i in 0 ..< N:
      err = f * (d[i] - y[i])
      grad0 += err
      grad1 += err * x[i]

    w0 += mu * grad0
    w1 += mu * grad1

    for i in 0 ..< N:
      y[i] = w0 + w1 * x[i]

  return (w0, w1)


let start = cpuTime()
echo gradientDescent(x, d, mu, nEpochs)
echo "Nim time: ", cpuTime() - start, " seconds"
