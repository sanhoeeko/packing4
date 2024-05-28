import ctypes as ct
import random as rd
from math import pi
import matplotlib.pyplot as plt

import numpy as np

debug_dll = "./x64/Debug/OMPFrame.dll"
release_dll = "./x64/Release/OMPFrame.dll"


class Kernel:
    def __init__(self):
        self.dll = ct.cdll.LoadLibrary(release_dll)


class State:
    def __init__(self):
        pass


ker = Kernel()
ker.dll.init()
ker.dll.setRod.argtypes = [ct.c_int, ct.c_float]
ker.dll.fastPotential.argtypes = [ct.c_float] * 3
ker.dll.fastPotential.restype = ct.c_float
ker.dll.precisePotential.argtypes = [ct.c_float] * 3
ker.dll.precisePotential.restype = ct.c_float

n = 4
d = 0.25

ker.dll.setRod(n, d)
a = 1 + (n - 1) / 2 * d
b = 1


N = 10000
xs = np.random.uniform(-2 * a, 2 * a, size=(N,))
ys = np.random.uniform(-(a+b), a+b, size=(N,))
ts = np.random.uniform(-2*pi, 2*pi, size=(N,))
rs = np.sqrt(xs*xs + ys*ys)
v0s = np.vectorize(ker.dll.precisePotential)(xs, ys, ts)
v1s = np.vectorize(ker.dll.fastPotential)(xs, ys, ts)

dif = np.abs(v1s - v0s)
print('max:', np.max(dif))
print('min:', np.min(dif))
