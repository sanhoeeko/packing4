import ctypes as ct
import random as rd
from math import pi
import matplotlib.pyplot as plt

import numpy as np

debug_dll = "./x64/Debug/OMPFrame.dll"
release_dll = "./x64/Release/OMPFrame.dll"

def compare(v: np.ndarray, v_ref: np.ndarray):
    res = np.abs(v - v_ref)
    nans = np.bitwise_xor(np.isnan(v), np.isnan(v_ref))
    res[np.bitwise_and(np.invert(nans), np.isnan(v_ref))] = 0
    res[nans] = np.inf
    return res


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
ker.dll.interpolatePotential.argtypes = [ct.c_float] * 3
ker.dll.interpolatePotential.restype = ct.c_float
ker.dll.precisePotential.argtypes = [ct.c_float] * 3
ker.dll.precisePotential.restype = ct.c_float

n = 3
d = 0.25

ker.dll.setRod(n, d)
a = 1 + (n - 1) / 2 * d
b = 1


N = 10000
xs = np.random.uniform(-2 * a, 2 * a, size=(N,))
ys = np.random.uniform(-(a+b), a+b, size=(N,))
ts = np.random.uniform(0, pi, size=(N,))
rs = np.sqrt(xs*xs + ys*ys)
v0s = np.vectorize(ker.dll.precisePotential)(xs, ys, ts)
v1s = np.vectorize(ker.dll.fastPotential)(xs, ys, ts)
v2s = np.vectorize(ker.dll.interpolatePotential)(xs, ys, ts)

dif1 = compare(v1s, v0s)
dif2 = compare(v2s, v0s)

print('max without interpolation:', np.max(dif1))
print('mean without interpolation:', np.mean(dif1))
print('max:', np.max(dif2))
print('mean:', np.mean(dif2))

"""
def tryInterpolate(x, y):
    try:
        return ker.dll.interpolatePotential(x, y, t)
    except:
        return 114514


t = pi / 4
xs = np.arange(-2 * a, 2 * a, 0.01)
ys = np.arange(-(a+b), a+b, 0.01)
X, Y = np.meshgrid(xs, ys)
V0 = np.vectorize(lambda x, y: ker.dll.precisePotential(x, y, t))(X, Y)
V1 = np.vectorize(lambda x, y: ker.dll.fastPotential(x, y, t))(X, Y)
V2 = np.vectorize(tryInterpolate)(X, Y)
# V2 = np.vectorize(lambda x, y: ker.dll.interpolatePotential(x, y, t))(X, Y)
"""
