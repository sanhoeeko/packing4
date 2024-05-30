import ctypes as ct

import numpy as np

from kernel import ker
from render import State


class StateGetter:
    def __init__(self, N, boundary_a, boundary_b):
        self.N = N
        self.A, self.B = boundary_a, boundary_b
        self.data_ptr = ker.createState(N, boundary_a, boundary_b)

    def get(self):
        array_pointer = ct.cast(ker.getStateData(self.data_ptr), ct.POINTER(ct.c_float * self.N * 3))
        numpy_array = np.ctypeslib.as_array(array_pointer.contents)
        # .copy() is necessary
        return State(self.N, self.A, self.B, numpy_array.copy())  

    def maxGradients(self):
        return ker.getStateMaxGradient(self.data_ptr)

    def initAsDisks(self):
        return ker.initStateAsDisks(self.data_ptr)


state = StateGetter(1000, 30, 30)
r0 = state.get().render()
r0.drawBoundary()
r0.drawParticles()
state.initAsDisks()
gs = state.maxGradients()
r = state.get().render()
r.drawBoundary()
r.drawParticles()

'''
n = 3
d = 0.25

ker.dll.setRod(n, d)
a = 1 + (n - 1) / 2 * d
b = 1

N = 10000
'''
