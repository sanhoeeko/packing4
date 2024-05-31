import matplotlib.pyplot as plt
import numpy as np

from kernel import ker
from render import State


class StateGetter:
    def __init__(self, N, n, d, boundary_a, boundary_b):
        self.N = N
        self.n, self.d = n, d
        self.A, self.B = boundary_a, boundary_b
        self.data_ptr = ker.createState(N, boundary_a, boundary_b)

    def get(self):
        return State(self.N, self.n, self.d, self.A, self.B, ker.getStateData(self.data_ptr, self.N))

    def maxGradients(self):
        return ker.getStateMaxGradients(self.data_ptr)

    def residualForce(self):
        return ker.getStateResidualForce(self.data_ptr, self.N)

    def initAsDisks(self):
        return ker.initStateAsDisks(self.data_ptr)


state = StateGetter(1000, 5, 0.25, 90, 15)
state.initAsDisks()
gs = state.maxGradients()
print(gs[-1])
fs = state.residualForce()
fabs = np.sqrt(fs[:, 0] ** 2 + fs[:, 1] ** 2 + fs[:, 2] ** 2)
r = state.get().render()
r.drawBoundary()
r.drawParticles(fabs)
plt.show()

'''
n = 3
d = 0.25

ker.dll.setRod(n, d)
a = 1 + (n - 1) / 2 * d
b = 1

N = 10000
'''
