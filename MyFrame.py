import matplotlib.pyplot as plt
import numpy as np

from kernel import ker
from render import State


class StateHandle:
    def __init__(self, N, n, d, boundary_a, boundary_b):
        self.N = N
        self.n, self.d = n, d
        self.A, self.B = boundary_a, boundary_b
        self.data_ptr = ker.createState(N, boundary_a, boundary_b)

    @classmethod
    def fromDensity(cls, N, n, d, fraction_as_disks, initial_boundary_aspect):
        B = np.sqrt(N / (fraction_as_disks * initial_boundary_aspect))
        A = B * initial_boundary_aspect
        return cls(N, n, d, A, B)

    def get(self):
        return State(self.N, self.n, self.d, self.A, self.B, ker.getStateData(self.data_ptr, self.N))

    def initPotential(self):
        return ker.setRod(self.n, self.d)

    def maxGradients(self):
        return ker.getStateMaxGradients(self.data_ptr)

    def residualForce(self):
        return ker.getStateResidualForce(self.data_ptr, self.N)

    def residualForceAmp(self):
        f2 = self.residualForce() ** 2
        return np.sqrt(np.sum(f2, axis=1))

    def initAsDisks(self):
        return ker.initStateAsDisks(self.data_ptr)

    def setBoundary(self, boundary_a, boundary_b):
        self.A, self.B = boundary_a, boundary_b
        return ker.setBoundary(self.data_ptr, boundary_a, boundary_b)

    def equilibriumGD(self):
        return ker.equilibriumGD(self.data_ptr)


if __name__ == '__main__':
    # A, B = 36, 36
    state = StateHandle.fromDensity(1000, 2, 0.25, 0.5, 1.0)
    state.initAsDisks()

    state.initPotential()
    for i in range(100):
        state.setBoundary(state.A, state.B - 0.2)
        energy = state.equilibriumGD()
        gs = state.maxGradients()
        print(i, f'G={gs[-1]}', f'E={energy}')
        # if gs[-1] > 0: break

    s = state.get()
    r = s.render()
    r.drawBoundary()
    # r.drawParticles(state.residualForceAmp())
    r.drawParticles()
    plt.show()
