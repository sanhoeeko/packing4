import matplotlib.pyplot as plt
import numpy as np

from kernel import ker
from myio import DataSet
from render import StateRenderer, State


class StateHandle:
    def __init__(self, N, n, d, boundary_a, boundary_b):
        self.N = N
        self.n, self.d = n, d
        self.A, self.B = boundary_a, boundary_b
        self.data_ptr = ker.createState(N, boundary_a, boundary_b)
        self.dataset = DataSet("data.h5", self.metadata)

    @classmethod
    def fromDensity(cls, N, n, d, fraction_as_disks, initial_boundary_aspect):
        B = np.sqrt(N / (fraction_as_disks * initial_boundary_aspect))
        A = B * initial_boundary_aspect
        return cls(N, n, d, A, B)

    def get(self):
        s = State(self.N, self.n, self.d, self.A, self.B, ker.getStateData(self.data_ptr, self.N))
        self.dataset.append(s)
        return s

    @property
    def metadata(self):
        return {
            'N': self.N,
            'n': self.n,
            'd': self.d,
        }

    def initPotential(self, potential_name):
        potential_id = {
            "Hertzian": 0,
            "ScreenedCoulomb": 1,
        }[potential_name]
        ker.setEnums(potential_id)
        return ker.setRod(self.n, self.d)

    def maxGradients(self):
        return ker.getStateMaxGradients(self.data_ptr)

    def iterationSteps(self):
        return ker.getNumOfIterations(self.data_ptr)

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
    state = StateHandle.fromDensity(400, 2, 0.125, 0.5, 1.0)
    state.initAsDisks()

    state.initPotential('ScreenedCoulomb')
    for i in range(70):
        state.setBoundary(state.A, state.B - 0.25)
        energy = state.equilibriumGD()
        gs = state.maxGradients()
        s = state.get()
        its = len(gs)
        print(i, f'G={gs[-1]}, E={energy}, nsteps={its}')
        # if gs[-1] > 0: break

    '''
    r = StateRenderer(s)
    r.drawBoundary()
    # r.drawParticles(state.residualForceAmp())
    r.drawParticles()
    plt.show()
    '''