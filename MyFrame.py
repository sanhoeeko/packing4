import time

import numpy as np

from kernel import ker
from myio import DataSet
from render import State


class StateHandle:
    def __init__(self, N, n, d, boundary_a, boundary_b):
        self.N = N
        self.n, self.d = n, d
        self.A, self.B = boundary_a, boundary_b
        self.data_ptr = ker.createState(N, boundary_a, boundary_b)
        self.dataset = DataSet("data.h5", self.metadata)
        self.cnt = 0
        self.energy_cache = None

    @classmethod
    def fromCircDensity(cls, N, n, d, fraction_as_disks, initial_boundary_aspect):
        B = np.sqrt(N / (fraction_as_disks * initial_boundary_aspect))
        A = B * initial_boundary_aspect
        return cls(N, n, d, A, B)

    @property
    def density(self):
        return self.N / (np.pi * self.A * self.B)

    def get(self):
        s = State(
            self.cnt, self.N, self.n, self.d, self.A, self.B,
            ker.getStateData(self.data_ptr, self.N),
            {
                'energy': self.energy_cache,
                'max_residual_force': np.max(self.residualForceAmp()),
            }
        )
        self.dataset.append(s)
        self.cnt += 1
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
        start_t = time.perf_counter()
        self.energy_cache = ker.equilibriumGD(self.data_ptr)
        end_t = time.perf_counter()
        elapse_t = end_t - start_t
        return elapse_t


if __name__ == '__main__':
    state = StateHandle.fromCircDensity(1000, 2, 0.25, 0.4, 1.0)
    state.initAsDisks()

    state.initPotential('Hertzian')
    for i in range(1000):
        if state.density > 1.2: break
        state.setBoundary(state.A, state.B - 0.1)
        dt = state.equilibriumGD()
        s = state.get()
        gs = state.maxGradients()
        its = len(gs)
        print(i, f'G={gs[-1]}, E={s.energy}, nsteps={its}, speed: {its / dt} it/s')

