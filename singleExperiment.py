import time

import numpy as np
import sympy as sp

from src.kernel import ker
from src.myio import DataSet
from src.render import State


class BoundaryScheduler:
    n, x = sp.symbols('n x')

    def __init__(self, func_a, func_b, A0, B0, max_step_size):
        """
        func_a, func_b :: (n: int, x: float) -> float, lambda expression
        returns the boundary A, B of the ith step, with initial A0, B0.
        """
        self.n = 0
        self.max_step_size = max_step_size
        n, x = BoundaryScheduler.n, BoundaryScheduler.x

        def reduce(f: sp.Expr, f0: float):
            x0 = sp.solve(sp.Eq(f.subs(n, 0), f0), x)[0]
            phi = f.subs(x, x0)
            return sp.lambdify(n, phi)

        self.func_a = reduce(func_a(n, x), A0)
        self.func_b = reduce(func_b(n, x), B0)

    def step(self):
        self.n += 1
        a, b = self.func_a(self.n), self.func_b(self.n)
        da, db = self.func_a(self.n - 1) - a, self.func_b(self.n - 1) - b
        if da > self.max_step_size:
            a = self.func_a(self.n - 1) - self.max_step_size
        if db > self.max_step_size:
            b = self.func_b(self.n - 1) - self.max_step_size
        return a, b

    @staticmethod
    def constant(n, x):
        return x


class StateHandle:
    def __init__(self, N, n, d, boundary_a, boundary_b, data_name='data'):
        self.N = N
        self.n, self.d = n, d
        self.A, self.B = boundary_a, boundary_b
        self.data_ptr = ker.createState(N, boundary_a, boundary_b)
        self.dataset = DataSet(f'{data_name}.h5', self.metadata)
        self.cnt = 0
        self.energy_cache = None
        self.boundary_scheduler = None

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

    def setBoundary(self, boundary_a, boundary_b):
        self.A, self.B = boundary_a, boundary_b
        return ker.setBoundary(self.data_ptr, boundary_a, boundary_b)

    def setBoundaryScheduler(self, func_a, func_b, max_step_size=0.1):
        self.boundary_scheduler = BoundaryScheduler(func_a, func_b, self.A, self.B, max_step_size)

    def compress(self):
        if self.boundary_scheduler is None:
            raise ValueError("No boundary scheduler")
        self.setBoundary(*self.boundary_scheduler.step())

    def initAsDisks(self):
        return ker.initStateAsDisks(self.data_ptr)

    def singleStep(self, mode: str):
        mode_id = {
            'Normal': 0,
            'AsDisks': 1,
        }[mode]

        def inner(step_size):
            return ker.singleStep(self.data_ptr, mode_id, step_size)

        return inner

    def equilibriumGD(self, max_iterations):
        start_t = time.perf_counter()
        self.energy_cache = ker.equilibriumGD(self.data_ptr, int(max_iterations))
        end_t = time.perf_counter()
        elapse_t = end_t - start_t
        return elapse_t


if __name__ == '__main__':
    q = 1 - 1e-3
    state = StateHandle.fromCircDensity(1000, 2, 0.25, 0.4, 1.0)
    state.initAsDisks()
    state.setBoundaryScheduler(BoundaryScheduler.constant, lambda n, x: x * q ** n)
    state.initPotential('ScreenedCoulomb')

    for i in range(1000):
        if state.density > 1.2: break
        state.compress()
        dt = state.equilibriumGD(2e5)
        s = state.get()
        gs = state.maxGradients()
        its = len(gs)
        print(i, f'G={gs[-1]}, E={s.energy}, nsteps={its}, speed: {its / dt} it/s')
