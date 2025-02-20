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


class Simulator:
    def __init__(self, N, n, d, boundary_a, boundary_b, potential_name: str):
        self.N = int(N)
        self.n, self.d = int(n), d
        self.A, self.B = boundary_a, boundary_b
        self.potential_name = potential_name
        self.data_ptr = None

    @classmethod
    def createState(cls, N, n, d, boundary_a, boundary_b, potential_name: str, data_name='data'):
        """
        deprecated
        """
        self = cls(N, n, d, boundary_a, boundary_b, potential_name)
        self.data_ptr = ker.createState(N, boundary_a, boundary_b)
        self.id = data_name
        self.dataset = DataSet(f'{data_name}.h5', self.metadata)
        self.cnt = 0
        self.energy_cache = None
        return self

    @classmethod
    def _fromCircDensity(cls, N, n, d, fraction_as_disks, initial_boundary_aspect, potential_name: str, data_name: str):
        B = np.sqrt(N / (fraction_as_disks * initial_boundary_aspect))
        A = B * initial_boundary_aspect
        return cls.createState(N, n, d, A, B, potential_name, data_name)

    @classmethod
    def fromDataPtr(cls, N, n, d, boundary_a, boundary_b, potential_name: str, data_ptr: int):
        self = cls(N, n, d, boundary_a, boundary_b, potential_name)
        self.data_ptr = data_ptr
        return self

    @property
    def density(self):
        return self.N / (np.pi * self.A * self.B)

    def initPotential(self, workers: int, save_data=True):
        potential_str = self.potential_name.split(':')[0]
        potential_id = {
            "Hertzian": 0,
            "ScreenedCoulomb": 1,
            "Power": 2,
        }[potential_str]
        ker.setEnums(potential_id)
        if potential_str == "Power":
            ker.setPotentialPower(float(self.potential_name.split(':')[1]))
        return ker.setRod(self.n, self.d, workers, save_data)

    def get(self, record=True):
        s = State(
            self.cnt, self.N, self.n, self.d, self.A, self.B,
            ker.getStateData(self.data_ptr, self.N),
            {
                'energy_curve': self.energyCurve(),
                'energy': self.energy_cache,
                'max_residual_force': self.maxResidualForce(),
            }
        )
        if record:
            self.dataset.append(s)
            self.cnt += 1
        return s

    def loadDataToKernel(self, data: np.ndarray):
        """
        Assume that data is of the shape (N, 3)
        """
        data_n4 = np.hstack([data, np.zeros((self.N, 1))]).reshape(-1)
        return ker.setStateData(self.data_ptr, data_n4)

    @property
    def metadata(self):
        return {
            'potential': self.potential_name,
            'N': self.N,
            'n': self.n,
            'd': self.d,
        }

    def maxGradients(self):
        """
        Only for `initAsDisks`. Otherwise, gradient curve would not be saved.
        """
        return ker.getStateMaxGradients(self.data_ptr)

    def energyCurve(self):
        return ker.getStateEnergyCurve(self.data_ptr)

    def iterationSteps(self):
        return ker.getNumOfIterations(self.data_ptr)

    def residualForce(self):
        return ker.getStateResidualForce(self.data_ptr, self.N)

    def maxResidualForce(self):
        return ker.getStateMaxResidualForce(self.data_ptr)

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

    def equilibriumTemplate(self, method: str, max_iterations):
        start_t = time.perf_counter()
        self.energy_cache = getattr(ker, method)(self.data_ptr, int(max_iterations))
        end_t = time.perf_counter()
        elapse_t = end_t - start_t
        return elapse_t

    def equilibriumGD(self, max_iterations):
        return self.equilibriumTemplate('equilibriumGD', max_iterations)

    def eqLineGD(self, max_iterations):
        return self.equilibriumTemplate('eqLineGD', max_iterations)

    def eqLBFGS(self, max_iterations):
        return self.equilibriumTemplate('eqLBFGS', max_iterations)

    def eqMix(self, max_iterations):
        return self.equilibriumTemplate('eqMix', max_iterations)

    # landscape

    def energyLandscapeAlongGradient(self, max_stepsize: float, n: int):
        return ker.landscapeAlongGradient(self.data_ptr, max_stepsize, n)

    def energyLandscapeLBFGS(self, max_stepsize: float, n: int):
        return ker.landscapeLBFGS(self.data_ptr, max_stepsize, n)

    def energyLandscapeOnGradientSections(self, max_stepsize: float, n: int):
        return ker.landscapeOnGradientSections(self.data_ptr, max_stepsize, n)

    def ERoot(self, max_stepsize: float):
        return ker.ERoot(self.data_ptr, max_stepsize)

    def bestStepSize(self, max_stepsize: float):
        return ker.bestStepSize(self.data_ptr, max_stepsize)


class CommonSimulator:
    def __init__(self):
        self.simulator = None  # CommonSimulator does not inherit Simulator because self.simulator can be None

    def load(self, s: State, load_potential=True):
        if self.simulator is None:
            self.simulator = s.makeSimulator(None, s.potential, 'data')
            if load_potential:
                self.simulator.initPotential(4)
        else:
            self.simulator = Simulator.fromDataPtr(
                s.N, s.n, s.d, s.A, s.B, s.potential, self.simulator.data_ptr
            )
            self.simulator.loadDataToKernel(s.xyt)


common_simulator = CommonSimulator()
