from functools import lru_cache

import numpy as np
from scipy.spatial import Delaunay

import src.utils as ut
from src.graph import Graph, MergedGraph

bins_of_angle_dist = 90


class RenderSetup:
    def __init__(self, colors: np.ndarray, style: str, real_size=False):
        self.colors = colors
        self.style = style
        self.real_size = real_size


class State:
    def __init__(self, id, N, n, d, boundary_a, boundary_b, configuration: np.ndarray, others=None):
        if others is None:
            others = {}
        self.id = id
        self.N = N
        self.n, self.d = n, d
        self.A, self.B = boundary_a, boundary_b
        self.a, self.b = 1, 1 / (1 + (n - 1) * d / 2)
        self.xyt = configuration
        for key, value in others.items():
            setattr(self, key, value)

    @property
    def x(self):
        return self.xyt[:, 0]

    @property
    def y(self):
        return self.xyt[:, 1]

    @property
    def t(self):
        return self.xyt[:, 2] % np.pi

    @property
    def Gamma(self):
        return self.A / self.B

    @property
    def rho(self):
        return self.N / (np.pi * self.A * self.B)

    @property
    def phi(self):
        return self.rho * (np.pi + 4 * (self.gamma - 1)) / self.gamma ** 2

    @property
    def gamma(self):
        return 1 + (self.n - 1) * self.d / 2

    @property
    def metadata(self):
        return {
            'id': self.id,
            'A': self.A,
            'B': self.B,
            'energy_curve': self.energy_curve,
            'energy': self.energy,
            'max_residual_force': self.max_residual_force,
        }

    @classmethod
    def load(cls, configuration, up_meta: dict, metadata: dict):
        obj = cls(
            metadata['id'],
            int(up_meta['N']), int(up_meta['n']), float(up_meta['d']),
            float(metadata['A']), float(metadata['B']),
            configuration,
            others={
                'energy_curve': metadata['energy_curve'],
                'energy': metadata['energy'],
                'max_residual_force': metadata['max_residual_force'],
                'potential': up_meta['potential'],
            }
        )
        return obj

    def makeSimulator(self, dataset, potential_name, data_name: str):
        from src.simulator import Simulator
        obj = Simulator.createState(self.N, self.n, self.d, self.A, self.B, potential_name, data_name)
        obj.dataset = dataset
        obj.loadDataToKernel(self.xyt)
        return obj

    # @lru_cache(maxsize=None)  # DO NOT cache this! It will cause a memory leak.
    def toSites(self, n=None):
        """
        convert each rod to disks
        """
        n = self.n if n is None else n
        xy = np.array([self.x, self.y]).T
        uxy = np.array([np.cos(self.t), np.sin(self.t)]).T * self.d
        n_shift = -(self.n - 1) / 2.0
        xys = [xy + (k + n_shift) * uxy for k in range(0, n)]
        return np.vstack(xys)

    # @lru_cache(maxsize=None)  # DO NOT cache this! It will cause a memory leak.
    def voronoiDiagram(self, n=None) -> MergedGraph:
        points = self.toSites(n)  # input of Delaunay is (n_point, n_dim)
        delaunay = Delaunay(points)
        voro_graph = Graph(len(points)).from_delaunay(delaunay.vertex_neighbor_vertices)
        del points  # to cope with memory leak
        return voro_graph.merge(self.N)

    # analysis

    @property
    def globalSx(self):
        return np.mean(np.cos(2 * self.t))

    @property
    def logE(self):
        return np.log(self.energy)

    @property
    def descent_curve(self):
        return self.energy_curve

    @property
    def maxResidualForce(self):
        # bug?
        # from src.simulator import common_simulator as cs
        # cs.load(self)
        # return cs.simulator.maxResidualForce()
        return np.sqrt(np.max(np.sum(self.gradient ** 2, axis=1)))

    @property
    def gradient(self):
        from src.simulator import common_simulator as cs
        cs.load(self)
        return cs.simulator.residualForce()

    @property
    def moment(self):
        return self.gradient[:, 2]

    @property
    def gradientAmp(self):
        return np.sqrt(np.sum(self.gradient ** 2, axis=1))

    @property
    def meanDistance(self):
        from src.simulator import common_simulator as cs
        cs.load(self)
        return cs.simulator.meanDistance()

    @property
    def meanZ(self):
        from src.simulator import common_simulator as cs
        cs.load(self)
        return cs.simulator.meanContactZ()

    @property
    def finalStepSize(self):
        from src.simulator import common_simulator as cs
        cs.load(self)
        try:
            return cs.simulator.bestStepSize(10.0)
        except:
            return np.nan

    @staticmethod
    def distance(s1: 'State', s2: 'State'):
        dq = s2.xyt - s1.xyt
        return np.sqrt(np.mean(dq ** 2))

    # @lru_cache(maxsize=None)  # DO NOT cache this! It will cause a memory leak.
    def S_field(self) -> np.ndarray:
        def anglesOf(indices: set[int]):
            return np.array([self.t[i] for i in indices])

        def Q_ave(angles: np.ndarray):
            c = np.cos(2 * angles)
            s = np.sin(2 * angles)
            n = len(angles)
            Q = np.zeros((2, 2))
            for i in range(n):
                Q += np.array([[c[i], s[i]], [s[i], -c[i]]])
            return Q / n

        def eigenvalue(mat2):
            return np.sqrt(-np.linalg.det(mat2))

        v = self.voronoiDiagram(3)
        s = np.zeros_like(self.x)
        for i in range(self.N):
            s[i] = eigenvalue(Q_ave(anglesOf(v.neighborsOf(i))))
        v.free_memory()
        return s

    @property
    def meanS(self):
        s = self.S_field()
        mean_s = np.mean(np.abs(s))
        del s
        return mean_s

    @lru_cache(maxsize=None)
    def angleDistribution(self):
        return np.histogram(self.t, bins=bins_of_angle_dist, range=(0, np.pi))[0]

    @property
    def entropyOfAngle(self):
        f = self.angleDistribution()
        return ut.KDE_entropyOf(self.t)
