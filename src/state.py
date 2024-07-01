from functools import lru_cache

import numpy as np
from scipy.spatial import Delaunay

from src.graph import Graph


class RenderSetup:
    def __init__(self, colors: np.ndarray, style: str):
        self.colors = colors
        self.style = style


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
            up_meta['N'], up_meta['n'], up_meta['d'],
            metadata['A'], metadata['B'],
            configuration,
            others={
                'energy_curve': metadata['energy_curve'],
                'energy': metadata['energy'],
                'max_residual_force': metadata['max_residual_force'],
            }
        )
        return obj

    def makeSimulator(self, dataset, data_name: str):

        from src.simulator import Simulator
        obj = Simulator(self.N, self.n, self.d, self.A, self.B,
                        potential_name=dataset.metadata['potential'], data_name=data_name)

        obj.dataset = dataset

        # pretreatment of data: map (N, 3) to (N, 4)
        data = np.hstack([self.xyt, np.zeros((self.N, 1))]).reshape(-1)
        obj.loadDataToKernel(data)
        return obj

    @lru_cache(maxsize=None)
    def toSites(self):
        """
        convert each rod to disks
        """
        xy = np.array([self.x, self.y]).T
        uxy = np.array([np.cos(self.t), np.sin(self.t)]).T * self.d
        n_shift = -(self.n - 1) / 2.0
        xys = [xy + (k + n_shift) * uxy for k in range(0, self.n)]
        return np.vstack(xys)

    @lru_cache(maxsize=None)
    def voronoiDiagram(self):
        points = self.toSites()  # input of Delaunay is (n_point, n_dim)
        delaunay = Delaunay(points)
        voro_graph = Graph(len(points)).from_delaunay(delaunay.vertex_neighbor_vertices)
        return voro_graph.merge(self.N)

    # analysis

    @property
    def globalS(self):
        EiPhi = np.mean(np.exp(2j * self.t))
        return abs(EiPhi), np.angle(EiPhi) / 2
        # return np.real(EiPhi), np.imag(EiPhi)

    # visualization

    def voronoi(self) -> RenderSetup:
        """
        for visualization
        """
        return RenderSetup(self.voronoiDiagram().neighborNums(), 'voronoi')

    def angle(self) -> RenderSetup:
        """
        for visualization
        """
        return RenderSetup(self.t, 'angle')
