import functools
import types
from functools import lru_cache

import numpy as np
from scipy.spatial import Delaunay

from graph import Graph


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
    def x(self): return self.xyt[:, 0]

    @property
    def y(self): return self.xyt[:, 1]

    @property
    def t(self): return self.xyt[:, 2] % np.pi

    @property
    def metadata(self):
        return {
            'id': self.id,
            'A': self.A,
            'B': self.B,
        }

    @classmethod
    def load(cls, configuration, metameta: dict, metadata: dict):
        obj = cls(
            metadata['id'],
            metameta['N'], metameta['n'], metameta['d'],
            metadata['A'], metadata['B'],
            configuration
        )
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
    def voronoi(self):
        points = self.toSites()  # input of Delaunay is (n_point, n_dim)
        delaunay = Delaunay(points)
        voro_graph = Graph(len(points)).from_delaunay(delaunay.simplices)
        return voro_graph.merge(self.N)
