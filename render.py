from functools import lru_cache

import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay

import art
from graph import Graph

my_colors = ['floralWhite', 'lemonchiffon', 'wheat', 'lightsalmon', 'coral', 'crimson',
             'paleturquoise', 'blue', 'teal', 'seagreen', 'green']


class State:
    def __init__(self, N, n, d, boundary_a, boundary_b, configuration: np.ndarray):
        self.N = N
        self.n, self.d = n, d
        self.A, self.B = boundary_a, boundary_b
        self.a, self.b = 1, 1 / (1 + (n - 1) * d / 2)
        self.xyt = configuration

    @property
    def x(self): return self.xyt[:, 0]

    @property
    def y(self): return self.xyt[:, 1]

    @property
    def t(self): return self.xyt[:, 2] % np.pi

    @lru_cache(maxsize=None)
    def render(self):
        return StateRenderer(self)

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


class StateRenderer(State):
    def __init__(self, state):
        self.__dict__ = state.__dict__
        self.handle = self.handle = plt.subplots()  # (ax, fig)

    def drawBoundary(self):
        fig, ax = self.handle
        ellipse = patches.Ellipse((0, 0), width=2 * self.A, height=2 * self.B, fill=False)
        ax.add_artist(ellipse)
        return self.handle


    def drawParticles(self, colors=None):
        """
        colors: can be either an array or a function
        """
        fig, ax = self.handle
        cmap = 'viridis'
        norm = None
        if colors is None:  # plot angle as default
            c = self.t
            cmap = 'hsv'
            norm = mcolors.Normalize(vmin=0, vmax=np.pi)
        elif isinstance(colors, np.ndarray):
            assert colors.shape == self.x.shape
            c = colors
            if np.issubdtype(colors.dtype, np.integer):
                cmap = mcolors.ListedColormap(my_colors)
                norm = mcolors.Normalize(vmin=0, vmax=len(my_colors))
                if np.any(colors < 0) or np.any(colors > len(my_colors)):
                    raise ValueError("Integer data out of range")
        else:
            raise TypeError

        # Create a list to hold the patches
        ellipses = []

        # For each point in the data, create a custom patch (ellipse) and add it to the list
        for xi, yi, ti in zip(self.x, self.y, np.degrees(self.t)):
            ellipse = patches.Ellipse((xi, yi), width=self.a, height=self.b, angle=ti)
            # ellipse = art.Capsule((xi, yi), width=self.a, height=self.b, angle=ti)
            ellipses.append(ellipse)

        # Create a collection with the ellipses and add it to the axes
        col = collections.PatchCollection(ellipses, array=c, norm=norm, cmap=cmap)
        ax.add_collection(col)

        # Set the limits of the plot
        ax.set_xlim(-self.A - 1, self.A + 1)
        ax.set_ylim(-self.B - 1, self.B + 1)
        ax.set_aspect('equal')

        # Create an axes for the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add a colorbar
        fig.colorbar(col, cax=cax)
        return self.handle
