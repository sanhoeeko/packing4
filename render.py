import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class State:
    def __init__(self, N, boundary_a, boundary_b, configuration: np.ndarray):
        self.N = N
        self.A, self.B = boundary_a, boundary_b
        self.a, self.b = 1, 1
        self.xyt = configuration.reshape((self.N, 3))

    @property
    def x(self): return self.xyt[:, 0]

    @property
    def y(self): return self.xyt[:, 1]

    @property
    def t(self): return self.xyt[:, 2]

    def render(self):
        return StateRenderer(self)


class StateRenderer(State):
    def __init__(self, state):
        self.__dict__ = state.__dict__
        self.handle = self.handle = plt.subplots()  # (ax, fig)

    def drawBoundary(self):
        fig, ax = self.handle
        ellipse = patches.Ellipse((0, 0), width=2 * self.A, height=2 * self.B, fill=False)
        ax.add_artist(ellipse)
        return self.handle

    def drawParticles(self, color_function=None):
        fig, ax = self.handle
        if color_function is None:
            c = np.zeros_like(self.x)

        # Create a list to hold the patches
        ellipses = []

        # For each point in the data, create a custom patch (ellipse) and add it to the list
        for xi, yi, ti in zip(self.x, self.y, np.degrees(self.t)):
            ellipse = patches.Ellipse((xi, yi), width=self.a, height=self.b, angle=ti)
            ellipses.append(ellipse)

        # Create a collection with the ellipses and add it to the axes
        col = collections.PatchCollection(ellipses, array=c, cmap='viridis')
        ax.add_collection(col)

        # Set the limits of the plot
        ax.set_xlim(np.min(self.x) - 1, np.max(self.x) + 1)
        ax.set_ylim(np.min(self.y) - 1, np.max(self.y) + 1)
        ax.set_aspect('equal')

        # Add a colorbar
        fig.colorbar(col, ax=ax)
        return self.handle