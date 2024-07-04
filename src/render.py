import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import art
from .state import State, RenderSetup

my_colors = ['floralWhite', 'lemonchiffon', 'wheat', 'lightsalmon', 'coral', 'crimson',
             'paleturquoise', 'blue', 'teal', 'seagreen', 'green']


class StateRenderer(State):
    def __init__(self, state):
        self.__dict__ = state.__dict__

    def plotEnergyCurve(self, handle):
        fig, ax = handle
        ax.plot(self.energy_curve)
        ax.set_xlabel('×10³ iterations')
        ax.set_ylabel('Energy')

    def drawBoundary(self, handle):
        fig, ax = handle
        ellipse = patches.Ellipse((0, 0), width=2 * self.A, height=2 * self.B, fill=False)
        ax.add_artist(ellipse)
        return handle

    def drawParticles(self, handle, setup: RenderSetup):
        """
        colors: can be either an array or a function
        """
        fig, ax = handle
        cmap, norm = None, None
        if setup.style == 'normal':
            cmap = 'viridis'
            norm = None
        elif setup.style == 'angle':
            cmap = 'hsv'
            norm = mcolors.Normalize(vmin=0, vmax=np.pi)
        elif setup.style == 'voronoi':
            cmap = mcolors.ListedColormap(my_colors)
            norm = mcolors.Normalize(vmin=0, vmax=len(my_colors))
            if np.any(setup.colors < 0) or np.any(setup.colors > len(my_colors)):
                raise ValueError("Integer data out of range.")
        else:
            raise TypeError("Unknown visualization style.")

        # Create a list to hold the patches
        ellipses = []

        # For each point in the data, create a custom patch (ellipse) and add it to the list
        if setup.real_size:
            aa, bb = np.sqrt(2) * self.a, np.sqrt(2) * self.b
            alpha = 0.7
        else:
            aa, bb = self.a, self.b
            alpha = 1.0
        for xi, yi, ti in zip(self.x, self.y, np.degrees(self.t)):
            # ellipse = patches.Ellipse((xi, yi), width=self.a, height=self.b, angle=ti)
            ellipse = art.Capsule((xi, yi), width=aa, height=bb, angle=ti)
            ellipses.append(ellipse)

        # Create a collection with the ellipses and add it to the axes
        col = collections.PatchCollection(ellipses, array=setup.colors, norm=norm, cmap=cmap, alpha=alpha)
        ax.add_collection(col)

        # Set the limits of the plot
        ax.set_xlim(-self.A - 1, self.A + 1)
        ax.set_ylim(-self.B - 1, self.B + 1)
        ax.set_aspect('equal')

        # Create an axes for the color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add a color bar
        fig.colorbar(col, cax=cax)
        return handle
