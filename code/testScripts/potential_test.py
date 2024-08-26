import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from .kernel_for_test import ker


def dynamicVisualize(n_thetas, x, y, theta, v):
    # Create a new figure for the image and the slider
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Create an axis for the slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])

    # Create a slider
    slider = Slider(ax_slider, 'Theta', 0, np.pi - 0.01, valinit=0, valstep=np.pi / n_thetas)

    # Initial image (theta = 0)
    im = ax.imshow(v[:, :, 0], extent=[np.min(x), np.max(x), np.min(y), np.max(y)], aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax)

    # Update function for the slider
    def update(val):
        # Get the current theta value from the slider
        theta_val = slider.val

        # Update the image for the current theta value
        im.set_data(v[:, :, int(theta_val / np.pi * n_thetas)])

        # Redraw the figure
        ax.set_aspect('equal')
        fig.canvas.draw_idle()

    # Connect the update function to the slider
    slider.on_changed(update)
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    n = 6
    d = 0.05
    a = 1
    b = 1 / (1 + (n - 1) * d / 2)
    ker.setEnums(0)
    ker.setRod(n, d)
    x = np.arange(-2 * a, 2 * a, 0.01)
    y = np.arange(-a - b, a + b, 0.01)
    t = np.arange(0, np.pi, 0.1)
    n_theta = len(t)
    X, Y, T = np.meshgrid(x, y, t)
    V = np.vectorize(ker.interpolatePotential)(X, Y, T)
    V[V < 1e-6] = np.nan
    dynamicVisualize(n_theta, X, Y, T, V)
