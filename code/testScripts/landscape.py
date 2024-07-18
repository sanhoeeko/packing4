import matplotlib.pyplot as plt
import numpy as np

from src.myio import DataSet
from src.simulator import common_simulator as cs


class Normalizer:
    def __init__(self, arr: np.ndarray):
        self.a = np.min(arr)
        self.b = np.max(arr)

    def normalize(self, x):
        return (x - self.b) / (self.a - self.b)

    def inverse(self, y):
        return (self.a - self.b) * y + self.b


def landscape1d(state, ss, n_samples):
    cs.load(state)
    ys = cs.simulator.energyLandscapeAlongGradient(ss, n_samples)
    xs = np.linspace(0, ss, n_samples + 1)[1:]
    plt.plot(xs, ys)
    plt.show()


def landscape1dAndFit(state, ss, n_samples, n_points, deg=4):
    cs.load(state)
    ys = cs.simulator.energyLandscapeAlongGradient(ss, n_samples)
    xs = np.linspace(0, ss, n_samples + 1)[1:]

    fitted_func = np.poly1d(np.polyfit(xs, ys, deg))

    ys = cs.simulator.energyLandscapeAlongGradient(ss, n_points)
    xs = np.linspace(0, ss, n_points + 1)[1:]
    y_pred = np.vectorize(fitted_func)(xs)

    plt.scatter(xs, ys, marker='.')
    plt.plot(xs, y_pred, c='orange')
    plt.show()


def lineSearch1dTest(state, max_stepsize, n_samples):
    cs.load(state)
    sc = cs.simulator.ERoot(max_stepsize)
    best_ss = cs.simulator.bestStepSize(max_stepsize)

    ys = cs.simulator.energyLandscapeAlongGradient(sc, n_samples)
    xs = np.linspace(0, sc, n_samples + 1)[1:]
    fitted_func = np.poly1d(np.polyfit(xs, ys, deg=3))
    y_pred = np.vectorize(fitted_func)(xs)

    plt.scatter(xs, ys, marker='.')
    plt.plot(xs, y_pred, c='orange')
    plt.axvline(x=best_ss, color='red')
    plt.show()


def landscape2d(state, ss, n_samples):
    cs.load(state)
    Z = cs.simulator.energyLandscapeOnGradientSections(ss, n_samples)
    xs = np.linspace(0, ss, n_samples + 1)[1:]
    ys = np.linspace(-ss, ss, 2 * n_samples + 1)
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot_surface(X, Y, Z, cmap=plt.cm.spring)
    plt.show()


# for n=6, d=0.05,
# Hertzian example: qxpp
# Screened Coulomb example: 356v

dataset = DataSet.loadFrom('data/356v.h5')
state = dataset.critical(10000)
ss = 6e-3
n_samples = 10
lineSearch1dTest(state, 1e-3, 100)
