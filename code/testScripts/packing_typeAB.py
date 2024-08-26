import matplotlib.pyplot as plt
import numpy as np

from .kernel_for_test import ker


def searchMinimum(func, x_min, x_max):
    xs = np.linspace(x_min, x_max, 1000)
    ys = func(xs)
    # plt.plot(xs, ys)
    # plt.show()
    index = np.argmin(ys)
    return xs[index]


def searchLocalExtreme(mode='maxima'):
    p = 1 if mode == 'minima' else -1

    def inner(func, x_min, x_max):
        xs = np.linspace(x_min, x_max, 1000)
        ys = func(xs)
        ascend = np.diff(ys) > 0
        pts = np.where(np.diff(ascend.astype(int)) == p)[0]
        return [xs[pt] for pt in pts]

    return inner


class PotentialOfGamma:
    def __init__(self, power: float, gamma: float):
        self.gamma = gamma
        d = 0.025
        n = round(1 + 2 * (gamma - 1) / d)
        n_real = 1 + 2 * (gamma - 1) / d
        if abs(n - n_real) > 1e-4:
            raise ValueError('Bad gamma value')
        ker.setEnums(2)
        ker.setPotentialPower(power)
        ker.declareRod(n, d)
        self.V = np.vectorize(ker.precisePotential)

    def U(self, x, y):
        return self.V(x, 0, 0) + self.V(0, y, 0) + 2 * self.V(x / 2, y / 2, 0)

    def Ur(self, rho):
        def inner(x):
            return self.U(x, 2 / (rho * x))

        return inner

    def Xm_A(self, rho):
        x_min = 1
        x_max = self.gamma / rho * 2
        Ms = searchLocalExtreme('maxima')(self.Ur(rho), x_min, x_max)
        M = x_max if len(Ms) == 0 else Ms[0]
        return searchMinimum(self.Ur(rho), x_min, M)

    def Xm_B(self, rho):
        x_min = 1
        x_max = self.gamma / rho * 2
        Ms = searchLocalExtreme('maxima')(self.Ur(rho), x_min, x_max)
        M = x_min if len(Ms) == 0 else Ms[-1]
        return searchMinimum(self.Ur(rho), M, x_max)

    def Xms(self, rho) -> list[list[float]]:
        x_min = 1
        x_max = self.gamma / rho * 2
        return searchLocalExtreme('minima')(self.Ur(rho), x_min, x_max)

    def Um_A(self, rho):
        return self.Ur(rho)(self.Xm_A(rho))

    def Um_B(self, rho):
        return self.Ur(rho)(self.Xm_B(rho))


po = PotentialOfGamma(1.5, 1.5)

# rhos = np.arange(1 / (2 * np.sqrt(3)), 1, 0.01)
# yss = [po.Xms(rho) for rho in np.arange(1 / (2 * np.sqrt(3)), 1, 0.01)]
# for rho, ys in zip(rhos, yss):
#     for y in ys:
#         plt.scatter(rho, y, color='blue', s=2)

rhos = np.arange(0.1, 1, 0.01)
yAs = np.vectorize(po.Um_A)(rhos)
yBs = np.vectorize(po.Um_B)(rhos)
plt.plot(rhos, yAs, rhos, yBs)

plt.show()
