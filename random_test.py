from math import pi

import matplotlib.pyplot as plt
import numpy as np

from kernel_for_test import ker


class TestResult:
    def __init__(self, args: tuple, dif: np.ndarray):
        self.args = args
        self.dif = dif
        self.mean_e = np.mean(dif)
        self.max_e = np.max(dif)

    def __repr__(self):
        return f"mean error: {self.mean_e}, max error: {self.max_e}"

    def show(self, expr=None, *num_list_of_args):
        if expr is None:
            plt.scatter(self.args[num_list_of_args[0]], self.dif)
        else:
            x = np.vectorize(expr)(*[self.args[i] for i in num_list_of_args])
            plt.scatter(x, self.dif)


def fastPotentialTest(m):
    """
    Test if the grid potential agrees with the directly calculated potential.
    m: number of sample points
    """
    x = np.random.uniform(-2 * a, 2 * a, (m,))
    y = np.random.uniform(-a - b, a + b, (m,))
    t = np.random.uniform(0, 2 * pi, (m,))
    v = np.vectorize(ker.fastPotential)(x, y, t)
    v_ref = np.vectorize(ker.precisePotential)(x, y, t)
    dif = np.abs(v - v_ref)
    return TestResult((x, y, t), dif)


def potentialTest(m):
    """
    Test if the interpolated potential agrees with the directly calculated potential.
    m: number of sample points
    """
    x = np.random.uniform(-2 * a, 2 * a, (m,))
    y = np.random.uniform(-a - b, a + b, (m,))
    t = np.random.uniform(0, 2 * pi, (m,))
    v = np.vectorize(ker.interpolatePotential)(x, y, t)
    v_ref = np.vectorize(ker.precisePotential)(x, y, t)
    dif = np.abs(v - v_ref)
    return TestResult((x, y, t), dif)


def gradientTest(m):
    """
    Test if the interpolated gradient agrees with the directly calculated gradient.
    m: number of sample points
    """
    x = np.random.uniform(-2 * a, 2 * a, (m,))
    y = np.random.uniform(-a - b, a + b, (m,))
    t1 = np.random.uniform(0, 2 * pi, (m,))
    t2 = np.random.uniform(0, 2 * pi, (m,))

    g = np.zeros((m, 6))
    g_ref = np.zeros((m, 6))
    for i in range(m):
        try:
            g[i] = ker.gradientTest(x[i], y[i], t1[i], t2[i])
            g_ref[i] = ker.gradientReference(x[i], y[i], t1[i], t2[i])
        except:
            g_should = ker.gradientReference(x[i], y[i], t1[i], t2[i])
            print(f"(x={x[i]}, y={y[i]}, t1={t1[i]}, t2={t2[i]}) should be {g_should}")

    dif = np.sum(np.abs(g - g_ref) ** 2, axis=1)
    return TestResult((x, y, t1, t2), dif)


n = 2
d = 0.25
ker.setRod(n, d)
a, b = 1, 1 / (1 + (n - 1) * d / 2)

m = 10000


res = gradientTest(m)
print(res)
# res.show(lambda x, y: x - y, 2, 3)
res.show(lambda x, y: np.sqrt(x**2 + y**2), 0, 1)
plt.show()