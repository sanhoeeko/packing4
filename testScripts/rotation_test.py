from math import sin, cos

import numpy as np

from .kernel_for_test import ker


def rotate(angle, tup: tuple):
    x, y, t1, t2 = tup
    return x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle), t1 + angle, t2 + angle


def rotationTest():
    x0, y0, t10, t20 = 1, 1, np.pi / 3, 2 * np.pi / 3
    arr = []
    for a in np.arange(0, 2 * np.pi, 0.1):
        arr.append(ker.gradientTest(*rotate(a, (x0, y0, t10, t20))))
    return np.array(arr)


if __name__ == '__main__':
    n = 2
    d = 1
    ker.setEnums(1)
    ker.setRod(n, d)
    mat = rotationTest()
    