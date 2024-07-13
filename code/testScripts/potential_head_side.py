import matplotlib.pyplot as plt
import numpy as np

from .kernel_for_test import ker


def Vhh(a, b):
    """
    Head-to-head interaction
    """
    return np.vectorize(
        lambda r: ker.interpolatePotential(r * b + 2 * (a - b), 0, 0)
    )


def Vss(a, b):
    """
    Side-to-side interaction
    """
    return np.vectorize(
        lambda r: ker.interpolatePotential(0, r * b, 0)
    )


def Vhs(a, b):
    """
    Head-to-side interaction
    """
    return np.vectorize(
        lambda r: ker.interpolatePotential(r * b + (a - b), 0, np.pi / 2)
    )


if __name__ == '__main__':
    n = 6
    d = 0.05
    a = 1
    b = 1 / (1 + (n - 1) * d / 2)
    ker.setEnums(0)
    ker.setRod(n, d)
    rs = np.arange(0.01, 2, 0.001)
    vhh = Vhh(a, b)(rs)
    vhs = Vhs(a, b)(rs)
    vss = Vss(a, b)(rs)
    plt.rcParams.update({'font.size': '22'})
    plt.plot(rs, vhh, rs, vhs, rs, vss)
    plt.legend(['head-to-head', 'head-to-side', 'side-to-side'])
    # ratio_hs = np.mean(vhs) / np.mean(vhh)
    # ratio_ss = np.mean(vss) / np.mean(vhh)
    # print(ratio_hs, ratio_ss)
    plt.show()
