"""
config.json example:
{
    "test1": {
        "N": 1000,
        "n": 6,
        "d": 0.05,
        "phi0": 0.5,
        "potential_name": "'ScreenedCoulomb'",
        "Gammas": "[0.5, 1.0, 2.0]",
        "SIBLINGS": 3
    }
}
"""

import numpy as np
import json
import re


def sub_generate_n(gamma, n):
    d = 2 * (gamma - 1) / (n - 1)
    return {
        "N": 1000,
        "n": n,
        "d": d,
        "phi0": 0.5,
        "potential_name": "'Hertzian'",
        "Gammas": "[0.5, 1.0, 2.0]",
        "SIBLINGS": 3
    }

def sub_generate_d(gamma, d):
    n = 1 + 2 * (gamma - 1) / d
    return {
        "N": 1000,
        "n": round(n),
        "d": d,
        "phi0": 0.5,
        "potential_name": "'Hertzian'",
        "Gammas": "[1.0, 1.0, 1.0, 1.0]",
        "SIBLINGS": 4
    }


if __name__ == '__main__':
    d_gamma = 1.0 / 20
    dic = {}
    for gamma in np.arange(1 + d_gamma, 2 + d_gamma, d_gamma):
        d = 1.0 / 40
        folder = re.sub(r'\.', '_', f'g{"{:.2f}".format(gamma)}')
        dic[folder] = sub_generate_d(gamma, d)
    with open('config.json', 'w') as w:
        json.dump(dic, w)
        