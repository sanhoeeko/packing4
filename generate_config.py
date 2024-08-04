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


def sub_generate(gamma, n):
    d = 2 * (gamma - 1) / (n - 1)
    return {
        "N": 1000,
        "n": n,
        "d": d,
        "phi0": 0.5,
        "potential_name": "'ScreenedCoulomb'",
        "Gammas": "[0.5, 1.0, 2.0]",
        "SIBLINGS": 3
    }


if __name__ == '__main__':
    d_gamma = 0.125
    dic = {}
    for gamma in np.arange(1 + d_gamma, 2 + d_gamma, d_gamma):
        for n in [6, 11, 21, 41]:
            folder = re.sub(r'\.', '_', f'g{gamma}n{n}')
            dic[folder] = sub_generate(gamma, n)
    with open('config.json', 'w') as w:
        json.dump(dic, w)
        