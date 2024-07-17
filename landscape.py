import matplotlib.pyplot as plt
import numpy as np
from src.myio import DataSet
from src.simulator import common_simulator as cs

# for n=6, d=0.05,
# Hertzian example: qxpp
# Screened Coulomb example: 356v

"""
dataset = DataSet.loadFrom('data/356v.h5')
ss = 2e-4
n_samples = 1000
"""

dataset = DataSet.loadFrom('data/qxpp.h5')
ss = 1e-2
n_samples = 1000

state = dataset.critical(1.0)
cs.load(state)
ys = cs.simulator.energyLandscapeAlongGradient(ss, n_samples)
xs = np.linspace(0, ss, n_samples + 1)[1:]
plt.plot(xs, ys)
plt.show()
