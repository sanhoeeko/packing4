import matplotlib.pyplot as plt

from kernel import ker
from render import State


class StateGetter:
    def __init__(self, N, n, d, boundary_a, boundary_b):
        self.N = N
        self.n, self.d = n, d
        self.A, self.B = boundary_a, boundary_b
        self.data_ptr = ker.createState(N, boundary_a, boundary_b)

    def get(self):
        return State(self.N, self.n, self.d, self.A, self.B, ker.getStateData(self.data_ptr, self.N))

    def initPotential(self):
        return ker.setRod(self.n, self.d)

    def maxGradients(self):
        return ker.getStateMaxGradients(self.data_ptr)

    def residualForce(self):
        return ker.getStateResidualForce(self.data_ptr, self.N)

    def initAsDisks(self):
        return ker.initStateAsDisks(self.data_ptr)

    def setBoundary(self, boundary_a, boundary_b):
        self.A, self.B = boundary_a, boundary_b
        return ker.setBoundary(self.data_ptr, boundary_a, boundary_b)

    def equilibriumGD(self):
        return ker.equilibriumGD(self.data_ptr)


if __name__ == '__main__':
    state = StateGetter(1000, 2, 0.25, 40, 40)
    state.initAsDisks()
    state.initPotential()
    for i in range(100):
        state.setBoundary(40, 40 - 0.1 * i)
        state.equilibriumGD()
        gs = state.maxGradients()
        print(i, gs[-1])

    s = state.get()
    r = s.render()
    r.drawBoundary()
    r.drawParticles()
    plt.show()
