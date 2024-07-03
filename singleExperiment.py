import src.simulator as simu
import src.utils as ut
from src.myio import DataSet


def rebootSimulation(filename: str):
    dataset = DataSet.loadFrom(filename)
    state_data = dataset.data[-1]
    simulator = state_data.makeSimulator(dataset, ut.fileNameToId(filename))
    simulator.initPotential(4)
    return simulator


def simulationExample():
    q = 1 - 1e-3
    state = simu.Simulator.fromCircDensity(
        1000, 2, 0.25, 0.4, 1.0, 'ScreenedCoulomb'
    )
    state.initAsDisks()
    state.setBoundaryScheduler(simu.BoundaryScheduler.constant, lambda n, x: x * q ** n)
    state.initPotential(4)

    for i in range(1000):
        if state.density > 1.2: break
        state.compress()
        dt = state.equilibriumGD(2e5)
        s = state.get()
        gs = state.maxGradients()
        its = len(gs)
        print(i, f'G={gs[-1]}, E={s.energy}, nsteps={its}, speed: {its / dt} it/s')


if __name__ == '__main__':
    state = rebootSimulation('j5e5.h5')
    q = 1 - 1e-2
    state.setBoundaryScheduler(simu.BoundaryScheduler.constant, lambda n, x: x * q ** n)

    for i in range(1000):
        if state.density > 1.2: break
        state.compress()
        dt = state.equilibriumGD(1e6)
        s = state.get()
        gs = state.maxGradients()
        its = len(gs)
        print(i, f'G={gs[-1]}, E={s.energy}, nsteps={its}, speed: {its / dt} it/s')
