import src.simulator as simu
import src.utils as ut
from src.myio import DataSet


def rebootSimulation(filename: str):
    dataset = DataSet.loadFrom(filename)
    state_data = dataset.data[-1]
    simulator = state_data.makeSimulator(dataset, dataset.metadata['potential'], ut.fileNameToId(filename))
    simulator.initPotential(4)
    return simulator


def simulationExample(method: str):
    """
    methods: equilibriumGD, eqLineGD, eqLBFGS
    """
    q = 1 - 1e-3
    state = simu.Simulator._fromCircDensity(
        1000, 6, 0.05, 0.5, 1.0,
        'ScreenedCoulomb', 'data'
    )
    state.initAsDisks()
    state.setBoundaryScheduler(simu.BoundaryScheduler.constant, lambda n, x: x * q ** n)
    state.initPotential(4)

    for i in range(1000):
        if state.density > 0.8: break
        state.compress()
        dt = getattr(state, method)(1e6)
        s = state.get()
        density = state.density
        its = state.iterationSteps()
        g = state.maxResidualForce()
        print(i, f'i={i}, rho={density}, G={g}, E={s.energy}, nsteps={its}K, speed: {its / dt} Kit/s')


def rebootExample():
    state = rebootSimulation('j5e5.h5')
    q = 1 - 1e-2
    state.setBoundaryScheduler(simu.BoundaryScheduler.constant, lambda n, x: x * q ** n)

    for i in range(1000):
        if state.density > 1.2: break
        state.compress()
        dt = state.equilibriumGD(1e6)
        s = state.get()
        density = state.density
        its = state.iterationSteps()
        g = state.maxResidualForce()
        print(i, f'i={i}, rho={density}, G={g}, E={s.energy}, nsteps={its}K, speed: {its / dt} Kit/s')


if __name__ == '__main__':
    simulationExample('eqLineGD')
