import src.simulator as simu

if __name__ == '__main__':
    q = 1 - 1e-3
    state = simu.Simulator.fromCircDensity(1000, 2, 0.25, 0.4, 1.0)
    state.initAsDisks()
    state.setBoundaryScheduler(simu.BoundaryScheduler.constant, lambda n, x: x * q ** n)
    state.initPotential('ScreenedCoulomb', 4)

    for i in range(1000):
        if state.density > 1.2: break
        state.compress()
        dt = state.equilibriumGD(2e5)
        s = state.get()
        gs = state.maxGradients()
        its = len(gs)
        print(i, f'G={gs[-1]}, E={s.energy}, nsteps={its}, speed: {its / dt} it/s')
