import random
import string
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import singleExperiment as se


class RandomStringGenerator:
    def __init__(self):
        self.lock = threading.Lock()

    def generate(self):
        with self.lock:
            return ''.join(random.choices(string.ascii_letters + string.digits, k=4))


_random_string_generator = RandomStringGenerator()


def randomString():
    return _random_string_generator.generate()


class TaskHandle(se.StateHandle):
    def __init__(self, N, n, d, boundary_a, boundary_b):
        self.id = randomString()
        self.log_file = f'{self.id}.log.txt'
        super().__init__(N, n, d, boundary_a, boundary_b)

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message)

    def execute(self):

        # TODO: to be abstracted

        q = 1 - 1e-3
        self.initAsDisks()
        n, x = se.BoundaryScheduler.n, se.BoundaryScheduler.x
        self.setBoundaryScheduler(x, x * q ** n)
        self.initPotential('ScreenedCoulomb')
        
        for i in range(1000):
            if self.density > 1.2: break
            self.compress()
            dt = self.equilibriumGD(2e5)
            s = self.get()
            gs = self.maxGradients()
            its = len(gs)
            self.log(f'{i}:  G={gs[-1]}, E={s.energy}, nsteps={its}, speed: {its / dt} it/s')


def ExperimentsFixed_gamma(N, n, d, phi0, Gammas: np.ndarray):
    return [TaskHandle.fromCircDensity(N, n, d, phi0, Gamma) for Gamma in Gammas]


def executeTask(task: TaskHandle):
    return task.execute()


if __name__ == '__main__':
    SIBLINGS = 4  # must be the same as in defs.h
    with ThreadPoolExecutor(max_workers=SIBLINGS) as executor:
        tasks = ExperimentsFixed_gamma(1000, 2, 0.25, 0.5, 
                                       np.linspace(1, 2, SIBLINGS, endpoint=True)
                                       )
        executor.map(executeTask, tasks)
