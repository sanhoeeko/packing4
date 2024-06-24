import glob
import os
import random
import string
import threading

import numpy as np
import py7zr

import singleExperiment as se
import src.utils as ut
from src.kernel import ker


class RandomStringGenerator:
    def __init__(self):
        self.lock = threading.Lock()

    def generate(self):
        with self.lock:
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))


_random_string_generator = RandomStringGenerator()


def randomString():
    return _random_string_generator.generate()


def collectResultFiles():
    files = []
    for pattern in ['*.h5', '*.log.txt']:
        files.extend(glob.glob(pattern))
    return [os.path.abspath(file) for file in files]


def clearResults():
    return any([os.remove(f) for f in collectResultFiles()])


def packResults(archive_name: str):
    with py7zr.SevenZipFile(archive_name, mode='w') as z:
        for file in collectResultFiles():
            z.write(file, arcname=os.path.basename(file))
    return clearResults()


class TaskHandle(se.StateHandle):
    def __init__(self, N, n, d, boundary_a, boundary_b):
        """
        potential_name: Hertzian | ScreenedCoulomb
        """
        self.id = randomString()
        self.log_file = f'{self.id}.log.txt'
        open(self.log_file, 'w')  # create the file
        super().__init__(N, n, d, boundary_a, boundary_b, self.id)

    def getSiblingId(self):
        return ker.getSiblingId(self.data_ptr)

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def diagnose(self):
        state = self.get()

        # check if there is nan
        if np.isnan(state.xyt).any():
            raise ValueError("Nan detected in a state.")

        # check if there is particle outside the boundary / unexpected value like 0xCC
        legal = np.logical_and(-self.A < state.x < self.A, -self.B < state.y < self.B)
        if not legal.all():
            for i in range(len(legal)):
                if not legal[i]:
                    print(f"Abnormal coordinates: {state.xyt[i]}")
            raise ValueError

    def execute(self):
        q = 1 - 1e-3
        self.initAsDisks()
        self.setBoundaryScheduler(se.BoundaryScheduler.constant, lambda n, x: x * q ** n)

        for i in range(100):
            if self.density > 1.2: break
            self.compress()
            dt = self.equilibriumGD(2e5)
            s = self.get()
            gs = self.maxGradients()
            its = len(gs)
            self.log(f'{i}:  G={gs[-1]}, E={s.energy}, nsteps={its}, speed: {its / dt} it/s')


def ExperimentsFixed_gamma(N, n, d, phi0, potential_name: str, Gammas: np.ndarray):
    tasks = [TaskHandle
             .fromCircDensity(N, n, d, phi0, Gamma)
             for Gamma in Gammas]
    tasks[0].initPotential(potential_name)
    return tasks


def executeTask(task: TaskHandle):
    return task.execute()


if __name__ == '__main__':
    SIBLINGS = 4  # must be less equal than in defs.h

    tasks = ExperimentsFixed_gamma(
        1000, 2, 0.25, 0.4,
        "ScreenedCoulomb",
        np.linspace(1, 2, SIBLINGS, endpoint=True),
    )

    # Redirect stderr (python error, not c++ error) to a file
    # sys.stderr = open('error.log.txt', 'w')
    results = list(ut.Map('Release')(executeTask, tasks))
    # sys.stderr.close()
    # sys.stderr = sys.__stderr__

    packResults('arc.7z')
