import glob
import os
import random
import string
import threading

import matplotlib.pyplot as plt
import numpy as np
import py7zr

import singleExperiment as se
import src.utils as ut
from src.kernel import ker
from src.render import StateRenderer
from src.state import RenderSetup


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

        q = 1 - 1e-3
        self.setBoundaryScheduler(se.BoundaryScheduler.constant, lambda n, x: x * q ** n)

    def getSiblingId(self):
        return ker.getSiblingId(self.data_ptr)

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def show(self):
        handle = plt.subplots()  # handle = fig, ax
        s = self.get(record=False)
        renderer = StateRenderer(s)
        renderer.drawBoundary(handle)
        renderer.drawParticles(handle, s.angle())
        plt.show()

    def execute(self):
        for i in range(200):
            if self.density > 1.2: break
            self.compress()
            dt = self.equilibriumGD(2e5)
            s = self.get()
            its = self.iterationSteps()
            self.log(f'{i}:  G={s.max_residual_force}, E={s.energy}, nsteps={its}K , speed: {its / dt}K it/s')


class ExperimentsFixedParticleShape:
    def __init__(self, N, n, d, phi0, potential_name: str, Gammas: np.ndarray):
        self.siblings = len(Gammas)
        if self.siblings > ker.getSiblingNumber():
            raise ValueError("Too many siblings.")
        self.tasks = [TaskHandle
                      .fromCircDensity(N, n, d, phi0, Gamma)
                      for Gamma in Gammas]
        self.tasks[0].initPotential(potential_name)

    def compress(self):
        return list(map(lambda x: x.compress(), self.tasks))

    def get(self):
        return list(map(lambda x: x.get(), self.tasks))

    def serialInit(self):
        return list(map(lambda x: x.initAsDisks(), self.tasks))

    def ExperimentSerial(self):
        self.serialInit()
        return list(ut.Map('Debug')(executeTask, self.tasks))

    def ExperimentAsync(self):
        self.serialInit()
        return list(ut.Map('Release')(executeTask, self.tasks))


def executeTask(task: TaskHandle):
    return task.execute()


if __name__ == '__main__':
    SIBLINGS = 3  # must be less equal than in defs.h

    tasks = ExperimentsFixedParticleShape(
        1000, 5, 0.25, 0.4,
        "ScreenedCoulomb",
        np.linspace(0.5, 0.8, SIBLINGS, endpoint=True),
    )
    tasks.ExperimentAsync()

    packResults('new data.7z')
