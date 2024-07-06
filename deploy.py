import glob
import os
import random
import string
import threading

import matplotlib.pyplot as plt
import numpy as np
import py7zr

import src.simulator as simu
import src.utils as ut
from src.kernel import ker
from src.render import StateRenderer


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


class TaskHandle(simu.Simulator):
    def __init__(self, N, n, d, boundary_a, boundary_b, potential_name: str):
        """
        potential_name: Hertzian | ScreenedCoulomb
        """
        Id = randomString()
        obj = simu.Simulator.createState(N, n, d, boundary_a, boundary_b, potential_name, Id)
        self.__dict__ = obj.__dict__

        self.id = Id
        self.log_file = f'{self.id}.log.txt'
        open(self.log_file, 'w')  # create the file

        q = 1 - 1e-3
        self.setBoundaryScheduler(simu.BoundaryScheduler.constant, lambda n, x: x * q ** n)

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
        try:
            self.initAsDisks()
            for i in range(10000):
                if self.density > 1.2: break
                self.compress()
                dt = self.equilibriumGD(1e6)
                s = self.get()
                density = self.density
                its = self.iterationSteps()
                g = self.maxResidualForce()
                print(i, f'i={i}, rho={density}, G={g}, E={s.energy}, nsteps={its}K, speed: {its / dt} Kit/s')
        except Exception as e:
            print("An exception occurred!\n", e)
            print(f"In sibling {self.getSiblingId()}, ID: {self.id}")


class ExperimentsFixedParticleShape:
    def __init__(self, N, n, d, phi0, potential_name: str, Gammas: np.ndarray, workers=1):
        self.siblings = len(Gammas)
        self.cores = workers
        self.potential_name = potential_name
        self.n, self.d = n, d
        self.tasks = [TaskHandle
                      .fromCircDensity(N, n, d, phi0, Gamma, potential_name)
                      for Gamma in Gammas]
        self.initPotential()

    def initPotential(self):
        self.tasks[0].initPotential(self.siblings * self.cores)

    def compress(self):
        return list(map(lambda x: x.compress(), self.tasks))

    def get(self):
        return list(map(lambda x: x.get(), self.tasks))

    def ExperimentSerial(self):
        return list(ut.Map('Debug')(executeTask, self.tasks))

    def ExperimentAsync(self):
        return list(ut.Map('Release')(executeTask, self.tasks))


def executeTask(task: TaskHandle):
    return task.execute()


if __name__ == '__main__':
    CORES = 2
    SIBLINGS = 1  # must be less equal than in defs.h

    with ut.MyThreadRecord('gengjie', CORES * SIBLINGS):
        tasks = ExperimentsFixedParticleShape(
            1000, 2, 0.25, 0.5,
            "ScreenedCoulomb",
            np.linspace(0.5, 0.8, SIBLINGS, endpoint=True),
            CORES,
        )
        tasks.ExperimentAsync()

        packResults('results.7z')
