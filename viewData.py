import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.utils as ut
from analysis import InteractiveViewer, RenderPipe
from src.myio import DataSet
from src.render import StateRenderer
from src.state import State

target_dir = None
viewer = None


class DataViewer:
    def __init__(self, datasets: list[DataSet]):
        self.abstract = pd.DataFrame()
        self.datasets = datasets

    def name(self, Id: str) -> DataSet:
        return ut.findFirst(self.datasets, lambda x: x.id == Id)

    def print(self):
        # re-generate the abstract
        self.abstract = pd.DataFrame()
        for d in self.datasets:
            self.abstract = pd.concat([self.abstract, d.toDataFrame()], ignore_index=True)
        print(self.abstract)

    def show(self, Id: str):
        InteractiveViewer(self.name(Id), RenderPipe(StateRenderer.angle)).show()

    def dispStateTemplate(self, state, method):
        sr = StateRenderer(state)
        handle = plt.subplots()
        sr.drawBoundary(handle)
        sr.drawParticles(handle, getattr(state, method)())
        plt.show()

    def disp(self, state):
        self.dispStateTemplate(state, 'angle')

    def voronoi(self, state):
        self.dispStateTemplate(state, 'voronoi')

    def density(self, density: float, Id: str) -> State:
        density = float(density)
        rhos = np.array([data.rho for data in self.name(Id).data])
        dr = np.abs(rhos - density)
        idx = np.argmin(dr)
        print('Find at density:', rhos[idx])
        return self.name(Id).data[idx]

    def curve(self, state):
        sr = StateRenderer(state)
        handle = plt.subplots()
        sr.plotEnergyCurve(handle)
        plt.show()

    def logE(self, Id: str):
        y = np.log(self.name(Id).curveTemplate('energy'))
        rhos = self.name(Id).rhos
        plt.plot(rhos, y)
        plt.xlabel('number density')
        plt.ylabel('log energy')
        plt.show()

    def globalS(self, Id: str):
        y = self.name(Id).curveTemplate('globalS')
        rhos = self.name(Id).rhos
        plt.plot(rhos, y)
        plt.xlabel('number density')
        plt.ylabel('global S')
        plt.show()

    def sort(self, prop: str):
        self.datasets.sort(key=lambda x: getattr(x, prop))
        self.print()
        return self


def collectResultFiles(path: str):
    files = []
    for pattern in [f'{path}/*.h5']:
        files.extend(glob.glob(pattern))
    return [os.path.abspath(file) for file in files]


def loadAll():
    data_files = collectResultFiles(target_dir)
    ds = ut.Map('Debug')(DataSet.loadFrom, data_files)
    return DataViewer(ds)


def parse(cmd: str):
    """
    Format: command [arg] [-kwarg] [ | pipe command]
    """

    def subTokenize(cmd: str):
        # Split the command by space
        tokens = cmd.split()

        # Initialize the variables
        command = None
        args = []
        kwargs = {}

        # Iterate over the tokens
        for token in tokens:
            if not command:
                # The first token is the command
                command = token.strip()
            elif token.startswith('-'):
                # If the token starts with '-', it's a kwarg
                key, value = token[1:].split('=')
                kwargs[key.strip()] = value.strip()
            else:
                # Otherwise, it's an arg
                args.append(token.strip())

        return command, args, kwargs

    tokens = list(map(subTokenize, cmd.split('|')))
    command, args, kwargs = tokens[0]
    result = getattr(viewer, command)(*args, **kwargs)
    for i in range(1, len(tokens)):
        command, args, kwargs = tokens[i]
        result = getattr(viewer, command)(result, *args, **kwargs)
    return result


if __name__ == '__main__':
    target_dir = input('data folder name: ')
    viewer = loadAll()
    viewer.print()
    while True:
        try:
            parse(input('>>>'))
        except Exception as e:
            print(e)
