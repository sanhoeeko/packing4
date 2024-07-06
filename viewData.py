import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.art as art
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
        self.legend = None
        self.sort('Gamma0')
        self.initDensityCurveTemplates()

    def name(self, Id: str) -> DataSet:
        return ut.findFirst(self.datasets, lambda x: x.id == Id)

    def sort(self, prop: str):
        self.datasets.sort(key=lambda x: getattr(x, prop))
        self.legend = list(map(lambda x: str(getattr(x, prop)), self.datasets))
        self.print()
        return self

    def print(self):
        # re-generate the abstract
        self.abstract = pd.DataFrame()
        for d in self.datasets:
            self.abstract = pd.concat([self.abstract, d.toDataFrame()], ignore_index=True)
        print(self.abstract)

    def render(self, Id: str, render_mode: str, *args):
        real = True if len(args) > 0 and args[0] == 'real' else False
        InteractiveViewer(self.name(Id), RenderPipe(getattr(StateRenderer, render_mode)), real).show()

    def show(self, Id: str):
        self.render(Id, 'angle')

    def dispStateTemplate(self, state, method, real=False):
        sr = StateRenderer(state)
        handle = plt.subplots()
        sr.drawBoundary(handle)
        sr.drawParticles(handle, getattr(state, method)(real))
        plt.show()

    def angle(self, state):
        self.dispStateTemplate(state, 'angle')

    def voronoi(self, state):
        self.dispStateTemplate(state, 'voronoi')

    def density(self, Id: str, density: float) -> State:
        density = float(density)
        rhos = self.name(Id).rhos
        dr = np.abs(rhos - density)
        idx = np.argmin(dr)
        print('Find at density:', rhos[idx])
        return self.name(Id).data[idx]

    def curveVsDensityTemplate(self, prop: str):
        def Y(Id: str):
            y = self.name(Id).curveTemplate(prop)
            rhos = self.name(Id).rhos
            plt.plot(rhos, y)
            plt.xlabel('number density')
            plt.ylabel(prop)
            plt.show()

        return Y

    def initDensityCurveTemplates(self):
        for prop in ['energy', 'logE', 'residualForce', 'globalS', 'globalSx']:
            setattr(self, prop, self.curveVsDensityTemplate(prop))

    def all(self, prop):
        if self.legend is None:
            raise ValueError('Cannot plot all until the data is sorted!')
        for d in self.datasets:
            y = d.curveTemplate(prop)
            plt.plot(d.rhos, y)
        plt.xlabel('number density')
        plt.ylabel(prop)
        plt.legend(self.legend)
        plt.show()

    def critical(self, Id: str, energy_threshold: str) -> State:
        energy_threshold = float(energy_threshold)
        es = self.name(Id).curveTemplate('energy')
        idx = ut.findFirstOfLastSubsequence(es, lambda x: x > energy_threshold)
        print('Find at density:', self.name(Id).rhos[idx])
        return self.name(Id).data[idx]

    def desCurve(self, Id: str):
        curves = self.name(Id).descentCurves
        curves = [cur / cur[0] - 1 if len(cur) > 1 else None for cur in curves]
        curves = list(filter(lambda x: x is not None, curves))
        art.plotListOfArray(curves)

    def distanceCurve(self, Id: str):
        d = self.name(Id)
        plt.plot(d.rhos[1:], d.distanceCurve)
        plt.xlabel('number density')
        plt.ylabel('distance between two states')
        plt.show()


def collectResultFiles(path: str):
    files = []
    for pattern in [f'{path}/*.h5']:
        files.extend(glob.glob(pattern))
    return [os.path.abspath(file) for file in files]


def loadAll():
    data_files = collectResultFiles(target_dir)
    ds = ut.Map('Release')(DataSet.loadFrom, data_files)
    return DataViewer(ds)


def parse(cmd: str):
    parser = ut.CommandQueue(viewer)
    tokens = cmd.split()
    for token in tokens:
        parser.push(token)
    return parser.result()


if __name__ == '__main__':
    target_dir = input('data folder name: ')
    viewer = loadAll()
    while True:
        try:
            parse(input('>>>'))
        except Exception as e:
            print(e)
