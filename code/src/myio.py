import os.path
from functools import lru_cache
from typing import Union

import h5py
import numpy as np
import pandas as pd

import src.utils as ut
from src.state import State


def isH5FileEmpty(file_path: str):
    if not os.path.exists(file_path):
        return True
    with h5py.File(file_path, 'r') as f:
        return len(f.attrs) == 0


def countH5Groups(file_path):
    with h5py.File(file_path, 'r') as f:
        return sum(1 for _ in f.values() if isinstance(_, h5py.Group))


class DataSet:
    def __init__(self, filename: str, metadata: dict):
        self.filename = filename
        self.metadata: dict = metadata  # See State.metadata
        self._data: list[State] = []
        # save metadata first, even if there is no data
        # if it is not in the reading mode and not in the appending mode
        if isH5FileEmpty(filename) and len(metadata) > 0:
            with h5py.File(self.filename, 'w') as f:
                for key, value in self.metadata.items():
                    f.attrs[key] = value

    def __len__(self):
        return self.length

    def __getattr__(self, name):
        """
        If getattr requires a non-exist attribute of this, try to find it in the metadata
        """
        if name in self.metadata:
            return self.metadata[name]
        else:
            return None

    @property
    def id(self):
        return ut.fileNameToId(self.filename)

    @property
    def data(self):
        if not self._data:
            with h5py.File(self.filename, 'r') as f:
                for key in f.keys():
                    self._data.append(self.load_key(f, key))
            self._data.sort(key=lambda x: x.id)
        return self._data

    @property
    def data_head(self):
        return self.load_data_by_id(0)

    def load_data_by_id(self, Id) -> State:
        try:
            with h5py.File(self.filename, 'r') as f:
                for key in f.keys():
                    if f[key].attrs['id'] == Id:
                        return self.load_key(f, key)
        except:
            print('broken data file:', self.filename)
            raise ValueError

    def load_key(self, h5_file_handle, key) -> State:
        try:
            configuration = h5_file_handle[key]['data'][...]  # [...] converts a h5 object to a numpy array
            dic = {}
            for k in h5_file_handle[key].attrs.keys():
                dic[k] = h5_file_handle[key].attrs[k]
            return State.load(configuration, self.metadata, dic)
        except:
            print('broken data file:', self.filename)
            raise ValueError

    @property
    def length(self):
        with h5py.File(self.filename, 'r') as f:
            return len(f.keys())

    @classmethod
    def loadFrom(cls, filename) -> Union['DataSet', None]:
        obj = cls(filename, dict())
        try:
            with h5py.File(filename, 'r') as f:
                for key in f.attrs.keys():
                    obj.metadata[key] = f.attrs[key]
            # check data integrity
            _ = obj.data_head
            return obj
        except Exception as e:
            print('An error occurred:', e)
            print('while reading:', filename)
            return None

    # Simulation Methods

    def append(self, state: State, need_data_in_python=False):
        if need_data_in_python:
            self._data.append(state)
        else:
            self._data.append(None)
        self.increase(state)

    def increase(self, state: State):
        n_existing_groups = len(self._data)
        with h5py.File(self.filename, 'a') as f:
            grp = f.create_group(str(n_existing_groups))
            grp.create_dataset('data', data=state.xyt)
            for key, value in state.metadata.items():
                grp.attrs[key] = value

    def saveAll(self):
        with h5py.File(self.filename, 'w') as f:
            for i, state in enumerate(self._data):
                grp = f.create_group(str(i))
                grp.create_dataset('data', data=state.xyt)
                for key, value in state.metadata.items():
                    grp.attrs[key] = value

    # Analysis Methods

    @property
    def gamma(self):
        return 1 + (self.n - 1) * self.d / 2

    @property
    def rho0(self):
        return self.data_head.rho

    @property
    def Gamma0(self):
        return self.data_head.Gamma

    @property
    def descentCurves(self):
        return [state.descent_curve for state in self.data]

    @lru_cache(maxsize=None)
    def curveTemplate(self, prop: str):
        # parallel calculation will cause a crash??
        parallel_mode = 'Debug' if prop in ['meanDistance', 'finalStepSize'] else 'Release'
        return np.array(ut.Map(parallel_mode)(lambda state: getattr(state, prop), self.data))

    @property
    def rhos(self):
        return self.curveTemplate('rho')

    @property
    def phis(self):
        return self.curveTemplate('phi')

    @property
    def distanceCurve(self):
        return ut.map2(State.distance, self.data)

    def toDataFrame(self):
        """
        Extract abstract information and construct a line of the dataframe
        """
        dic = {
            'id': self.id,
            'length': self.length,
            'gamma': self.gamma,
            'rho0': self.rho0,
            'Gamma0': self.Gamma0,
            **self.metadata,
        }
        return pd.DataFrame([dic])

    def stateAtDensity(self, density: float) -> State:
        dr = np.abs(self.rhos - density)
        idx = np.argmin(dr)
        print('Find at density:', self.rhos[idx])
        return self.data[idx]

    def critical(self, energy_threshold: float) -> State:
        es = self.curveTemplate('energy')
        idx = ut.findFirstOfLastSubsequence(es, lambda x: x > energy_threshold)
        print('Find at density:', self.rhos[idx])
        return self.data[idx]

    def angleDistribution(self):
        return np.array(ut.Map('Release')(lambda x: x.angleDistribution(), self.data)).T

    def SiDistribution(self):
        return np.array(ut.Map('Release')(lambda x: x.SiDistribution(), self.data)).T

    def neighborAngleDist(self):
        return np.array(ut.Map('Debug')(lambda x: x.neighborAngleDist, self.data)).T