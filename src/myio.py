from functools import lru_cache

import h5py
import numpy as np
import pandas as pd

import src.utils as ut
from .state import State


class DataSet:
    def __init__(self, filename: str, metadata: dict):
        self.filename = filename
        self.metadata: dict = metadata
        self.data: list[State] = []
        # save metadata first, even if there is no data
        # if it is not in the reading mode
        if len(metadata) > 0:
            with h5py.File(self.filename, 'w') as f:
                for key, value in self.metadata.items():
                    f.attrs[key] = value

    @property
    def id(self):
        return ut.fileNameToId(self.filename)

    @classmethod
    def loadFrom(cls, filename):
        obj = cls(filename, dict())
        with h5py.File(filename, 'r') as f:
            for key in f.attrs.keys():
                obj.metadata[key] = f.attrs[key]
            for key in f.keys():
                configuration = f[key]['data'][...]  # [...] converts a h5 object to a numpy array
                dic = {}
                for k in f[key].attrs.keys():
                    dic[k] = f[key].attrs[k]
                obj.data.append(State.load(configuration, obj.metadata, dic))
        obj.data.sort(key=lambda x: x.id)
        return obj

    # Simulation Methods

    def append(self, state: State):
        self.data.append(state)
        self.increase(state)

    def increase(self, state: State):
        n_existing_groups = len(self.data)
        with h5py.File(self.filename, 'a') as f:
            grp = f.create_group(str(n_existing_groups))
            grp.create_dataset('data', data=state.xyt)
            for key, value in state.metadata.items():
                grp.attrs[key] = value

    def saveAll(self):
        with h5py.File(self.filename, 'w') as f:
            for i, state in enumerate(self.data):
                grp = f.create_group(str(i))
                grp.create_dataset('data', data=state.xyt)
                for key, value in state.metadata.items():
                    grp.attrs[key] = value

    # Analysis Methods

    @property
    def rho0(self):
        return self.data[0].rho

    @property
    def Gamma0(self):
        return self.data[0].Gamma

    @property
    def rhos(self):
        return np.array([state.rho for state in self.data])

    @lru_cache(maxsize=None)
    def curveTemplate(self, prop: str):
        return np.array([getattr(state, prop) for state in self.data])

    def toDataFrame(self):
        """
        Extract abstract information and construct a line of the dataframe
        """
        dic = {
            'id': self.id,
            **self.metadata,
            'rho0': self.rho0,
            'Gamma0': self.Gamma0,
        }
        return pd.DataFrame([dic])
