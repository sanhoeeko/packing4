import numpy as np

from src.kernel import ker
from src.state import State
import src.utils as ut

class StateLoader(State):
    def __init__(self, state: State):
        self.__dict__ = state.__dict__
        self.data_ptr = self.init_cpp()

    def init_cpp(self):
        # load the state to cpp
        data_n4 = np.hstack([self.xyt, np.zeros((self.N, 1))]).reshape(-1)
        return ker.loadState(data_n4, self.N, self.A, self.B)

    def __del__(self):
        ker.freeState(self.data_ptr)

    def meanDistance_(self):
        return ker.meanDistance(self.data_ptr)

    def meanContactZ_(self):
        return ker.meanContactZ(self.data_ptr, self.gamma)

    def meanS_(self):
        return ker.meanS(self.data_ptr, self.gamma)

    def absPhi_(self, p: int):
        return ker.absPhi(self.data_ptr, self.gamma, p)

    def Si_(self):
        return ker.Si(self.data_ptr, self.N, self.gamma)

    def neighborAngleDist_(self, bins: int):
        return ker.neighborAngleDist(self.data_ptr, self.gamma, bins)
