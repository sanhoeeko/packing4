import ctypes as ct

import numpy as np

from src.kernel import Kernel

dof = 4


class TestKernel(Kernel):
    def __init__(self):
        super().__init__()
        self.dll.interpolatePotential.argtypes = [ct.c_float] * 3
        self.dll.interpolatePotential.restype = ct.c_float
        self.dll.precisePotential.argtypes = [ct.c_float] * 3
        self.dll.precisePotential.restype = ct.c_float
        self.dll.fastPotential.argtypes = [ct.c_float] * 3
        self.dll.fastPotential.restype = ct.c_float
        self.dll.interpolateGradient.argtypes = [ct.c_float] * 3
        self.interpolateGradient = self.returnFixedArray(self.dll.interpolateGradient, 3)
        self.dll.gradientTest.argtypes = [ct.c_float] * 4
        self._gradientTest = self.returnFixedArray(self.dll.gradientTest, 2 * dof)
        self.dll.gradientReference.argtypes = [ct.c_float] * 4
        self._gradientReference = self.returnFixedArray(self.dll.gradientReference, 2 * dof)
        self.dll.getMirrorOf.argtypes = [ct.c_float] * 5
        self.getMirrorOf = self.returnFixedArray(self.dll.getMirrorOf, 3)

    def gradientTest(self, x, y, t1, t2):
        arr = self._gradientTest(x, y, t1, t2)
        return np.hstack([arr[0:3], arr[4:7]])

    def gradientReference(self, x, y, t1, t2):
        arr = self._gradientReference(x, y, t1, t2)
        return np.hstack([arr[0:3], arr[4:7]])

    def interpolatePotential(self, x, y, t):
        return self.dll.interpolatePotential(x, y, t)

    def precisePotential(self, x, y, t):
        return self.dll.precisePotential(x, y, t)

    def fastPotential(self, x, y, t):
        return self.dll.fastPotential(x, y, t)


ker = TestKernel()
