import ctypes as ct

from kernel import Kernel


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
        self.gradientTest = self.returnFixedArray(self.dll.gradientTest, 6)
        self.dll.gradientReference.argtypes = [ct.c_float] * 4
        self.gradientReference = self.returnFixedArray(self.dll.gradientReference, 6)

    def interpolatePotential(self, x, y, t):
        return self.dll.interpolatePotential(x, y, t)

    def precisePotential(self, x, y, t):
        return self.dll.precisePotential(x, y, t)

    def fastPotential(self, x, y, t):
        return self.dll.fastPotential(x, y, t)


ker = TestKernel()
