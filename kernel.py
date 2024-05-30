import ctypes as ct
import numpy as np


class Kernel:
    def __init__(self):
        self.dll = ct.cdll.LoadLibrary("./x64/Release/OMPFrame.dll")
        self.dll.init()
        self.dll.setRod.argtypes = [ct.c_int, ct.c_float]
        self.dll.createState.argtypes = [ct.c_int, ct.c_float, ct.c_float]
        self.dll.createState.restype = ct.c_void_p
        self.dll.getStateData.argtypes = [ct.c_void_p]
        self.dll.getStateData.restype = ct.c_void_p
        self.dll.getStateIterations.argtypes = [ct.c_void_p]
        self.dll.getStateIterations.restype = ct.c_int
        self.dll.getStateMaxGradient.argtypes = [ct.c_void_p]
        self.dll.getStateMaxGradient.restype = ct.c_void_p
        self.dll.initStateAsDisks.argtypes = [ct.c_void_p]

    def createState(self, N, boundary_a, boundary_b):
        return self.dll.createState(N, boundary_a, boundary_b)

    def getStateData(self, address):
        return self.dll.getStateData(address)

    def getStateMaxGradient(self, address):
        iterations = int(self.dll.getStateIterations(address))
        array_pointer = ct.cast(self.dll.getStateMaxGradient(address), ct.POINTER(ct.c_float * iterations))
        return np.ctypeslib.as_array(array_pointer.contents)

    def initStateAsDisks(self, address):
        return self.dll.initStateAsDisks(address)

    def setRod(self, n, d):
        pass


ker = Kernel()
