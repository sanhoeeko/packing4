import ctypes as ct
import time

import numpy as np


class Kernel:
    def __init__(self):
        self.dll = ct.cdll.LoadLibrary("./x64/Release/OMPFrame.dll")
        self.dll.init()
        self.dll.setRod.argtypes = [ct.c_int, ct.c_float]
        self.dll.interpolatePotential.argtypes = [ct.c_float, ct.c_float, ct.c_float]
        self.dll.interpolatePotential.restype = ct.c_float
        self.dll.setBoundary.argtypes = [ct.c_void_p, ct.c_float, ct.c_float]
        self.dll.createState.argtypes = [ct.c_int, ct.c_float, ct.c_float]
        self.dll.createState.restype = ct.c_void_p
        self.dll.getStateData.argtypes = [ct.c_void_p]
        self.dll.getStateData.restype = ct.c_void_p
        self.dll.getStateIterations.argtypes = [ct.c_void_p]
        self.dll.getStateIterations.restype = ct.c_int
        self.dll.getStateMaxGradients.argtypes = [ct.c_void_p]
        self.dll.getStateMaxGradients.restype = ct.c_void_p
        self.dll.getStateResidualForce.argtypes = [ct.c_void_p]
        self.dll.getStateResidualForce.restype = ct.c_void_p
        self.dll.initStateAsDisks.argtypes = [ct.c_void_p]
        self.dll.equilibriumGD.argtypes = [ct.c_void_p]
        self.dll.equilibriumGD.restype = ct.c_float

    def returnFixedArray(self, dll_function, length):
        dll_function.restype = ct.POINTER(ct.c_float)

        def inner(*args):
            arr_ptr = dll_function(*args)
            return [arr_ptr[i] for i in range(length)]

        return inner

    def createState(self, N, boundary_a, boundary_b):
        return self.dll.createState(N, boundary_a, boundary_b)

    def getStateData(self, address, N):
        array_pointer = ct.cast(self.dll.getStateData(address), ct.POINTER(ct.c_float * N * 3))
        return np.ctypeslib.as_array(array_pointer.contents).copy().reshape((N, 3))

    def getStateMaxGradients(self, address):
        iterations = int(self.dll.getStateIterations(address))
        array_pointer = ct.cast(self.dll.getStateMaxGradients(address), ct.POINTER(ct.c_float * iterations))
        return np.ctypeslib.as_array(array_pointer.contents).copy()

    def getStateResidualForce(self, address, N):
        array_pointer = ct.cast(self.dll.getStateResidualForce(address), ct.POINTER(ct.c_float * N * 3))
        return np.ctypeslib.as_array(array_pointer.contents).copy().reshape((N, 3))

    def initStateAsDisks(self, address):
        return self.dll.initStateAsDisks(address)

    def setBoundary(self, address, a, b):
        return self.dll.setBoundary(address, a, b)

    def setRod(self, n, d):
        start_t = time.perf_counter()
        self.dll.setRod(n, d)
        end_t = time.perf_counter()
        dt = round(end_t - start_t, 2)
        print(f"Initialized the potential in {dt} seconds.")

    def equilibriumGD(self, address):
        return self.dll.equilibriumGD(address)


ker = Kernel()
