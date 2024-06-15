import ctypes as ct
import os
import sys
import time

import numpy as np

import src.fp as fp


def getLibraryPath():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    if sys.platform.startswith('win'):
        return os.path.join(dir_path, '..', 'x64', 'Release', 'OMPFrame.dll')
    elif sys.platform.startswith('linux'):
        return os.path.join(dir_path, '..', 'OMPFrame', 'OMPFrame.so')
    else:
        raise SystemError


class Kernel:
    def __init__(self):
        self.potential_path = 'potential.dat'
        self.potential_meta_path = 'potential.metadata.json'
        dll_path = getLibraryPath()
        self.dll = ct.cdll.LoadLibrary(dll_path)
        self.dll.init()
        self.dll.setEnums.argtypes = [ct.c_int]
        self.dll.setRod.argtypes = [ct.c_int, ct.c_float]
        self.dll.setBoundary.argtypes = [ct.c_void_p, ct.c_float, ct.c_float]
        self.dll.readPotential.argtypes = [ct.c_int, ct.c_float]
        self.dll.getPotentialId.restype = ct.c_int
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
        self.dll.singleStep.argtypes = [ct.c_void_p, ct.c_int, ct.c_float]
        self.dll.equilibriumGD.argtypes = [ct.c_void_p, ct.c_int]
        self.dll.equilibriumGD.restype = ct.c_float

    def returnFixedArray(self, dll_function, length):
        dll_function.restype = ct.POINTER(ct.c_float)

        def inner(*args):
            arr_ptr = dll_function(*args)
            return np.array([arr_ptr[i] for i in range(length)])

        return inner

    def createState(self, N, boundary_a, boundary_b):
        return self.dll.createState(N, boundary_a, boundary_b)

    def getStateData(self, address, N):
        array_pointer = ct.cast(self.dll.getStateData(address), ct.POINTER(ct.c_float * N * 3))
        return np.ctypeslib.as_array(array_pointer.contents).copy().reshape((N, 3))

    def getNumOfIterations(self, address):
        return int(self.dll.getStateIterations(address))

    def getStateMaxGradients(self, address):
        iterations = self.getNumOfIterations(address)
        array_pointer = ct.cast(self.dll.getStateMaxGradients(address), ct.POINTER(ct.c_float * iterations))
        return np.ctypeslib.as_array(array_pointer.contents).copy()

    def getStateResidualForce(self, address, N):
        array_pointer = ct.cast(self.dll.getStateResidualForce(address), ct.POINTER(ct.c_float * N * 3))
        return np.ctypeslib.as_array(array_pointer.contents).copy().reshape((N, 3))

    def initStateAsDisks(self, address):
        return self.dll.initStateAsDisks(address)

    def setEnums(self, potential_func):
        """
        PotentialFunc: Hertzian=0, ScreenCoulomb=1
        """
        return self.dll.setEnums(potential_func)

    def setBoundary(self, address, a, b):
        return self.dll.setBoundary(address, a, b)

    def setRod(self, n, d, save=True):
        if not save:
            self.generatePotential(n, d)
            return
        if self.checkPotential(n, d):
            print("Load existing potential from file.")
            self.readPotential(n, d)
        else:
            self.generatePotential(n, d)
            self.writePotential(n, d)

    def generatePotential(self, n, d):
        """
        generate potential look-up table for temporary usage
        """
        try:
            start_t = time.perf_counter()
            self.dll.setRod(n, d)
            end_t = time.perf_counter()
            dt = round(end_t - start_t, 2)
            print(f"Initialized the potential in {dt} seconds.")
        except Exception as e:
            print(e)
            print("you may forget to call `setEnums` to set a few key parameters")
            raise BaseException

    def readPotential(self, n, d):
        self.dll.readPotential(n, d)

    def checkPotential(self, n, d):
        if not os.path.exists(self.potential_path) or not os.path.exists(self.potential_meta_path):
            return False
        metadata = fp.readJson(self.potential_meta_path)

        # check properties
        if metadata['n'] != n or metadata['d'] != d:
            return False
        if metadata['potential func id'] != self.dll.getPotentialId():
            return False

        # check file integrity
        if metadata['size'] != fp.getFileSize(self.potential_path) or metadata['hash'] != fp.getFileHash(
                self.potential_path):
            return False
        return True

    def writePotential(self, n, d):
        self.dll.writePotential()

        # write metadata file
        metadata = {
            'n': n,
            'd': d,
            'potential func id': self.dll.getPotentialId(),
            'size': fp.getFileSize(self.potential_path),
            'hash': fp.getFileHash(self.potential_path),
        }
        fp.writeJson(self.potential_meta_path, metadata)

    def singleStep(self, address, mode: int, step_size: float):
        return self.dll.singleStep(address, mode, step_size)

    def equilibriumGD(self, address, max_iterations: int):
        return self.dll.equilibriumGD(address, max_iterations)


ker = Kernel()
