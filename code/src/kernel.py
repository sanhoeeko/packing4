import ctypes as ct
import os
import sys
import threading
import time

import numpy as np

import src.utils as ut

kernel_mode = 'Release'


def getLibraryPath():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    if sys.platform.startswith('win'):
        if kernel_mode == 'Debug':
            return os.path.join(dir_path, '..', 'x64', 'Debug', 'OMPFrame.dll')
        else:
            return os.path.join(dir_path, '..', 'x64', 'Release', 'OMPFrame.dll')
    elif sys.platform.startswith('linux'):
        return os.path.join(dir_path, '..', 'OMPFrame', 'OMPFrame.so')
    else:
        raise SystemError


class Kernel:
    def __init__(self):
        self.lock = threading.Lock()
        self.potential_path = 'potential.dat'
        self.potential_meta_path = 'potential.metadata.json'
        dll_path = getLibraryPath()
        self.dll = ct.cdll.LoadLibrary(dll_path)
        self.dll.init()
        self.dll.setEnums.argtypes = [ct.c_int]
        self.dll.setPotentialPower.argtypes = [ct.c_float]
        self.dll.declareRod.argtypes = [ct.c_int, ct.c_float]
        self.dll.setRod.argtypes = [ct.c_int, ct.c_float, ct.c_int]
        self.dll.setBoundary.argtypes = [ct.c_void_p, ct.c_float, ct.c_float]
        self.dll.loadState.argtypes = [ct.c_void_p, ct.c_int, ct.c_float, ct.c_float]
        self.dll.loadState.restype = ct.c_void_p
        self.dll.freeState.argtypes = [ct.c_void_p]
        self.dll.readPotential.argtypes = [ct.c_int, ct.c_float]
        self.dll.getPotentialId.restype = ct.c_int
        self.dll.createState.argtypes = [ct.c_int, ct.c_float, ct.c_float]
        self.dll.createState.restype = ct.c_void_p
        self.dll.getSiblingId.argtypes = [ct.c_void_p]
        self.dll.getSiblingId.restype = ct.c_int
        self.dll.getStateData.argtypes = [ct.c_void_p]
        self.dll.getStateData.restype = ct.c_void_p
        self.dll.getStateIterations.argtypes = [ct.c_void_p]
        self.dll.getStateIterations.restype = ct.c_int
        self.dll.getStateMaxGradOrEnergy.argtypes = [ct.c_void_p]
        self.dll.getStateMaxGradOrEnergy.restype = ct.c_void_p
        self.dll.getStateResidualForce.argtypes = [ct.c_void_p]
        self.dll.getStateResidualForce.restype = ct.c_void_p
        self.dll.getStateMaxResidualForce.argtypes = [ct.c_void_p]
        self.dll.getStateMaxResidualForce.restype = ct.c_float
        self.dll.initStateAsDisks.argtypes = [ct.c_void_p]
        self.dll.singleStep.argtypes = [ct.c_void_p, ct.c_int, ct.c_float]
        self.dll.equilibriumGD.argtypes = [ct.c_void_p, ct.c_int]
        self.dll.equilibriumGD.restype = ct.c_float
        self.dll.eqLineGD.argtypes = [ct.c_void_p, ct.c_int]
        self.dll.eqLineGD.restype = ct.c_float
        self.dll.eqLBFGS.argtypes = [ct.c_void_p, ct.c_int]
        self.dll.eqLBFGS.restype = ct.c_float
        self.dll.eqMix.argtypes = [ct.c_void_p, ct.c_int]
        self.dll.eqMix.restype = ct.c_float
        self.dll.setStateData.argtypes = [ct.c_void_p, ct.c_void_p]
        self.dll.landscapeAlongGradient.argtypes = [ct.c_void_p, ct.c_float, ct.c_int]
        self.dll.landscapeAlongGradient.restype = ct.c_void_p
        self.dll.landscapeLBFGS.argtypes = [ct.c_void_p, ct.c_float, ct.c_int]
        self.dll.landscapeLBFGS.restype = ct.c_void_p
        self.dll.landscapeOnGradientSections.argtypes = [ct.c_void_p, ct.c_float, ct.c_int]
        self.dll.landscapeOnGradientSections.restype = ct.c_void_p
        self.dll.testERoot.argtypes = [ct.c_void_p, ct.c_float]
        self.dll.testERoot.restype = ct.c_float
        self.dll.testBestStepSize.argtypes = [ct.c_void_p, ct.c_float]
        self.dll.testBestStepSize.restype = ct.c_float
        self.dll.meanDistance.argtypes = [ct.c_void_p, ct.c_float]
        self.dll.meanDistance.restype = ct.c_float
        self.dll.meanContactZ.argtypes = [ct.c_void_p, ct.c_float]
        self.dll.meanContactZ.restype = ct.c_float
        self.dll.meanS.argtypes = [ct.c_void_p, ct.c_float]
        self.dll.meanS.restype = ct.c_float
        self.dll.absPhi.argtypes = [ct.c_void_p, ct.c_float, ct.c_int]
        self.dll.absPhi.restype = ct.c_float
        self.dll.Si.argtypes = [ct.c_void_p, ct.c_float]
        self.dll.Si.restype = ct.c_void_p
        self.dll.neighborAngleDist.argtypes = [ct.c_void_p, ct.c_float, ct.c_int]
        self.dll.neighborAngleDist.restype = ct.c_void_p

    def returnFixedArray(self, dll_function, length):
        dll_function.restype = ct.POINTER(ct.c_float)

        def inner(*args):
            arr_ptr = dll_function(*args)
            return np.array([arr_ptr[i] for i in range(length)])

        return inner

    def createState(self, N, boundary_a, boundary_b):
        return self.dll.createState(N, boundary_a, boundary_b)

    def getSiblingId(self, address) -> int:
        return self.dll.getSiblingId(address)

    def getStateData(self, address, N):
        array_pointer = ct.cast(self.dll.getStateData(address), ct.POINTER(ct.c_float * N * 4))
        raw_data = np.ctypeslib.as_array(array_pointer.contents).copy().reshape((N, 4))
        return raw_data[:, :3]

    def getNumOfIterations(self, address):
        return int(self.dll.getStateIterations(address))

    def getStateMaxGradOrEnergy(self, address):
        iterations = self.getNumOfIterations(address)
        if iterations == 0:
            return []
        array_pointer = ct.cast(self.dll.getStateMaxGradOrEnergy(address), ct.POINTER(ct.c_float * iterations))
        return np.ctypeslib.as_array(array_pointer.contents).copy()

    def getStateMaxGradients(self, address):
        return self.getStateMaxGradOrEnergy(address)

    def getStateEnergyCurve(self, address):
        return self.getStateMaxGradOrEnergy(address)

    def getStateResidualForce(self, address, N):
        array_pointer = ct.cast(self.dll.getStateResidualForce(address), ct.POINTER(ct.c_float * N * 4))
        raw_data = np.ctypeslib.as_array(array_pointer.contents).copy().reshape((N, 4))
        return raw_data[:, :3]

    def getStateMaxResidualForce(self, address):
        return self.dll.getStateMaxResidualForce(address)

    def initStateAsDisks(self, address):
        return self.dll.initStateAsDisks(address)

    def setEnums(self, potential_func):
        """
        PotentialFunc: Hertzian=0, ScreenCoulomb=1, GeneralizedHertzian=2
        """
        return self.dll.setEnums(potential_func)

    def setPotentialPower(self, power: float):
        return self.dll.setPotentialPower(power)

    def setBoundary(self, address, a, b):
        return self.dll.setBoundary(address, a, b)

    def declareRod(self, n, d):
        self.dll.declareRod(n, d)

    def setRod(self, n, d, n_threads=4, save=True):
        if not save:
            self.generatePotential(n, d, n_threads)
            return
        if self.checkPotential(n, d):
            print("Load existing potential from file.")
            self.readPotential(n, d)
        else:
            self.generatePotential(n, d, n_threads)
            self.writePotential(n, d)

    def setStateData(self, address, configuration: np.ndarray):
        return self.dll.setStateData(address, ut.ndarrayAddress(configuration.astype(np.float32)))

    def loadState(self, configuration: np.ndarray, N: int, a: float, b: float):
        """
        this method is thread-safe, and returns a void pointer (State*)
        """
        with self.lock:
            return self.dll.loadState(ut.ndarrayAddress(configuration.astype(np.float32)), N, a, b)

    def freeState(self, address):
        with self.lock:
            self.dll.freeState(address)

    def generatePotential(self, n, d, n_threads):
        """
        generate potential look-up table for temporary usage
        """
        try:
            start_t = time.perf_counter()
            self.dll.setRod(n, d, n_threads)
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
        metadata = ut.readJson(self.potential_meta_path)

        # check properties
        if metadata['n'] != n or metadata['d'] != d:
            return False
        if metadata['potential func id'] != self.dll.getPotentialId():
            return False

        # check file integrity
        if metadata['size'] != ut.getFileSize(self.potential_path) or metadata['hash'] != ut.getFileHash(
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
            'size': ut.getFileSize(self.potential_path),
            'hash': ut.getFileHash(self.potential_path),
        }
        ut.writeJson(self.potential_meta_path, metadata)

    def singleStep(self, address, mode: int, step_size: float):
        return self.dll.singleStep(address, mode, step_size)

    def equilibriumGD(self, address, max_iterations: int):
        return self.dll.equilibriumGD(address, max_iterations)

    def eqLineGD(self, address, max_iterations: int):
        return self.dll.eqLineGD(address, max_iterations)

    def eqLBFGS(self, address, max_iterations: int):
        return self.dll.eqLBFGS(address, max_iterations)

    def eqMix(self, address, max_iterations: int):
        return self.dll.eqMix(address, max_iterations)

    def landscapeAlongGradient(self, address, max_stepsize: float, n: int):
        return self.returnFixedArray(self.dll.landscapeAlongGradient, n)(address, max_stepsize, n)

    def landscapeLBFGS(self, address, max_stepsize: float, n: int):
        return self.returnFixedArray(self.dll.landscapeLBFGS, n)(address, max_stepsize, n)

    def landscapeOnGradientSections(self, address, max_stepsize: float, n: int):
        m = (2 * n + 1) * n
        mat = self.returnFixedArray(self.dll.landscapeOnGradientSections, m)(address, max_stepsize, m)
        return mat.reshape((2 * n + 1, n))

    def ERoot(self, address, max_stepsize: float):
        return self.dll.testERoot(address, max_stepsize)

    def bestStepSize(self, address, max_stepsize: float):
        return self.dll.testBestStepSize(address, max_stepsize)

    def meanDistance(self, address, gamma: float):
        return self.dll.meanDistance(address, gamma)

    def meanContactZ(self, address, gamma: float):
        return self.dll.meanContactZ(address, gamma)

    def meanS(self, address, gamma: float):
        return self.dll.meanS(address, gamma)

    def absPhi(self, address, gamma: float, p: int):
        return self.dll.absPhi(address, gamma, p)

    def Si(self, address, N: int, gamma: float):
        return self.returnFixedArray(self.dll.Si, N)(address, gamma)

    def neighborAngleDist(self, address, gamma: float, bins: int):
        return self.returnFixedArray(self.dll.neighborAngleDist, bins)(address, gamma, bins)


ker = Kernel()
