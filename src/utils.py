import ctypes
import datetime
import hashlib
import inspect as insp
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import numpy as np
import pandas as pd


def applyPipeline(obj, funcs):
    return reduce(lambda x, f: f(x), funcs, obj)


def reverseClassMethod(func, *args):
    """
    Assume that the class method is curry, i.e, only has a parameter `self`.
    If not, please curry it first.
    """

    def inner(obj):
        return func(obj, *args)

    return inner


def map2(func, lst):
    """
    like np.diff
    """
    return [func(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]


def getFileHash(path):
    hash_obj = hashlib.sha256()
    with open(path, 'rb') as f:
        chunk = f.read(8192)
        while chunk:
            hash_obj.update(chunk)
            chunk = f.read(8192)
    return hash_obj.hexdigest()


def getFileSize(path):
    return os.path.getsize(path)


def writeJson(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def readJson(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def Map(mode):
    """
    called as: ut.Map('Debug')(func, iterable)
    Debug: serial
    Release: parallel
    """
    if mode == 'Debug':
        def map_func(func, tasks):
            return [func(task) for task in tasks]
    elif mode == 'Release':
        def map_func(func, tasks):
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                return list(executor.map(func, tasks))
    else:
        raise ValueError("Invalid mode. Please set it to 'Debug' or 'Release'.")

    return map_func


def fileNameToId(s):
    return s.split('\\')[-1].split('.')[0]


def findFirst(lst, lambda_expr):
    return next((item for item in lst if lambda_expr(item)), None)


def findFirstOfLastSubsequence(arr, lambda_expr):
    mask = lambda_expr(arr)
    mask_shifted = np.roll(mask, 1)
    mask_shifted[0] = False
    starts = np.where(mask & ~mask_shifted)[0]
    if starts.size > 0:
        return starts[-1]
    else:
        return None


def ndarrayAddress(a: np.ndarray):
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def sortListByDataFrame(df, lst):
    # list -> DataFrame -> sort -> list
    lst_df = pd.DataFrame([(obj.id, obj) for obj in lst], columns=['id', 'object'])
    merged_df = pd.merge(df, lst_df, on='id',
                         how='inner')  # inner: collect if it appears both at left and right DataFrame
    return merged_df['object'].tolist()


class MyThreadRecord:
    def __init__(self, user_name: str, num_threads):
        self.user_name = user_name
        self.num_threads = num_threads
        self.file_path = "/home/share/thread.txt"
        self.pid_str = f"[{os.getpid()}]"

    def __enter__(self):
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, 'a') as f:
            time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{self.pid_str} {self.user_name} {self.num_threads} threads, {time_str}\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        with open(self.file_path, 'w') as f:
            for line in lines:
                if not line.startswith(self.pid_str):  # delete the line that starts with the current PID
                    f.write(line)


class CommandQueue:
    def __init__(self, current_obj):
        self.parameter_queue: list[str] = []
        self.action_queue: list[str] = []
        self.current_obj = current_obj

    def push(self, token: str):
        if token.startswith('-'):
            self.action_queue.append(token[1:])
        else:
            self.parameter_queue.append(token)
        if len(self.action_queue) > 0 and len(self.parameter_queue) >= self.requires():
            self.pop()

    def top(self):
        return getattr(self.current_obj, self.action_queue[0])

    def requires(self):
        return len(insp.signature(self.top()).parameters)

    def pop(self):
        req = self.requires()
        params = self.parameter_queue[:req]
        self.current_obj = self.top()(*params)
        self.action_queue = self.action_queue[1:]
        self.parameter_queue = self.parameter_queue[req:]

    def result(self):
        return self.current_obj
