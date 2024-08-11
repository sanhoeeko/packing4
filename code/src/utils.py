import ctypes
import hashlib
import inspect as insp
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


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
            with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as executor:
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


def indicesOfTheSame(df: pd.DataFrame, props: list[str]) -> list[tuple[int]]:
    if not all(prop in df.columns for prop in props):
        return None
    return df.groupby(props).apply(lambda x: tuple(x.index)).tolist()


def groupAndMergeRows(df: pd.DataFrame, keys: list[str]):
    if not all(key in df.columns for key in keys):
        return None

    def merge_group(group):
        merged_row = {}
        for col in group.columns:
            if group[col].nunique() == 1:
                merged_row[col] = group[col].iloc[0]
            else:
                merged_row[col] = None
        return merged_row

    grouped = df.groupby(keys).apply(merge_group).reset_index(drop=True)
    return pd.DataFrame(list(grouped))


def safe_log(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.log(x))


def entropyOf(dist: np.ndarray) -> float:
    # This is a naive algorithm to estimate entropy
    return -np.dot(dist, safe_log(dist))


def KDE_entropyOf(data: np.ndarray) -> float:
    kde = gaussian_kde(data)
    x_grid = np.linspace(min(data), max(data), 1000)
    p_x = kde.evaluate(x_grid)
    return -np.sum(p_x * safe_log(p_x)) * (x_grid[1] - x_grid[0])


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
