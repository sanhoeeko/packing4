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

bins_of_distribution = 500


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


def KDE_distribution(data: np.ndarray, interval: (float, float)) -> (np.ndarray, np.ndarray):
    """
    return: (xs: np.ndarray, distribution: np.ndarray)
    """
    length = interval[1] - interval[0]
    kde = gaussian_kde(data, bw_method=length / bins_of_distribution * 4)
    x_grid = np.linspace(*interval, bins_of_distribution)
    p_x = kde.evaluate(x_grid)
    return x_grid, p_x


def BIN_distribution(data: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    return: (xs: np.ndarray, distribution: np.ndarray)
    """
    p_x, x_grid = np.histogram(data, bins=bins_of_distribution)
    return x_grid[:-1], p_x


def standard_error(matrix: np.ndarray, axis=0) -> np.ndarray:
    # Calculate the standard deviation along axis 1, ignoring NaN values
    std_dev = np.nanstd(matrix, axis=axis)
    # Calculate the number of non-NaN values along axis 1
    n = np.sum(~np.isnan(matrix), axis=axis)
    # Calculate the standard error
    std_error = std_dev / np.sqrt(n)
    return std_error


def sample_equal_stride(lst: list, n_samples: int) -> list:
    stride = len(lst) // n_samples
    start = len(lst) - n_samples * stride
    return lst[start:][::stride]


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
