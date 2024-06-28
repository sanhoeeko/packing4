import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce


def applyPipeline(obj, funcs):
    return reduce(lambda x, f: f(x), funcs, obj)


def reverseClassMethod(func):
    """
    Assume that the class method is curry, i.e, only has a parameter `self`.
    If not, please curry it first.
    """

    def inner(obj):
        return func(obj)

    return inner


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
