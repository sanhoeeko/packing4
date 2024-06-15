import hashlib
import json
import os
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
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def getFileSize(path):
    return os.path.getsize(path)


def writeJson(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def readJson(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
