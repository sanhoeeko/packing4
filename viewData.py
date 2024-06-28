import glob
import os

import pandas as pd

import src.utils as ut
from analysis import InteractiveViewer, RenderPipe
from src.myio import DataSet
from src.render import StateRenderer

target_dir = None
viewer = None


class DataViewer:
    def __init__(self, datasets: list[DataSet]):
        self.abstract = pd.DataFrame()
        self.datasets = datasets
        for d in datasets:
            self.abstract = pd.concat([self.abstract, d.toDataFrame()], ignore_index=True)

    def show(self, Id: str):
        d = ut.findFirst(self.datasets, lambda x: x.id == Id)
        InteractiveViewer(d, RenderPipe(StateRenderer.angle)).show()


def collectResultFiles(path: str):
    files = []
    for pattern in [f'{path}/*.h5']:
        files.extend(glob.glob(pattern))
    return [os.path.abspath(file) for file in files]


def loadAll():
    data_files = collectResultFiles(target_dir)
    ds = [DataSet.loadFrom(file) for file in data_files]
    return DataViewer(ds)


def parse(cmd: str):
    """
    Format: command [arg] [-kwarg] [ | pipe command]
    """

    def subTokenize(cmd: str):
        # Split the command by space
        tokens = cmd.split()

        # Initialize the variables
        command = None
        args = []
        kwargs = {}

        # Iterate over the tokens
        for token in tokens:
            if not command:
                # The first token is the command
                command = token.strip()
            elif token.startswith('-'):
                # If the token starts with '-', it's a kwarg
                key, value = token[1:].split('=')
                kwargs[key.strip()] = value.strip()
            else:
                # Otherwise, it's an arg
                args.append(token.strip())

        return command, args, kwargs

    tokens = list(map(subTokenize, cmd.split('|')))
    command, args, kwargs = tokens[0]
    result = getattr(viewer, command)(*args, **kwargs)
    for i in range(1, len(tokens)):
        command, args, kwargs = tokens[i]
        result = getattr(viewer, command)(result, *args, **kwargs)
    return result


if __name__ == '__main__':
    target_dir = input('data folder name: ')
    viewer = loadAll()
    print(viewer.abstract)
    while True:
        parse(input('>>>'))
