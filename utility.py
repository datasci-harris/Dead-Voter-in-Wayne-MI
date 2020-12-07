# Make some utility function to simplify the work
import numpy as np
import pandas as pd

def ziptorange(dataset, rangeset, zipvariable, rangename):
    dataset[rangename] = np.where(dataset[zipvariable].isin(rangeset), 1, 0)
    return

def birthyeartoage(dataset, birthyear, currentyear):
    assert isinstance(currentyear, int), 'It should be the current year number.'
    dataset['Age'] = 2020-dataset[birthyear]
    return

def over100(dataset, birthyear, currentyear):
    dataset['Over100'] = np.where(dataset[birthyear] < (currentyear - 99), 1, 0)
    return

# Make a function to organize any variable by zipvariable
def organize_byzip(dataset, variable, zipvariable):
    dataset = dataset.groupby([zipvariable, variable]).size().reset_index(name='Count')
    dataset = dataset.pivot(index=zipvariable, columns=variable, values='Count')
    dataset.columns.name = None
    afterzip = dataset.reset_index()
    return afterzip
