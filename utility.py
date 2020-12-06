# Make some utility function to simplify the work
import numpy as np
import pandas as pd

def ziptorange(dataset, rangeset, zipvariable, rangename):
    dataset[rangename] = np.where(dataset[zipvariable].isin(rangeset), True, False)
    return

def birthyeartoage(dataset, birthyear, currentyear):
    assert isinstance(currentyear, int), 'It should be the current year number.'
    dataset['Age'] = 2020-dataset[birthyear]
    return

def over100(dataset, birthyear, currentyear):
    dataset['over100'] = np.where(dataset[birthyear] < (currentyear - 99), True, False)
    return
