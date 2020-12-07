import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import utility                  # Self-created
import geopandas as gpd

from pathlib import Path
from scipy.stats import ttest_ind

over100_voted_MI = pd.read_json(Path(os.getcwd())/'out/dead_voters_who_voted.json', orient=str)
over100_registered_MI = pd.read_json(Path(os.getcwd())/'out/registered_dead_voters.json', orient=str)
registered_incomplete_wayne = pd.read_csv(Path(os.getcwd())/'mi_wa_voterfile.csv')


## Plotting
# Make a function to show the simple distribution
def simple_distri_plot(dataset, variable, xlab, title, categorical=True):
    assert isinstance(variable, str), 'It should be the name of the variable in the dataset you gave.'
    assert isinstance(title, str), 'It should be the title of the plot you want to output.'
    assert isinstance(xlab, str), 'It should be the xlab of the plot you want to output.'

    fig, ax = plt.subplots(figsize=(12,9))
    if categorical == True:
        bin = np.arange(len(dataset[variable].unique())+1)-0.5
    else:
        bin = np.array(sorted(set(dataset[variable].unique())))+0.5
        ax.locator_params(axis='x', integer=True)
    ax.hist(dataset[variable], bins=bin, edgecolor='grey', rwidth=0.8)
    ax.set_ylabel('Number')
    ax.set_xlabel(xlab)
    ax.set_title(title)
    plt.savefig('plot/{}.png'.format(title))
    plt.show()
    plt.close()
    return

# Distribution of all registered voters by race in Wayne County
simple_distri_plot(registered_incomplete_wayne, 'race_code', 'Race', 'Registered Voters by Race in Wayne County (Incomplete Data)')

# Distribution of registered voters over 100 by race in Wayne County
simple_distri_plot(registered_incomplete_wayne.loc[registered_incomplete_wayne["birth_year"]<=1920], 
                   'race_code', 'Race', 'Registered Voters Over 100 by Race in Wayne County (Incomplete Data)')

# Distribution of registered voters over 100 by birth_year in Michigan
simple_distri_plot(over100_registered_MI, 'BirthYear', 'Birth Year', 'Registered Voters over 100 by Birth Year in Michigan', False)

## Testing
# Make a t.test function
def checkthebalance(dataset, groupby, variable, alpha):
    assert isinstance(groupby, str), 'It should be the name of the group_by variable.'
    assert isinstance(variable, str), 'It should be the name of the variable being checked.'

    a, b = dataset.groupby(groupby)[variable].apply(lambda x:list(x))
    if ttest_ind(a, b).pvalue < alpha:
        print('The difference between these two groups is signficant at {} level.'.format(alpha))
    else:
        print('The difference between these two groups is insignficant at {} level.'.format(alpha))


# Test the difference of age distribution between detroit and non-detroit area
detroit = pd.read_csv(Path(os.getcwd())/'detroit_ziprange.csv')
rename = {'first_name':'FirstName', 'last_name':'LastName', 'birth_year':'BirthYear', 'zip_code':'ZipCode'}
whole = pd.concat([over100_registered_MI, registered_incomplete_wayne.rename(columns=rename)], join='inner')
whole
birthyeartoage(whole, 'BirthYear', 2020)
ziptorange(whole, detroit['zip'], 'ZipCode', 'detroit')
checkthebalance(whole, 'detroit', 'Age', 0.05)

# Test the difference of vote stuats distribution between detroit and non-detroit area
ziptorange(over100_registered_MI, detroit['zip'], 'ZipCode', 'detroit')
checkthebalance(over100_registered_MI, 'detroit', 'Voted', 0.05)