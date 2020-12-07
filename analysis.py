# Please run utility.py at first

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import utility                  # Self-created
import geopandas as gpd
import statsmodels.api as sm    # import statsmodels

from pathlib import Path
from scipy.stats import ttest_ind

over100_voted_MI = pd.read_json(Path(os.getcwd())/'out/dead_voters_who_voted.json', orient=str)
over100_registered_MI = pd.read_json(Path(os.getcwd())/'out/registered_dead_voters.json', orient=str)
registered_incomplete_wayne = pd.read_csv(Path(os.getcwd())/'mi_wa_voterfile.csv')

wayne_zip = registered_incomplete_wayne['zip_code'].unique().tolist()
over100_wayne = over100_registered_MI.loc[over100_registered_MI['ZipCode'].isin(wayne_zip)]

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
simple_distri_plot(over100_registered_MI, 'BirthYear', 'Birth Year', 'Registered Voters Over 100 by Birth Year in Michigan', False)


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


# Test the difference of over-100 vote stuats distribution between detroit and non-detroit area
detroit = pd.read_csv(Path(os.getcwd())/'detroit_ziprange.csv')
ziptorange(over100_registered_MI, detroit['zip'], 'ZipCode', 'detroit')
checkthebalance(over100_registered_MI, 'detroit', 'Voted', 0.05)

# Test the difference of age distribution between detroit and non-detroit area in Wayne County
rename = {'first_name':'FirstName', 'last_name':'LastName', 'birth_year':'BirthYear', 'zip_code':'ZipCode'}
wayne = pd.concat([over100_wayne, registered_incomplete_wayne.rename(columns=rename)], join='outer')

birthyeartoage(wayne, 'BirthYear', 2020)
ziptorange(wayne, detroit['zip'], 'ZipCode', 'detroit')
checkthebalance(wayne, 'detroit', 'Age', 0.05)


# Regression analysis
## Rename and create variable
over100(wayne, 'BirthYear', 2020)
wayne = organize_byzip(wayne, 'Over100', 'ZipCode')
wayne['Over100_per'] = 100*wayne[1]/(wayne[1]+wayne[0])

## Merge in gross income dataset
income = pd.read_csv(Path(os.getcwd())/'income by zipcode.csv').dropna()
income['ZipCode'] = income['zipcode'].astype(int)
ziptorange(income, detroit['zip'], 'ZipCode', 'Detroit')

df = pd.merge(wayne, income, on=['ZipCode'])
df['income'] = np.log2(df['income'])

## Regression
X = df[['Over100_per','income']]
y = df['Detroit']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

## Print out the statistics
model.summary()
