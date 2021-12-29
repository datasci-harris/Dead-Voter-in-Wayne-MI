# Please run utility.py at first

import pandas as pd
import matplotlib
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import utility                  # Self-created

from sklearn.metrics import confusion_matrix, make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA

from pathlib import Path

"""
General grading comments:
- You make good use of functions, but left a lot of code out of them that should have been organized into one
- Good project and nicely written code
"""

# Read in dataset
over100_registered_MI = pd.read_json(Path(os.getcwd())/'out/registered_dead_voters.json', orient=str)
registered_incomplete_wayne = pd.read_csv(Path(os.getcwd())/'mi_wa_voterfile.csv')
detroit = pd.read_csv(Path(os.getcwd())/'detroit_ziprange.csv')

# Filter, clean, and merge
## Filter out all observation in Wayne County
wayne_zip = registered_incomplete_wayne['zip_code'].unique().tolist()
over100_wayne = over100_registered_MI.loc[over100_registered_MI['ZipCode'].isin(wayne_zip)]

## Rename and create variable
rename = {'first_name':'FirstName', 'last_name':'LastName', 'birth_year':'BirthYear', 'zip_code':'ZipCode'}
wayne = pd.concat([over100_wayne, registered_incomplete_wayne.rename(columns=rename)], join='outer')
over100(wayne, 'BirthYear', 2020)

wayne = organize_byzip(wayne, 'Over100', 'ZipCode')
wayne['Over100_per'] = wayne[1]/(wayne[1]+wayne[0])


# Machine Learning - Supervised
## Data Prepration
income = pd.read_csv(Path(os.getcwd())/'income by zipcode.csv').dropna()
income['ZipCode'] = income['zipcode'].astype(int)
ziptorange(income, detroit['zip'], 'ZipCode', 'Detroit')

df = pd.merge(wayne, income, on=['ZipCode'])
df['income'] = np.log2(df['income'])

## Model Selection
X = df[['Over100_per','income']]
Y = df['Detroit']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=123)

models = [('Dec Tree', DecisionTreeClassifier()),
          ('Lin Disc', LinearDiscriminantAnalysis()),
          ('GaussianNB', GaussianNB()),
          ('SVC', SVC(gamma = 'auto'))]

## Cited from the lecture note and the post https://piazza.com/class/kfook5s99he5uo?cid=101
def weighted_score_table(model_list):
    assert isinstance(model_list, list), 'It should be the list of models that you want to test.'
    score_list = [precision_score, recall_score, f1_score]
    results = []

    for name, model in model_list:
        name = [name]

        # Append the accuracy score at first since it does not need to be weighted
        kf = StratifiedKFold(n_splits=10)
        res = cross_val_score(model, X_train, Y_train, cv = kf, scoring = 'accuracy')
        name.append(round(res.mean(), 4))
        name.append(round(res.std(), 4))
        # Loop through other three scores that need to be weighted
        for score in score_list:
            scorer = make_scorer(score, average = 'weighted')
            kf = StratifiedKFold(n_splits=10)
            res = cross_val_score(model, X_train, Y_train, cv = kf, scoring = scorer)
            name.append(round(res.mean(), 4))
            name.append(round(res.std(), 4))

        results.append(name)

    score_table = pd.DataFrame(results, columns = ['name', 'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std', 'recall_mean', 'recall_std', 'f1_mean', 'f1_std'])
    return score_table

score_table = weighted_score_table(models)
print(score_table)


# Prediction on test dataset
def test_on(X_test, Y_test, model):
    assert isinstance(model, str), 'Either Guassian Naive Bayes or Linear Discriminate, as string.'

    if model == 'Guassian Naive Bayes':
        model = GaussianNB()
    elif model == 'Linear Discriminate':
        model = LinearDiscriminantAnalysis()
    predict = model.fit(X_train, Y_train).predict(X_test)
    predict = pd.DataFrame(predict)
    result = pd.concat([df['ZipCode'], Y_test, predict], axis = 1)
    result.columns = ['ZipCode', 'Result', 'Predict']

    mat = confusion_matrix(Y_test, predict)
    ax = sns.heatmap(mat, square = True, annot = True, cbar = False)
    matplotlib.pyplot.show(ax) #JL: plt.show(ax)

    return result

GNB_set = test_on(X_test, Y_test, 'Guassian Naive Bayes')
LD_set = test_on(X_test, Y_test, 'Linear Discriminate')


# Machine Learning - Supervised
## Dimensionality reduction
model = PCA(n_components=2)
model.fit(X)
X_2D = model.transform(X)
df['PCA1'] = X_2D[:,0]
df['PCA2'] = X_2D[:,1]
plot = sns.lmplot('PCA1', 'PCA2', hue='Detroit', data=df, fit_reg=False)
plot.savefig('plot/Dimmensionality Reduction Plot.png');

## Clustering
model2 = GMM(n_components=2, covariance_type='full')
model2.fit(X)
gmm_predictions = model2.predict(X)
df['cluster'] = gmm_predictions
plot = sns.lmplot('PCA1', 'PCA2', data=df, col='cluster',  hue='Detroit', fit_reg=False)
plot.savefig('plot/Clustering Plot.png');
