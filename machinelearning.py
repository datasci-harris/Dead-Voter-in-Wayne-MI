import utility

over100_registered_MI = pd.read_json(Path(os.getcwd())/'out/registered_dead_voters.json', orient=str)
registered_incomplete_wayne = pd.read_csv(Path(os.getcwd())/'mi_wa_voterfile.csv')

wayne_zip = registered_incomplete_wayne['zip_code'].unique().tolist()
over100_wayne = over100_registered_MI.loc[over100_registered_MI['ZipCode'].isin(wayne_zip)]

rename = {'first_name':'FirstName', 'last_name':'LastName', 'birth_year':'BirthYear', 'zip_code':'ZipCode'}
whole = pd.concat([over100_wayne, registered_incomplete_wayne.rename(columns=rename)], join='inner')
whole.head()
over100(whole, 'BirthYear', 2020)
whole1 = whole.groupby(['ZipCode', 'over100']).size().reset_index(name='count')
whole1 = whole1.pivot(index='ZipCode', columns='BirthYear', values='Count')
whole1.columns.name = None                                                  # remove columns name, 'LineCode'
whole1 = whole1.reset_index()                                       # index to columns
whole.head()


def organize_byzip(dataset, variable):
    dataset = dataset.loc[dataset['ZipCode'].isin(detroit['zip'])]
    dataset = dataset.groupby(['ZipCode', variable]).size().reset_index(name='Count')
    dataset = dataset.pivot(index='ZipCode', columns=variable, values='Count')
    dataset.columns.name = None                                                  # remove columns name, 'LineCode'
    afterzip = dataset.reset_index()                                       # index to columns

    return afterzip

organize_byzip(whole, 'over100')










import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import utility                  # Self-created

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from pathlib import Path


dead_voted = pd.read_json(Path(os.getcwd())/'out/dead_voters_who_voted.json', orient=str)
dead_registered = pd.read_json(Path(os.getcwd())/'out/registered_dead_voters.json', orient=str)
registered = pd.read_csv(Path(os.getcwd())/'mi_wa_voterfile.csv')


# Introduce gross income indicator into deadvoter analysis
income = pd.read_csv(Path(os.getcwd())/'income by zipcode.csv').dropna()
income.rename(columns = {'zipcode':'ZipCode'}, inplace = True)
income["ZipCode"]=income["ZipCode"].astype(int)

df = pd.merge(dead_registered, income, on=['ZipCode']) # Merge income and registered dead voters datasets
df["Voted"] = df["Voted"] + 0 # Binary on turnout

#Turnout prediction using SVC model
X = df[['BirthYear','income']]
Y = df['Voted']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=123)

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predict = model.predict(X_test)


print(classification_report(Y_test, predict))

#Cross-validation


models = [('Dec Tree', DecisionTreeClassifier()),
          ('Lin Disc', LinearDiscriminantAnalysis()),
          ('SVC', SVC(gamma='auto'))]
results = []

for name, model in models:
    kf = StratifiedKFold(n_splits=10)
    res = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')
    res_mean = round(res.mean(), 4)
    res_std  = round(res.std(), 4)
    results.append((name, res_mean, res_std))

for line in results:
    print(line[0].ljust(10), str(line[1]).ljust(6), str(line[2]))

# Model Selection
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
def test_on(test_set, model):
    assert isinstance(model, str), 'Either Decision Tree or Linear Discriminate, as string.'
    X_test, Y_test = x_y_split_nadrop(test_set)

    if model == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model == 'Linear Discriminate':
        model = LinearDiscriminantAnalysis()
    predict = model.fit(X_train, Y_train).predict(X_test)
    predict = pd.DataFrame(predict)
    result = pd.concat([test_set['State'], Y_test, predict], axis = 1)
    result.columns = ['State', 'Result', 'Predict']

    mat = confusion_matrix(Y_test, predict)
    ax = sns.heatmap(mat, square = True, annot = True, cbar = False)
    matplotlib.pyplot.show(ax)

    return result
