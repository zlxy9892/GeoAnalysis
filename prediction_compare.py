# -*- coding: utf-8 -*-
"""
@author: Lei Zhang
@Task:   constraint kmeans sampling
"""

import os
os.chdir('D:/MyWork/myres/GeoAnalysis')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn import neighbors
from sklearn import linear_model

def calcDist(v1, v2, sum_axis=None):
    dist = 0.
    if sum_axis is None:
        dist = np.sum((v1 - v2)**2)
    else:
        dist = np.sum((v1 - v2)**2, axis=sum_axis)
    return dist

def transformFactors(df, factor_name):
    factors_unique = np.unique(df[factor_name])
    factorMat = np.zeros(shape=(np.shape(df)[0], len(factors_unique)))
    for i in range(factorMat.shape[0]):
        index = np.argwhere(factors_unique == df[factor_name][i])[0][0]
        factorMat[i][index] = 1
    return factorMat

def getNearestElemsIds(df, sampleLocations, xname='x', yname='y'):
    index = [n for n in range(len(sampleLocations))]
    df_loc = np.array(df[[xname, yname]])
    for i in range(len(sampleLocations)):
        ix = np.argmin(np.sum((sampleLocations[i] - df_loc)**2, axis=1))
        index[i] = ix
    return index

# load environmental data
df = pd.read_csv('./data/envdata_raf.csv')
X = np.array(df[['env2', 'env3', 'env4', 'env5', 'env6', 'env7']])
factorsMat = transformFactors(df, 'env1')
X = np.concatenate((factorsMat, X), axis=1)
X = preprocessing.minmax_scale(X)
y = np.array(df['soiltype'])

# load training set
#train_loc_fn = './data/result/test1/kms_10+50.csv'
train_loc_fn = './data/result/test1/cLHS_10+50.csv'
train_loc = np.array(pd.read_csv(train_loc_fn))[:,0:2]
train_index = getNearestElemsIds(df, train_loc)
X_train = X[train_index]
y_train = y[train_index]

# load test set
data_test = np.array(pd.read_csv('./data/result/test1/valid_samples_raf.csv'))
test_loc = data_test[:,0:2]
test_index = getNearestElemsIds(df, test_loc)
X_test = X[test_index]
y_test = y[test_index]

# predict by rf
np.random.seed(314)
#clf = ensemble.RandomForestClassifier(n_estimators=1000)
#clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, y_test_pred)
print('score:', score)
