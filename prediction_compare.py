# -*- coding: utf-8 -*-
"""
@author: Lei Zhang
@Task:   constraint kmeans sampling
"""

import os
os.chdir('E:/MyWork/myres/GeoAnalysis')

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

def getNearestElemsIds(df, sampleLocations, xname='x', yname='y'):
    index = [n for n in range(len(sampleLocations))]
    df_loc = np.array(df[[xname, yname]])
    for i in range(len(sampleLocations)):
        ix = np.argmin(np.sum((sampleLocations[i] - df_loc)**2, axis=1))
        index[i] = ix
    return index

# load environmental data
# for raffelson dataset
df = pd.read_csv('./data/envdata_raf.csv')
X = np.array(df[['env2', 'env3', 'env4', 'env5', 'env6', 'env7']])
X_factors = np.array(df[['env1']])
enc = preprocessing.OneHotEncoder()
enc.fit(X_factors)
X_factors_transformed = enc.transform(X_factors).toarray()
X = np.concatenate((X_factors_transformed, X), axis=1)
X = preprocessing.minmax_scale(X)
y = np.array(df['soiltype'])

# for heshan dataset
#df = pd.read_csv('./data/envdata_hs.csv')
#X = df[['env1', 'env2', 'env3', 'env4']]
#X = np.array(X)
#X = preprocessing.minmax_scale(X)
#y = np.array(df['soiltype'])

# load training set
train_loc_fn = './data/result/test4/kms_0+10.csv'
#train_loc_fn = './data/result/test4/cLHS_0+10.csv'
train_loc = np.array(pd.read_csv(train_loc_fn))[:,0:2]
train_index = getNearestElemsIds(df, train_loc)
X_train = X[train_index]
y_train = y[train_index]

# load test set
#data_test = np.array(pd.read_csv('./data/result/test1/valid_samples_raf.csv'))
#test_loc = data_test[:,0:2]
#test_index = getNearestElemsIds(df, test_loc)
#X_test = X[test_index]
#y_test = y[test_index]
X_test = X[:]
y_test = y[:]

# predict by rf
np.random.seed(314)
#clf = ensemble.RandomForestClassifier(n_estimators=100)
#clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf = linear_model.LogisticRegression(solver='saga', multi_class='multinomial')
#clf = linear_model.LogisticRegressionCV()
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, y_test_pred)
print('score:', round(score, 3))
