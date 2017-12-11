# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import cluster
from sklearn import preprocessing

# calculate environmental similarity
def calcEnvSimi(envList1, envList2):
    simiVector = 1 - np.abs((envList1 - envList2))
    simi = np.min(simiVector)
    return simi

def transformFactors(factors):
    factors_unique = np.unique(df['env1'])
    factorMat = np.zeros(shape=(np.shape(df)[0], len(factors_unique)))
    for i in range(factorMat.shape[0]):
        index = np.argwhere(factors_unique == factors[i])[0][0]
        factorMat[i][index] = 1
    return factorMat

# load data
df = pd.read_csv('E:/MyWork/myres/master_thesis/code/data/raffelson/envData_raf.csv')
factorsMat = transformFactors(df['env1'])

X = df[['env2', 'env3', 'env4', 'env5', 'env6', 'env7']]
X = np.array(X)
X = np.concatenate((factorsMat, X), axis=1)
X = preprocessing.minmax_scale(X)

np.random.seed(314)
res = cluster.k_means(X, n_clusters=1)

centers = res[0]
sampleList = []
for center in centers:
    sample = X[0]
    simi_max = 0.0
    for e in X:
        simi_tmp = calcEnvSimi(center, e)
        if simi_tmp > simi_max:
            simi_max = simi_tmp
            sample = e
    sampleList.append(sample)

for sample in sampleList:
    for index in range(len(X)):
        if (X[index] == sample).all() == True:
            print(str(df.x[index]) + ',' + str(df.y[index]))
