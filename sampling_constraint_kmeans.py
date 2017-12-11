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

def calcDist(envList1, envList2):
    dist = np.sum((envList1 - envList2)**2)
    return dist

def transformFactors(factors):
    factors_unique = np.unique(df['env1'])
    factorMat = np.zeros(shape=(np.shape(df)[0], len(factors_unique)))
    for i in range(factorMat.shape[0]):
        index = np.argwhere(factors_unique == factors[i])[0][0]
        factorMat[i][index] = 1
    return factorMat

def getClosestXIndex(df, sampleLocations):
    index = [n for n in range(len(sampleLocations))]
    for i in range(len(sampleLocations)):
        loc = sampleLocations[i]
        dist_min = 99999999
        for j in range(len(df)):
            x = df.loc[j]
            dist_tmp = (x['x'] - loc[0])**2 + (x['y'] - loc[1])**2
            if dist_tmp < dist_min:
                dist_min = dist_tmp
                index[i] = j
    return index

# load data and preprocessing
df = pd.read_csv('E:/MyWork/myres/GeoAnalysis/data/envdata_raf.csv')
factorsMat = transformFactors(df['env1'])
X = df[['env2', 'env3', 'env4', 'env5', 'env6', 'env7']]
X = np.array(X)
X = np.concatenate((factorsMat, X), axis=1)
X = preprocessing.minmax_scale(X)
#pd.DataFrame(X).to_csv('./data.csv')

# read existed samples, find the nearest pixel to the given location (x,y)
existedSamplesLoc = np.array(pd.read_csv('E:/MyWork/myres/GeoAnalysis/data/existed_samples_raf_3.csv'))
existedSampleIndex = getClosestXIndex(df, existedSamplesLoc)
centers = X[existedSampleIndex] 

isFixed = []
for i in range(len(centers)):
    isFixed.append(True)

# generate random additional samples
np.random.seed(314)
additionSampleCount = 20
randIndex = np.random.randint(0, X.shape[0], size=additionSampleCount)
centers = np.row_stack((centers, X[randIndex]))
for i in range(additionSampleCount):
    isFixed.append(False)

# constraint k-means clustering
category = [0 for n in range(len(X))]
maxloop = 100
for iterTime in range(maxloop):
    print(iterTime)
    # update category
    for i in range(len(X)):
        #simi_max = 0.0
        dist_min = 0.0
        for j in range(len(centers)):
            dist_tmp = calcDist(X[i], centers[j])
            if dist_min < dist_tmp:
                dist_min = dist_tmp
                category[i] = j
    # update centers
    isNoMove = True
    for j in range(len(centers)):
        if isFixed[j]:
            continue
        featSum = np.array([0.0 for n in range(X.shape[1])])
        count = 0
        for i in range(len(X)):
            if category[i] == j:
                featSum += X[i]
                count += 1
        newCenter = featSum / count
        oldCenter = centers[j]
        if calcDist(oldCenter, newCenter) > 0.001:
            isNoMove = False
        centers[j] = newCenter
    # terminal
    if isNoMove:
        print('cluster done!')
        break

#np.random.seed(314)
#res = cluster.k_means(X, n_clusters=1)
#centers = res[0]

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
