# -*- coding: utf-8 -*-
"""
@author: Lei Zhang
@Task:   Functions for constraint kmeans sampling
"""
import pandas as pd
import numpy as np
#from sklearn import cluster
from sklearn import preprocessing


# calculate environmental similarity
def calcEnvSimi(envList1, envList2):
    simiVector = 1 - np.abs((envList1 - envList2))
    simi = np.min(simiVector)
    return simi

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

def showSampleLocByNearestFeatures(df, X, centers, xname='x', yname='y'):
    df_loc = []
    for center in centers:
        id = np.argmin(calcDist(X, center, sum_axis=1))
        #print(str(df[xname][id]) + ',' + str(df[yname][id]))
        print(str(X[id][0]) + ',' + str(X[id][1]))
        df_loc.append([df[xname][id], df[yname][id]])
    return df_loc

# constraint k-means clustering
def ckms(X, n_clusters, prior=None, max_iter=50, tol=1e-4, ntry=1, trace=False):
    if trace:
        print('start ckms...')
    # init variables
    bestCenters = []
    centers = []
    isFixed = []
    if prior is not None:
        for i in range(len(prior)):
            isFixed.append(True)
    for i in range(n_clusters):
        isFixed.append(False)
    # start kmeans
    categories = np.array([0 for n in range(len(X))])
    dist_sum_min = 10.0**100
    dist_sum_tmp = 0.0
    for tryTime in range(ntry):
        # refresh centers
        centers = []
        if prior is not None:
            centers = prior
            randIndex = np.random.randint(0, X.shape[0], size=n_clusters)
            centers = np.row_stack((centers, X[randIndex]))
        else:
            randIndex = np.random.randint(0, X.shape[0], size=n_clusters)
            centers = X[randIndex]
        dist_sum_tmp = 0.0
        categories_tmp = np.zeros(len(X))
        for iterTime in range(max_iter):
            # update categories for each element of X (E step)
            dist_sum_tmp = 0.0
            for i in range(len(X)):
                id_cate = np.argmin(calcDist(X[i], centers, sum_axis=1))
                categories_tmp[i] = id_cate
            # update centers (M step)
            isNoMove = True
            for j in range(len(centers)):
                if isFixed[j]:
                    continue
                belong_center_ids = np.where(categories_tmp == j)
                if len(belong_center_ids[0]) <= 0:
                    continue
                newCenter = np.mean(X[belong_center_ids], axis=0)
                oldCenter = centers[j]
                if calcDist(oldCenter, newCenter) > tol:
                    isNoMove = False
                centers[j] = newCenter
            # terminal
            if isNoMove:
                break
            # show the total distance in current iteration
            if trace:
                dist_sum_tmp = 0.0
                for j in range(len(centers)):
                    belong_center_ids = np.where(categories_tmp == j)
                    dist_sum_tmp += calcDist(centers[j], X[belong_center_ids])
                print('iter:', iterTime+1, ' dist_sum_tmp:', dist_sum_tmp)
        # recalculate the total distance
        dist_sum_tmp = 0.0
        for j in range(len(centers)):
            belong_center_ids = np.where(categories_tmp == j)
            dist_sum_tmp += calcDist(centers[j], X[belong_center_ids])
        if dist_sum_min > dist_sum_tmp:
            dist_sum_min = dist_sum_tmp
            categories = categories_tmp
            bestCenters = centers
        if trace:
            print('### try time:', tryTime+1, ' dist_sum_tmp:', dist_sum_tmp, ' dist_sum_min:', dist_sum_min, '###')
    if trace:
        print('cluster done')
    res = []
    res.append(bestCenters)
    res.append(categories)
    res.append(dist_sum_min)
    return res



# load data and preprocessing
#df = pd.read_csv('./data/envdata_raf.csv')
#X = df[['env2', 'env3', 'env4', 'env5', 'env6', 'env7']]
#X = np.array(X)
#factorsMat = transformFactors(df['env1'])
#X = np.concatenate((factorsMat, X), axis=1)
#X = preprocessing.minmax_scale(X)
#pd.DataFrame(X).to_csv('./data.csv')

# read existed samples, find the nearest pixel to the given location (x,y)
#existedSamplesLoc = np.array(pd.read_csv('./data/result/test1/legacy_10.csv'))
#existedSampleIndex = getNearestXIndex(df, existedSamplesLoc)
#priorCenters = X[existedSampleIndex]

# ckms
#res = ckms(X, n_clusters=20, prior=None, max_iter=100, ntry=3)
#centers = res[0]
#categories = res[1]

# show centers points location
#showSampleLocByNearestFeatures(df, X, centers)
