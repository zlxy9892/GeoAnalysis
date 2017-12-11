# -*- coding: utf-8 -*-
"""
@author: Lei Zhang
@Task:   Constraint kmeans sampling
"""

import os
os.chdir('D:/MyWork/myres/GeoAnalysis')

import pandas as pd
import numpy as np
#from sklearn import cluster
from sklearn import preprocessing
import functions4kms as kms


# load data and preprocessing
df = pd.read_csv('./data/envdata_raf.csv')
X = df[['env2', 'env3', 'env4', 'env5', 'env6', 'env7']]
X = np.array(X)
factorsMat = kms.transformFactors(df, 'env1')
X = np.concatenate((factorsMat, X), axis=1)
X = preprocessing.minmax_scale(X)
#pd.DataFrame(X).to_csv('./data.csv')

# read existed samples, find the nearest pixel to the given location (x,y)
legacySamplesLoc = np.array(pd.read_csv('./data/result/test1/legacy_10.csv'))
legacySampleIndex = kms.getNearestElemsIds(df, legacySamplesLoc)
priorCenters = X[legacySampleIndex]

# ckms
np.random.seed(272)
res = kms.ckms(X, n_clusters=50, prior=priorCenters, max_iter=100, ntry=3, trace=True)
#res = kms.ckms(X, n_clusters=40, prior=None, max_iter=100, ntry=5, trace=True)
centers = res[0]
categories = res[1]

# show centers points location
kms.showSampleLocByNearestFeatures(df, X, centers)
