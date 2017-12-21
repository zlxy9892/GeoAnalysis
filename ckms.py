# -*- coding: utf-8 -*-
"""
@author: Lei Zhang
@Task:   Constraint kmeans sampling
"""

#import os
#os.chdir('D:/MyWork/myres/GeoAnalysis')

import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn import preprocessing
import functions4kms as kms


# load data and preprocessing
# for raffelson dataset
df = pd.read_csv('./data/envdata_raf.csv')
X = df[['env2', 'env3']]#, 'env4', 'env5', 'env6', 'env7']]
X = np.array(X)
#X_factors = np.array(df[['env1']])
#enc = preprocessing.OneHotEncoder()
#enc.fit(X_factors)
#X_factors_transformed = enc.transform(X_factors).toarray()
#X = np.concatenate((X_factors_transformed, X), axis=1)
X = preprocessing.minmax_scale(X)

# for heshan dataset
#df = pd.read_csv('./data/envdata_hs.csv')
#X = df[['env1', 'env2', 'env3', 'env4']]
#X = np.array(X)
#X = preprocessing.minmax_scale(X)

# read existed samples, find the nearest pixel to the given location (x,y)
#legacySamplesLoc = np.array(pd.read_csv('./data/result/test1/legacy_10.csv'))
#legacySampleIndex = kms.getNearestElemsIds(df, legacySamplesLoc)
#priorCenters = X[legacySampleIndex]

# ckms
samplesize = 10
np.random.seed(272)
#res = kms.ckms(X, n_clusters=samplesize, prior=priorCenters, max_iter=100, ntry=10, trace=True)
#res = kms.ckms(X, n_clusters=samplesize, prior=None, max_iter=100, ntry=20, trace=True)
res = cluster.k_means(X, n_clusters=samplesize, verbose=True, n_init=10)
centers = res[0]
categories = res[1]
dist = res[2]

# show centers points location
df_loc = kms.showSampleLocByNearestFeatures(df, X, centers)
pd.DataFrame(df_loc).to_csv('./data/result/test6/kms_0+'+str(samplesize)+'.csv', index=False, header=['x', 'y'])
