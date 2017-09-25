# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics

# load data
df = pd.read_csv('E:/data/xc/samples2.csv')
X = df[['x', 'y', 'GEO', 'PLANC', 'PRECI', 'TWI', 'TEMPR', 'SLOPE']]
y = df['SOMB']
X = np.array(X)
y = np.array(y)

np.random.seed(314)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

numberList = np.arange(1, 100)
rmseList = []

for number in numberList:
    regr = ensemble.RandomForestRegressor(n_estimators=number, max_depth=4, min_samples_split=2)
    regr.fit(X_train, y_train)
    y_test_pred = regr.predict(X_test)
    mse = metrics.mean_squared_error(y_test, regr.predict(X_test))
    rmse = np.sqrt(mse)
    print(number, ':\t', rmse)
    rmseList.append(rmse)

plt.scatter(numberList, rmseList)

#for i in range(10):
#    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
#    regr = ensemble.RandomForestRegressor(n_estimators=100) #max_depth=2
#    regr.fit(X_train, y_train)
#    y_test_pred = regr.predict(X_test)
#    mse = metrics.mean_squared_error(y_true=y_test, y_pred=y_test_pred)
#    rmse = np.sqrt(mse)
#    rmseList.append(rmse)
#rmse_mean = np.mean(rmseList)
#print(rmse_mean)
