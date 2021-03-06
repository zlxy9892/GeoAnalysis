# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
#    params = {'n_estimators': number, 'max_depth': 4, 'min_samples_split': 2,
#              'learning_rate': 0.01, 'loss': 'ls'}
#    clf = ensemble.GradientBoostingRegressor(**params)
    clf = ensemble.GradientBoostingRegressor(n_estimators=number)
    clf.fit(X_train, y_train)
    mse = metrics.mean_squared_error(y_test, clf.predict(X_test))
    rmse = np.sqrt(mse)
    print(number, ':\t', rmse)
    rmseList.append(rmse)

plt.scatter(numberList, rmseList)
