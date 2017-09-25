# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
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

clf = svm.SVR(kernel='rbf')
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
mse = metrics.mean_squared_error(y_true=y_test, y_pred=y_test_pred)
rmse = np.sqrt(mse)
print(rmse)
