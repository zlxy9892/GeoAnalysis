# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
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

regr = xgb.XGBRegressor(
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:linear",
        nthread=4,
        scale_pos_weight=1,
        seed=314
        )
regr.fit(X_train, y_train)
mse = metrics.mean_squared_error(y_test, regr.predict(X_test))
rmse = np.sqrt(mse)
print('rmse: ', rmse)
