# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection

### ---------- hyperparameters ---------- ###
LR = 0.01
MAX_LOOP = 100000
### ---------- hyperparameters ---------- ###


# neural network layer
def nn_layer(inputs, in_dim, out_dim, act=None):
    weights = tf.Variable(tf.random_normal(shape=[in_dim, out_dim]))
    biases = tf.Variable(tf.zeros(shape=[out_dim]) + 0.1)
    z = tf.matmul(inputs, weights) + biases
    z = tf.nn.dropout(z, keep_prob)
    if act is None:
        return z
    else:
        return act(z)


# load data
df = pd.read_csv('E:/data/xc/samples2.csv')
X = df[['x', 'y', 'GEO', 'PLANC', 'PROFC', 'TWI', 'TEMPR', 'SLOPE']]
y = df['SOMB']
X = np.array(X)
X = preprocessing.minmax_scale(X)
y = np.array(y)[:, np.newaxis]
np.random.seed(314)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

### start ###
# define placeholder
in_dim = np.shape(X)[1]
out_dim = 1
xs = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
ys = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
keep_prob = tf.placeholder(tf.float32)

hidden_layer_1 = nn_layer(xs, in_dim, 10, act=tf.nn.relu)
hidden_layer_2 = nn_layer(hidden_layer_1, 10, 10, act=tf.nn.relu)
hidden_layer_3 = nn_layer(hidden_layer_2, 10, 10, act=tf.nn.relu)
hidden_layer_4 = nn_layer(hidden_layer_3, 10, 10, act=tf.nn.relu)
y_pred = nn_layer(hidden_layer_4, 10, out_dim, act=None)
#y_pred = hidden_layer_1

loss = tf.reduce_mean(tf.square(y_pred - ys))
#train_step = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)
train_step = tf.train.AdamOptimizer(LR).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(MAX_LOOP):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.85})
        if i % 50 == 0:
            print('train error:\t', np.sqrt(sess.run(loss, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})))
            print('test error:\t', np.sqrt(sess.run(loss, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})))
    
    print('RMSE: ', np.sqrt(sess.run(loss, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})))









