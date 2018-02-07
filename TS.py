#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:24:29 2018

@author: wsf
"""

#Time Series Split
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]]) #6行2列
y = np.array([1, 2, 3, 4, 5, 6])
tscv=TimeSeriesSplit(n_splits=5)

for train,test in tscv.split(X):
    print("%s %s"%(train,test))

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    



































