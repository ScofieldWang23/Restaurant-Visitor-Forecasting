#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:25:08 2018

@author: wsf
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

iris = load_iris()

X = iris.data
y = iris.target

k_range = list(range(1, 31))
# create a parameter grid: map the parameter names to the values that should be searched
# 下面是构建parameter grid，其结构是key为参数名称，value是待搜索的数值列表的一个字典结构
param_grid = dict(n_neighbors=k_range)
knn = KNeighborsClassifier(n_neighbors=5)

# instantiate the grid
# 这里GridSearchCV的参数形式和cross_val_score的形式差不多，其中param_grid是parameter grid所对应的参数
# GridSearchCV中的n_jobs设置为-1时，可以实现并行计算（如果你的电脑支持的情况下）
#我们可以知道，这里的grid search针对每个参数进行了5次交叉验证，并且一共对30个参数进行相同过程的交叉验证

grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

grid.fit(X, y)
grid.grid_scores_

# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]

plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
           
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


###### 同时对多个参数进行搜索 ######
'''
这里我们使用knn的两个参数，分别是n_neighbors和weights，其中weights参数默认是uniform，该参数将所有数据看成等同的，
而另一值是distance，它将近邻的数据赋予更高的权重，而较远的数据赋予较低权重。
'''
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']

param_grid = dict(n_neighbors=k_range, weights=weight_options)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print(grid.best_score_)
print(grid.best_params_)

# train your model using all data and the best known parameters
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)
# make a prediction on out-of-sample data
knn.predict([3, 5, 4, 2])

# shortcut: GridSearchCV automatically refits the best model using all of the data
grid.predict([3, 5, 4, 2])




############## 使用RandomizeSearchCV来降低计算代价 ##############
'''
RandomizeSearchCV用于解决多个参数的搜索过程中计算代价过高的问题
RandomizeSearchCV搜索参数中的一个子集，这样你可以控制计算代价 
'''
from sklearn.grid_search import RandomizedSearchCV

# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)

# n_iter controls the number of searches
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
rand.fit(X, y)
rand.grid_scores_

print(rand.best_score_)
print(rand.best_params_)

# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
    
'''
当你的调节参数是连续的，比如回归问题的正则化参数，有必要指定一个连续分布而不是可能值的列表，
这样RandomizeSearchCV就可以执行更好的grid search。
'''

























