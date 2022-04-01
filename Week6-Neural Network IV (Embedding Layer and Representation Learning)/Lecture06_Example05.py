# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:03:09 2018

@author: jgao5111
adopted from
http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from sklearn import decomposition
from sklearn import datasets
from sklearn import preprocessing

np.random.seed(5)

iris_train_df = pd.read_csv('Lec5_iris.csv')
#iris_train_df = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris_train_df.iloc[:,0:4].values

# Actually PCA does not need target information as it is an unsupervised learning
# We extract the target for visualisation purpose
t = iris_train_df.iloc[:,4]
le = preprocessing.LabelEncoder()
le.fit(t)
t = le.transform(t)
class_names = le.classes_


#centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)   # We use 3 component to visualise 4D data X
pca.fit(X)
X3 = pca.transform(X)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X3[y == label, 0].mean(),
              X3[y == label, 1].mean() + 1.5,
              X3[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
t = np.choose(t, [1, 2, 0]).astype(np.float)
ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=t, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
