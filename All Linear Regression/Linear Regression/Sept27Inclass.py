# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:32:48 2019

@author: Instructor
"""

'''
Scikit-learn
'''
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()

print(iris.DESCR)

diabetes = datasets.load_diabetes()
breast_cancer = datasets.load_breast_cancer()

print(iris.data)
print(iris.feature_names)

import pandas as pd
df = pd.DataFrame(iris.data)


import matplotlib.pyplot as plt
df = pd.read_csv("E:\\CS400\BMI.csv")

heights = df['Heights']
weights = df['Weights']

plt.title('Healthy BMI')
plt.xlabel('Heights in cms')
plt.ylabel('Weights in pounds')

plt.plot(heights, weights, 'k.')
plt.axis([0, 200, 0, 200])
plt.grid(True) 

#independent variable
X = df.iloc[:, :-1].values
#dependent variable
y = df.iloc[:, 1:].values

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

r_sq = model.score(X, y)
print('coefficient of determination:', r_sq)

weight = model.predict([[165.00]])
np.set_printoptions(precision = 2)
print(weight)

plt.plot(X, model.predict(X), color ='g')

print(model.intercept_)
print(model.coef_)

y_pred = model.intercept_ + model.coef_ * X

import pickle

filename = 'E:\\CS400\\model1.sav'

pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.score(X,y)











