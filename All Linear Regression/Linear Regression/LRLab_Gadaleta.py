'''
Linear regression Demo
a: Jake Gadaleta
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

grades = pd.read_csv('/home/jake/Documents/code/Machine-Learning-2019/Linear Regression/' + 'LinearRegression.csv')

sat = grades['SAT']
gpa = grades['GPA']

plt.title('SAT vs GPA')
plt.xlabel('SAT')
plt.ylabel('GPA')

plt.plot(sat, gpa, 'k.')
plt.grid(True)


# Linear 
x = grades.iloc[:,:-1].values
y = grades.iloc[:,1:].values

model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
print('coefficient of determination:', r_sq)

plt.plot(x, model.predict(x), color ='b')

check = model.predict([[1664.00]])
np.set_printoptions(precision = 2)
print('check[1664]', check)

print("model intercept", model.intercept_)
print("model cof", model.coef_)

y_pred = model.intercept_ + model.coef_ * x
    