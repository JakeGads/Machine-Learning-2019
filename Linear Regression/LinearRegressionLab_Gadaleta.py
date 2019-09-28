'''
Linear regression Demo
a: Jake Gadaleta
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

grades = pd.read_csv('LinearRegression.csv')

sat = grades['SAT']
gpa = grades['GPA']

plt.title('SAT vs GPA')
plt.xlabel('SAT')
plt.ylabel('GPA')

plt.plot(sat, gpa,)
plt.grid(True)






# Linear 
x = grades.iloc[:,:-1].values
y = grades.iloc[:,1:].values

model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)

