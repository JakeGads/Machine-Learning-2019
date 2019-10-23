import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')

# creating independent and dependent variables
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Training and test


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .2, random_state = 0
)


lin_reg = LinearRegression()
lin_reg.fit(X,y)

# fitting Polynomial Ression to the @staticmethod

poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(
    X, 
    lin_reg2.predict(poly_reg.fit_transform(X)), 
    color = 'blue'
)

plt.title('Postopn vs Salaries')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()
