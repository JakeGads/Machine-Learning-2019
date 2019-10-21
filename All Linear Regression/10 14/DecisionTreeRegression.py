import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Position_Salaries.csv")

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

X_grid = np.arange(min(X), max(X), .01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")

plt.title('Decision tree ')
plt.show()