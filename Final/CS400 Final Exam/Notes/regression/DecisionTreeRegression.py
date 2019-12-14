import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


from sklearn.model_selection import train_test_split


df = pd.read_csv("housing.csv")
y = df.iloc[:, 2].values

headers = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "housingholds",
    "median_income",
    "median_house_value",
    "ocean_proximity"
]
"""
for i in range(len(headers)):
    try:
        X = df.iloc[:, i:i+1].values
    except:
        break

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    regressor = DecisionTreeRegressor(random_state = 0)
    try:
        regressor.fit(X_train, y_train)
    except:
        print(f'Skipping {headers[i]} because of a regressor error')
        continue

    y_pred = regressor.predict(X_test)

    X_grid = np.arange(min(X), max(X), .01)
    X_grid = X_grid.reshape((len(X_grid), 1))

    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = "blue")

    plt.title(f'Decision tree:\n{headers[i]}')
    plt.show()
"""

# answer is housing median age

i = 2 # median age loc
try:
    X = df.iloc[:, i:i+1].values
except:
    exit


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

regressor = DecisionTreeRegressor(random_state = 0)
try:
    regressor.fit(X_train, y_train)
except:
    print(f'Skipping {headers[i]} because of a regressor error')
    

y_pred = regressor.predict(X_test)

X_grid = np.arange(min(X), max(X), .01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")

plt.title(f'Decision tree:\n{headers[i]} x Median_House_Value')
plt.show()
