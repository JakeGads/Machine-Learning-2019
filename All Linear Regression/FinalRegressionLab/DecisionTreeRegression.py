import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
for i in range(10):
    try:
        X = df.iloc[:, i:i+1].values
    except:
        break

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