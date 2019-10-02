import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4:5].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Spliting into a training and a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,  test_size = .25, random_state = 0
)

from sklearn.linear_model import LinearRegression as LR

regressor = LR()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test, y_test)

plt.plot(X_test, y_test, color='g')
plt.plot(X_test, y_pred, color='b')

from sklearn.metrics import mean_squared_error
import math
regression_model_mse = mean_squared_error(y_pred, y_test)
regression_model_mse_sq = math.sqrt(regression_model_mse)