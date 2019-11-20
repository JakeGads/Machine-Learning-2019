import numpy as np
import warnings
from itertools import permutations

import pandas as pd
import sklearn
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing as preprocessing

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from classification import convert_index

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class Accuracy:
    def __init__(self, X, y, degree, accuracy_score):
        self.X = X
        self.y = y
        self.degree = degree
        self.accuracy_score = accuracy_score

    def remake(self, X, y, accuracy_score):
        self.X = X
        self.y = y
        self.accuracy_score = accuracy_score

    def printable(self):
        return f"""
        X:{self.X}, y:{self.y}
        score: {self.accuracy_score}
        """

    def writtable(self):

        return f"{self.X},{self.y},{self.degree}" \
               f", {self.accuracy_score:.2f}"


def poly_regression(X_loc, y_loc, file, max_degree=10, supress_text=False):

    # prepares the file
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # sets my indicies
    X_loc = convert_index(dataset.columns, X_loc)
    y_loc = convert_index(dataset.columns, y_loc)

    # generate the data
    if not isinstance(X_loc, list):
        X_loc = [X_loc]
    if not isinstance(y_loc, list):
        y_loc = [y_loc]
    X = dataset.iloc[:, X_loc]
    y = dataset.iloc[:, y_loc]

    if not supress_text:
        print("All data is gathered begining to test")

    super_accuracy = Accuracy(X_loc, y_loc, 0, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    for i in range(1, max_degree):
        polynomial_features = PolynomialFeatures(degree=i)
        X_poly = polynomial_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_poly_pred = model.predict(X_poly)

        rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
        r2 = r2_score(y, y_poly_pred)

        sub_accuracy = Accuracy(X_loc, y_loc, i, r2)

        if sub_accuracy.accuracy_score > super_accuracy.accuracy_score:
            super_accuracy = sub_accuracy

    print(super_accuracy.printable())


def linears_regression(X_loc, y_loc, file, max_degree=10, supress_text=False):

    # prepares the file
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # sets my indicies
    X_loc = convert_index(dataset.columns, X_loc)
    y_loc = convert_index(dataset.columns, y_loc)

    # generate the data
    if not isinstance(X_loc, list):
        X_loc = [X_loc]
    if not isinstance(y_loc, list):
        y_loc = [y_loc]
    X = dataset.iloc[:, X_loc]
    y = dataset.iloc[:, y_loc]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    regr = LinearRegression()

    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    mean_squared_error(y_test, y_pred)
    a = Accuracy(X_loc, y_loc, 0, r2_score(y_test, y_pred))
    return a


def multiple_regression(X_locs, y_loc, file, max_degree=10, supress_text=False):

    # prepares the file
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # sets my indicies
    if isinstance(X_locs, list):
        for loc in range(X_locs):
            X_locs[loc] = convert_index(dataset.columns, X_locs[loc])
    else:
        X_locs = convert_index(dataset.columns, X_locs)

    y_loc = convert_index(dataset.columns, y_loc)

    # generate the data
    if not isinstance(X_locs, list):
        X_locs = [X_locs]
    if not isinstance(y_loc, list):
        y_loc = [y_loc]
        
    X = dataset.iloc[:, X_locs]
    y = dataset.iloc[:, y_loc]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    regr = LinearRegression()

    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    mean_squared_error(y_test, y_pred)
    a = Accuracy(X_locs, y_loc, 0, r2_score(y_test, y_pred))
    return a