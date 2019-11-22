""" 
Does battery and talk time affect the price of the phone? regression

Does memory (all types) affect the price of the phone? regression

Research online to find which aspects of a phone affect the price and then use the models to confirm that theory.
"""
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


class Accuracy:
    def __init__(self, X, y, degree, accuracy_score):
        self.X = X
        self.y = y
        self.degree = degree
        self.accuracy_score = accuracy_score

    def remake(self, X, y, degree, accuracy_score):
        self.X = X
        self.y = y
        self.degree = degree
        self.accuracy_score = accuracy_score

    def printable(self):
        return f"X:{self.X}, y:{self.y}, degree:{self.degree}, score: {self.accuracy_score}"


def clean_data(file):
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # no encodeing needed

    return dataset


def polynomial(file, X_loc, y_loc, max_degree=20):
    dataset = clean_data(file)
    accuracy = Accuracy(0, 0, 0, 0)

    if not isinstance(X_loc, list):
        X_loc = [X_loc]
    if not isinstance(y_loc, list):
        y_loc = [y_loc]


    y = dataset.iloc[:, y_loc]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_perms = []

    for i in range(1, len(dataset.columns)):
        X_perms += [permutations(Xs, i)]

    X = dataset.iloc[:, X_loc]
    for i in range(1, max_degree):
        polynomial_features = PolynomialFeatures(degree=i)
        X_poly = polynomial_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_poly_pred = model.predict(X_poly)

        rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
        r2 = r2_score(y, y_poly_pred)

        if accuracy.accuracy_score < r2:
            accuracy.remake(X_loc, y_loc, i, r2)

    return accuracy.printable()


def linear_regression(file, X_loc, y_loc):
    dataset = clean_data(file)

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
    a = Accuracy(X_loc, y_loc, 1, r2_score(y_test, y_pred))
    return a.printable()


if __name__ == "__main__":
    file = "Data/mobile-price-classification/train.csv"

    columns = [
        "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory", "m_dep", "mobile_wt",
        "n_cores", "pc", "px_height", "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi",
        "price_range"
    ]

    price = [columns.index("price_range")]

    # Does battery and talk time affect the price of the phone? regression
    one = [columns.index("battery_power"), columns.index("talk_time")]

    print(f"""
    Polynomial:\t{polynomial(file, one, price)}
    Multiple:\t{linear_regression(file, one, price)}
    """)

    # Does memory (all types) affect the price of the phone? regression
    # TODO figure out what is mem in our columns

