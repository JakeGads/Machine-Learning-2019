import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, PolynomialFeatures)


def linear(file, X, y):
    """
    Applys Linear Regression to the model
    Shows the PLT and the train values

    Parameters:
    file (str): The location of the .csv it will read
    X (str): The name of the indenpent col
    y (str): The name of the depenent colum 

   """

    df = pd.read_csv("file")
    df = df.dropna()
    df = df.reset_index(drop=True)

    plt.title(f"{X} vs {y}")
    plt.xlabel(X)
    plt.ylabel(y)
    plt.plot(df[X], y, 'c.')
    plt.grid(True)

    line_X = df[y].values
    line_y = df[X].values

    model = LR()

    model.fit(line_X, line_y)

    plt.plot(line_X, model.predict(line_X), color = 'r')

    plt.savefig(plt.title(f"{X} vs {y}"))

def multiple(file):
    None

def polynomial(file):
    None
