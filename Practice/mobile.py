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

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


class Accuracy:
    def __init__(self, X, y, k, accuracy_score):
        self.X = X
        self.y = y
        self.k = k
        self.accuracy_score = accuracy_score

    def remake(self, X, y, k, accuracy_score):
        self.X = X
        self.y = y
        self.k = k
        self.accuracy_score = accuracy_score

    def printable(self):
        return f"X:{self.X}, y:{self.y}, k:{self.k}, score: {self.accuracy_score}"


def convert_index(data, y):
    if not isinstance(y, int):
        for i in range(len(data)):
            if data[i] == y:
                y = i
                return y
        if not isinstance(y, int):
            print("Failed to generate a index for y")
            exit()

    return y

def poly_regression(X_loc, y_loc, file, max_degree=10, supress_text=False):

    # prepares the file
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # sets my indicies
    if isinstance(X_loc, list):
        for i in range(len(X_loc)):
            X_loc[i] = convert_index(dataset.colmns, X_loc[i])
    else:
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


if __name__ == "__main__":
    train_data = 'Data/mobile-price-classification/train.csv'
