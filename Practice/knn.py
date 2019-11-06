import warnings
from collections import Counter
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
1. Classification (RI)
2. Regression
"""
"""
        try:
            knn_classifier.fit(X_train, y_train)
        except:
            lab_enc = preprocessing.LabelEncoder()
            try:
                X_train = lab_enc.fit_transform(X_train)
                X_test = lab_enc.fit_transform(X_test)
            except:
                None
            try:
                y_train = lab_enc.fit_transform(y_train)
                y_test = lab_enc.fit_transform(y_test)
            except:
                None
        try:
            knn_classifier.fit(X_train, y_train)
            knn_scores.append(knn_classifier.score(X_test, y_test))
            print("\t\tPassed")
        except:
            knn_scores.append(0.0)
            print("Failed")
"""


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


def knn(file, y, max_k=100, max_perm=0):
    # open loads and cleans data
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # Makes y and converts it to an index and applys it with the dataset
    # X is generated On The Fly
    y = convert_index(dataset.columns, y)
    y = dataset.iloc[:, y]
    # getting my X's
    Xs = list(range(len(dataset.columns)))

    # generates permustions
    X_perms = []
    if not max_perm:
        X_perms = permutations(Xs, max_perm)
    else:
        for i in range(1, len(dataset.columns)):
            X_perms.extend(permutations(Xs, max_perm))

    del Xs  # doing this becuase we are already heavy on mem

    super_counter = 0

    for X_list in X_perms:
        X_list = list(X_list)
        X = dataset.iloc[:, X_list]
        del X_list
        
# Wrote KNN functiality by hand to understand what was happening under the covers

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return distance ** .5


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


if __name__ == "__main__":
    # Test cases
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    prediction = predict_classification(dataset, dataset[0], 3)
    print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
