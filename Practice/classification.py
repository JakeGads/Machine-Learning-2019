import warnings
from itertools import permutations

import pandas as pd
import sklearn
import sklearn.preprocessing as preprocessing
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
1. Classification (RI)
2. Regression
"""


class Accuracy:
    def __init__(self, X, y, accuracy_score):
        self.X = X
        self.y = y
        self.accuracy_score = accuracy_score

    def remake(self, X, y, accuracy_score):
        self.X = X
        self.y = y
        self.accuracy_score = accuracy_score

    def printable(self):
        return f"X:{self.X}, y:{self.y} score: {self.accuracy_score}"

    def writtable(self):
        comb = "["
        for i in self.X:
            comb += str(i) + " "
        comb += "]"
        return f"{comb},{self.y},{self.accuracy_score:.2f}"

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


def knn_definedTestSplit(test_file, train_file, y, max_k = 100, max_perm = 3, supress_text = False):
    # open loads and cleans data
    test_dataset = pd.read_csv(test_file)
    test_dataset = test_dataset.dropna()
    test_dataset = test_dataset.reset_index(drop=True)

    train_dataset = pd.read_csv(train_file)
    train_dataset = train_dataset.dropna()
    train_dataset = train_dataset.reset_index(drop=True)

     # Makes y and converts it to an index and applys it with the dataset
    # X is generated On The Fly
    y = convert_index(dataset.columns, y)
    y_loc = y
    test_y = test_dataset.iloc[:, [y]]
    train_y = train_dataset.iloc[:, [y]]

    # getting my X's
    Xs = list(range(len(dataset.columns)))

    # generates permustions
    X_perms = []
    if not max_perm:
        for i in range(1, len(dataset.columns)):
            X_perms = permutations(Xs, i)
    else:
        for i in range(1, max_perm + 1):
            X_perms += list(permutations(Xs, i))

    del Xs  # doing this becuase we are already heavy on mem

    super_counter = 0
    super_accuracy = Accuracy(0, 0, 0)
    for X_list in X_perms:
        # ensures that we have a list
        X_list = list(X_list)
        # make it the dataset
        test_X = test_dataset.iloc[:, X_list]
        train_X = train_dataset.iloc[:, X_list]

        # to hold precentage 
        super_counter += 1

        # a defualted accuracry with a minal set of 0
        sub_accuracy = Accuracy(0, 0, 0)

        # loops through all my possible ks
        for k in range(max_k + 1):
            if not supress_text: # if you selected to allow text prints out the super and sup percentage
                print(f"super: {100 * (super_counter / len(X_perms)):.2f}% sub: {100 * (k / max_k):.2f}%", end=" ")
            # classifies with the current k as a value
            knn_classifier = KNeighborsClassifier(n_neighbors=k)

            # try catch chain to trainsforms when needed, if it fails it will let you know
            try:
                knn_classifier.fit(train_X, train_y)
            except:
                lab_enc = preprocessing.LabelEncoder()
                try:
                    X_train = lab_enc.fit_transform(X_train)
                    X_test = lab_enc.fit_transform(X_test)
                except:
                    if not supress_text:
                        print("X is a nope", end=" ")
                try:
                    y_train = lab_enc.fit_transform(y_train)
                    y_test = lab_enc.fit_transform(y_test)
                except:
                    if not supress_text:
                        print("y is a nope", end=" ")

            # trys to refit predic and then calculate the accuracy
            try:
                knn_classifier.fit(X_train, y_train)
                y_pred = knn_classifier.predict(X_test)
                current_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

                if current_accuracy > sub_accuracy.accuracy_score: # checks to see if the new accuarcy is bigger than the largest and reassings if it is 
                    sub_accuracy.remake(X_list, y_loc, current_accuracy)

                if not supress_text:
                    print("Passed")

            except:
                if not supress_text:
                    print("Failed")
        # checks to see if the accuracy for that perm is getter than the overall accuracy
        if sub_accuracy.accuracy_score > super_accuracy.accuracy_score:
            super_accuracy = sub_accuracy
    # prints the score
    print(super_accuracy.printable())
    # returns it, for writing
    return super_accuracy



def knn(file, y, max_k=100, max_perm=3, supress_text=False):
    # open loads and cleans data
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # Makes y and converts it to an index and applys it with the dataset
    # X is generated On The Fly
    y = convert_index(dataset.columns, y)
    y_loc = y
    y = dataset.iloc[:, [y]]
    # getting my X's
    Xs = list(range(len(dataset.columns)))

    # generates permustions
    X_perms = []
    if not max_perm:
        for i in range(1, len(dataset.columns)):
            X_perms = permutations(Xs, i)
    else:
        for i in range(1, max_perm + 1):
            X_perms += list(permutations(Xs, i))

    del Xs  # doing this becuase we are already heavy on mem

    super_counter = 0
    super_accuracy = Accuracy(0, 0, 0)
    for X_list in X_perms:
        # ensures that we have a list
        X_list = list(X_list)
        # make it the dataset
        X = dataset.iloc[:, X_list]

        # training the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # to hold precentage 
        super_counter += 1

        # a defualted accuracry with a minal set of 0
        sub_accuracy = Accuracy(0, 0, 0)

        # loops through all my possible ks
        for k in range(max_k + 1):
            if not supress_text: # if you selected to allow text prints out the super and sup percentage
                print(f"super: {100 * (super_counter / len(X_perms)):.2f}% sub: {100 * (k / max_k):.2f}%", end=" ")
            # classifies with the current k as a value
            knn_classifier = KNeighborsClassifier(n_neighbors=k)

            # try catch chain to trainsforms when needed, if it fails it will let you know
            try:
                knn_classifier.fit(X_train, y_train)
            except:
                lab_enc = preprocessing.LabelEncoder()
                try:
                    X_train = lab_enc.fit_transform(X_train)
                    X_test = lab_enc.fit_transform(X_test)
                except:
                    if not supress_text:
                        print("X is a nope", end=" ")
                try:
                    y_train = lab_enc.fit_transform(y_train)
                    y_test = lab_enc.fit_transform(y_test)
                except:
                    if not supress_text:
                        print("y is a nope", end=" ")

            # trys to refit predic and then calculate the accuracy
            try:
                knn_classifier.fit(X_train, y_train)
                y_pred = knn_classifier.predict(X_test)
                current_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

                if current_accuracy > sub_accuracy.accuracy_score: # checks to see if the new accuarcy is bigger than the largest and reassings if it is 
                    sub_accuracy.remake(X_list, y_loc, current_accuracy)

                if not supress_text:
                    print("Passed")

            except:
                if not supress_text:
                    print("Failed")
        # checks to see if the accuracy for that perm is getter than the overall accuracy
        if sub_accuracy.accuracy_score > super_accuracy.accuracy_score:
            super_accuracy = sub_accuracy
    # prints the score
    print(super_accuracy.printable())
    # returns it, for writing
    return super_accuracy


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
