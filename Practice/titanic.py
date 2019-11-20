"""
Apply all classification models to the titanic dataset

Question - Does sex, siblings, fare, embarked and who is traveling decide the class of that travel.
"""

import warnings
from itertools import permutations

import pandas as pd
import sklearn
import sklearn.preprocessing as preprocessing
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

    def writtable(self):
        comb = "["
        try:
            for i in self.X:
                comb += str(i) + " "
        except:
            comb += str(self.X)
        comb += "]"
        return f"{comb},{self.y},{self.k}, {self.accuracy_score:.2f}"

def knn(file, y, Xs, max_k=100, supress_text=False):
    # open loads and cleans data
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # creating a dict file
    dataset.sex.unique()
    # converting categorical data
    sex = {'male': 1, 'female': 2}

    dataset.embarked.unique()
    embarked = {'S': 1, 'C': 2, 'Q': 3, 0: 0}

    dataset.who.unique()
    who = {'man': 1, 'woman': 2, 'child': 3}

    dataset.deck.unique()
    deck = {'C': 1, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 7, 0: 0}

    dataset.embark_town.unique()
    embark_town = {'Southampton': 1, 'Cherbourg': 2, 'Queenstown': 3, 0: 0}

    dataset.alive.unique()
    alive = {'no': 1, 'yes': 2}

    classV = {'First': 1, 'Second': 2, 'Third': 3}

    dataset.sex = [sex[item] for item in dataset.sex]
    dataset.embarked = [embarked[item] for item in dataset.embarked]
    dataset.who = [who[item] for item in dataset.who]
    dataset.deck = [deck[item] for item in dataset.deck]
    dataset.embark_town = [embark_town[item] for item in dataset.embark_town]
    dataset.alive = [alive[item] for item in dataset.alive]
    dataset.classV = [classV[item] for item in dataset.classV]

    # Makes y and converts it to an index and applys it with the dataset
    # X is generated On The Fly

    y_loc = y
    y = dataset.iloc[:, y]

    X = dataset.iloc[:, Xs]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    accuracy = Accuracy(0, 0, 0, 0)

    for k in range(1, max_k + 1):
        # classifies with the current k as a value
        knn_classifier = KNeighborsClassifier(n_neighbors=k)

        # trys to refit predic and then calculate the accuracy
        try:
            try:
                knn_classifier.fit(X_train, y_train)
            except:
                knn_classifier.fit(X_train, y_train.values.ravel())
            y_pred = knn_classifier.predict(X_test)
            current_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

            if current_accuracy > accuracy.accuracy_score:  # checks to see if the new accuarcy is bigger than the largest and reassings if it is
                accuracy.remake(Xs, y_loc, current_accuracy)

            if not supress_text:
                print("Passed")

        except:
            if not supress_text:
                print("Failed")

    print(accuracy.printable())


if __name__ == "__main__":
    # survived,pclass,sex,age,sibsp,parch,fare,embarked,classV,who,adult_male,deck,embark_town,alive,alone
    columns = ["survived", "pclass", "sex","age", "sibsp", "parch", "fare", "embarked", "classV", "who", "adult_male",
               "deck","embark_town","alive","alone"]
    x = [
        columns.index("sex"),
        columns.index("sibsp"),
        columns.index("fare"),
        columns.index("embarked"),
        columns.index("who")
    ]
    y = [columns.index("pclass")]
    del columns

    file = "Data/titanic.csv"

    knn(file, y, x)

