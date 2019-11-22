"""
A classification problem requires that examples be classified into one of two or more classes.
A classification can have real-valued or discrete input variables.
A problem with two classes is often called a two-class or binary classification problem.
A problem with more than two classes is often called a multi-class classification problem.
A problem where an example is assigned multiple classes is called a multi-label classification problem.
"""
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
        for i in self.X:
            comb += str(i) + " "
        comb += "]"
        return f"{comb},{self.y},{self.k}, {self.accuracy_score:.2f}"


def q1(Xs):
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    y = dataset.iloc[:, [17]]
    X_perms = []

    for i in range(len(Xs) + 1):
        X_perms.extend(list(permutations(Xs, i)))

    del Xs  # doing this becuase we are already heavy on mem

    highAccuracy = Accuracy(0, 0, 0, 0)

    for X_list in X_perms:
        try:
            # ensures that we have a list
            X_list = list(X_list)
        except:
            None
            
        # make it the dataset
        X = dataset.iloc[:, X_list]
    
        # training the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        for k in range(100 + 1):
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
                    None
                try:
                    y_train = lab_enc.fit_transform(y_train)
                    y_test = lab_enc.fit_transform(y_test)
                except:
                    None

            try:

                knn_classifier.fit(X_train, y_train)

                y_pred = knn_classifier.predict(X_test)
                current_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

            except:
                current_accuracy = 0

            if current_accuracy > highAccuracy.accuracy_score:
                highAccuracy.remake(X_list, 17, k, current_accuracy)
    return highAccuracy.printable()


if __name__ == "__main__":
    # @Dr Gupta, this will take a while to run there is no need to do it on your end

    # data headers
    # ZOO
        # animal_name,hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail domestic,catsize,class_type
    file = "Data/zoo-animal-classification/zoo.csv"
    # Does the class type depend on the byproducts (eggs,  milk) of an animal?
    
    print("Feather and Eggs")
    print(q1([3,4])) # X:[4, 3], y:17, k:6, score: 0.6774193548387096

    # Does the class type depend on the physical features (hair, feathers, toothed, backbone, fins, legs, tail) of an animal? 
    print("\n\n\nhair, feathers, toothed, backbone, fins, legs, tail")
    print(q1([1,2, 8, 9, 12, 13, 14])) # X:[13, 1, 14, 9], y:17, k:1, score: 1.0
    # Is the class type defined by the predator and venomous nature of the animal?
    print("\n\npredator and venomous")
    print(q1([7,11])) # X:[7, 11], y:17, k:1, score: 0.4838709677419355

