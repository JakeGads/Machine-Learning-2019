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


from classification import knn, 
from regression import poly_regression, multiple_regression, linears_regression

def q1(Xs):
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    X = dataset.iloc[:, Xs]
    y = dataset.iloc[:, [17]]
    X_perms = []
    
    
    for i in range(1, len(Xs)):
        X_perms = permutations(Xs, i)


    del Xs  # doing this becuase we are already heavy on mem

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
        print("testing the set")
        for k in range(15 + 1):
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

            print(X_list, current_accuracy)


if __name__ == "__main__":
    # data headers
    # ZOO
        # animal_name,hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail domestic,catsize,class_type
    file = "Data/zoo-animal-classification/zoo.csv"
    # Does the class type depend on the byproducts (eggs,  milk) of an animal?
    
    print("Feather and Eggs")
    q1([3,4]) # ~ 0.6774193548387096
    # Does the class type depend on the physical features (hair, feathers, toothed, backbone, fins, legs, tail) of an animal? 
    print("\n\n\nhair, feathers, toothed, backbone, fins, legs, tail")
    q1([1,2, 8, 9, 12, 13, 14])
    # Is the class type defined by the predator and venomous nature of the animal?
