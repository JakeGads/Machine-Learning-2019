import warnings
from collections import Counter
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
1. Classification (RI)
2. Regression
"""

def knn(file, y):
    # open the df and prep it
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)
    
    # adds scoring lists
    scoringList = []
    columnList1 = []
    columnList2 = []
    knn_scores = []
    log_loss_score = []

    # preparing the data for conversion
    a = np.arange(0,len(dataset.columns),1)
    #list out all perms
    perm = list(permutations(a))

    y = convertIndex(dataset.columns, y)

    y = dataset.iloc[:, y].values

    for i in perm:
        X = dataset.iloc[:, [i[0], i[1]]].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        for k in range(1,100):
            knn_classifier = KNeighborsClassifier(n_neighbors = k)
            knn_classifier.fit(X_train, y_train)
            knn_scores.append(knn_classifier.score(X_test, y_test))

        # ind = np.argmax(knn_scores)
        # columnList1.append(i[0])
        # columnList2.append(i[1])
            
        classifier = KNeighborsClassifier(n_neighbors = ind+1, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)

            # Predicting the Test set results
        y_pred = classifier.predict(X_test)
    
        # calculate accuracy
        scoring = (metrics.accuracy_score(y_test, y_pred))
            
        columnList1.append(i[0])
        columnList2.append(i[1])
        scoringList.append(scoring)

    ind = np.argmax(scoringList)

    freq = Counter(scoringList)
    finalColumns = np.where(scoringList == scoringList[0])



    
def convertIndex(data, y):
    if not isinstance(y, int):
        for i in range(len(data)):
            if data.columns[i] == y:
                y = i
                return y
        if not isinstance(y, int):
            print("Failed to generate a index for y")
            exit()
    

if __name__ == "__main__":
    knn("Data/glass.csv", "RI")
