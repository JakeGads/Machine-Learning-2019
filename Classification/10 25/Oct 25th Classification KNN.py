# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:59:10 2019

@author: guptap
"""

# Logistic Regression

from itertools import permutations

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'E:\CS400 Fall 2019\breast_cancer_dataset.csv') #Prediction
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
colNames = dataset.columns.values.tolist()

scoringList = []
columnList1 = []
columnList2 = []
knn_scores = []

a = np.arange(0,7,1)
perm = list(permutations(a, 2))

for i in perm: 
    X = dataset.iloc[:, [i[0], i[1]]].values
    y = dataset.iloc[:, 9].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    
    for k in range(1,21):
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
        knn_classifier.fit(X_train, y_train)
        knn_scores.append(knn_classifier.score(X_test, y_test))
    
    ind = np.argmax(knn_scores)
    columnList1.append(i[0])
    columnList2.append(i[1])
        
    classifier = KNeighborsClassifier(n_neighbors = ind+1, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # calculate accuracy
    scoring = metrics.accuracy_score(y_test,y_predict)
       
    columnList1.append(i[0])
    columnList2.append(i[1])
    scoringList.append(scoring)
    
ind = np.argmax(scoringList)
print (columnList1[ind],columnList2[ind] )

freq = Counter(scoringList)
finalColums = np.where(scoringList)
print(finalColums)
