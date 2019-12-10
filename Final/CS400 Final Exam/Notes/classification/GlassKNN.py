# Importing the libraries
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', 
                        category=DataConversionWarning)
warnings.simplefilter(action='ignore', 
                      category=FutureWarning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv(r'E:\CS400 Fall 2019\glass.csv') 
ColumnNames = list(dataset.columns.values.tolist())
NumOfColumns = dataset.shape[1]
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)

scoringList = []
finalColumns = []
numOfNeighbors = []
ListColumns = []

from itertools import permutations 
a = np.arange(0,(NumOfColumns-1),1)

y = dataset.iloc[:, NumOfColumns-1].values

for i in range(2,NumOfColumns-2,1):
    perm = list(permutations(a, i))
    #print(perm)
    scoringList = []
    numOfNeighbors = []
    for j in perm:
        listofCols = list(j)
        X = dataset.iloc[:, listofCols].values
              
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
            
        # Fitting K-NN to the Training set
        knn_scores = []
        for k in range(1,21):
            knn_classifier = KNeighborsClassifier(n_neighbors = k)
            knn_classifier.fit(X_train, y_train)
            knn_scores.append(knn_classifier.score(X_test, y_test))
        
        ind = np.argmax(knn_scores)
           
        classifier = KNeighborsClassifier(n_neighbors = ind+1, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        # calculate accuracy
        scoring = metrics.accuracy_score(y_test, y_pred)
          
        numOfNeighbors.append(ind+1)
        scoringList.append(scoring)
        
    indF = np.argmax(scoringList)
    neighborValue = numOfNeighbors[indF]
    
    finalColumns = []
    for k in range(len(scoringList)): 
        if (scoringList[k] == scoringList[indF]):
            finalColumns.append(k)
    #print(finalColumns)
    
    ListofColumns = []
    
    for l in finalColumns : 
        for m in range(0,i,1):
            ListofColumns.append(perm[l][m])
             
    def Remove(duplicate): 
        final_list = [] 
        for num in duplicate: 
            if num not in final_list: 
                final_list.append(num) 
        return final_list 
    ListofColumns = Remove(ListofColumns)
    
    print('The final columns used for analysis:')
    for n in ListofColumns : 
        print(ColumnNames[n])
        
        #Columns
    X = dataset.iloc[:, ListofColumns].values
    y = dataset.iloc[:, 9].values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting KNN to the Training set
    #numOfNeighbors[indF]+1
    classifier = KNeighborsClassifier(n_neighbors = neighborValue, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # calculate accuracy
    scoring = metrics.accuracy_score(y_test, y_pred)
    print ('Neighbor(s) used: ', neighborValue)
    print ('Accuracy Score: ', round(scoring*100,2), '%')
  