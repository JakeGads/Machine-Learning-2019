import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('breast_cancer_dataset.csv') #Prediction
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
colNames = dataset.columns.values.tolist()

scoringList = []
columnList1 = []
columnList2 = []
knn_scores = []
log_loss_score = []

from itertools import permutations 
a = np.arange(0,7,1)
perm = list(permutations(a, 2))

for i in perm: 
    X = dataset.iloc[:, [i[0], i[1]]].values
    y = dataset.iloc[:, 9].values
        
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting K-NN to the Training set
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
    scoring = (metrics.accuracy_score(y_test, y_pred))
        
    columnList1.append(i[0])
    columnList2.append(i[1])
    scoringList.append(scoring)
ind = np.argmax(scoringList)
print (columnList1[ind],columnList2[ind] )


freq = Counter(scoringList)
finalColumns = np.where(scoringList == scoringList[0])
print(finalColumns)


X = dataset.iloc[:, [0,1,2]].values
y = dataset.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
scoring = (metrics.accuracy_score(y_test, y_pred))
print(scoring)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# calculate accuracy
from sklearn import metrics
scoring = (metrics.accuracy_score(y_test, y_pred))
print(scoring)
