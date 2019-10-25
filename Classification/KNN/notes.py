import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
# fotting knn to the trainging set
from sklearn.neighbors import KNeighborsClassifier
#scaling
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("")
dataset = dataset.dropna()
dataset = dataset.reset_index(drop = True)

X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 9].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

knn_scores = []

for k in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    knn_scores.append(knn_classifier.score(X_test, y_test))


plt.plot([k for k in range(1,21)], knn_scores, color = 'red')

classifier = KNeighborsClassifier(n_neighbors = 2, metric = "minkowski", p = 2)
classifier.fit(X_train, y_train)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


scoring = (metrics.accuracy_score(y_test, y_pred))
print(scoring)
