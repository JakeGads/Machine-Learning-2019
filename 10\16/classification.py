import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# dependent how many distinct values

df = pd.read_csv("breast_cancer_dataset.csv")
df = df.dropna()
df = df.reset_index(drop = True)

columList1 = columList2 = scoringList = []

from itertools import permutations
a = np.arange(0, 8)
perm = permutations(a, 2)

y = df.iloc[:, 9].values

for i in list(perm):
    X = df.iloc[:, [i[0], i[1]]].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    from sklearn import metrics
    scoring = metrics.accuracy_score(y_test, y_pred)
    print(scoring)

    columList1.append(i[0])
    columList2.append(1[1])
    scoringList.append(scoring)

ind = np.argmax(scoring)
print(columList1[ind], columList2[ind])

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                stop = X_set[:, 0].max() + 1, step = 0.01),
                    
np.arange(start = X_set[:, 1].min() - 1,
          stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha = 0.75, cmap = ListedColormap(('orange', 'blue')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i),
                label = j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('1')
plt.ylabel('2')
plt.legend()