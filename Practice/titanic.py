"""
Apply all classification models to the titanic dataset

Question - Does sex, siblings, fare, embarked and who is traveling decide the class of that travel.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")  # suppresses all warning


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
        return f"X:{self.X}, y:{self.y}, k:{self.k}, score: ~{self.accuracy_score}"

    def writtable(self):
        comb = "["
        try:
            for i in self.X:
                comb += str(i) + " "
        except:
            comb += str(self.X)
        comb += "]"
        return f"{comb},{self.y},{self.k}, {self.accuracy_score:.2f}"


def clean_dataset(file):
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

    return dataset


def knn(file, ys, Xs, max_k=100, supress_text=False):
    # open loads and cleans data
    dataset = clean_dataset(file)

    X = dataset.iloc[:, Xs]
    y = dataset.iloc[:, ys]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # def __init__(self, X, y, k, accuracy_score):
    accuracy = Accuracy(0, 0, 0, 0)

    for k in range(1, max_k):

        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train.values.ravel())
        # didn't like my list had to give it the ravel

        knn_accuracy = knn.score(X, y)
        # handles outfitting
        if knn_accuracy > .97:
            continue

        if knn_accuracy > accuracy.accuracy_score:
            accuracy.remake(Xs, ys, k, knn_accuracy)

    return accuracy.printable()


def logistic_regression(file, ys, Xs):
    dataset = clean_dataset(file)

    X = dataset.iloc[:, Xs]
    y = dataset.iloc[:, ys]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # def __init__(self, X, y, k, accuracy_score):
    accuracy = Accuracy(0, 0, 0, 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    scoring = metrics.accuracy_score(y_test, y_pred)

    accuracy = Accuracy(Xs, ys, 0, scoring)

    return accuracy.printable()


def decision_tree(file, ys, Xs):
    dataset = clean_dataset(file)

    X = dataset.iloc[:, Xs]
    y = dataset.iloc[:, ys]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    classifier = DecisionTreeClassifier(criterion='entropy',
                                        random_state=0)  # Entropy is nothing but the measure of disorder.
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    acc = Accuracy(Xs, ys, 0, metrics.accuracy_score(y_test, y_pred))

    return acc.printable()


def random_forrest(file, ys, Xs):
    dataset = clean_dataset(file)

    X = dataset.iloc[:, Xs]
    y = dataset.iloc[:, ys]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    classifier = RandomForestClassifier(n_estimators=600, max_depth=300, max_features='sqrt')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    acc = Accuracy(Xs, ys, 0, metrics.accuracy_score(y_test, y_pred))

    return acc.printable()


if __name__ == "__main__":
    # survived,pclass,sex,age,sibsp,parch,fare,embarked,classV,who,adult_male,deck,embark_town,alive,alone
    columns = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "classV", "who", "adult_male",
               "deck", "embark_town", "alive", "alone"]
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

    print(f"""
    K Nearest Neighbor:\t\t{knn(file, y, x)}
    Logistic Regression:\t{logistic_regression(file, y, x)}
    Decision Tree:\t\t\t{decision_tree(file, y, x)}
    Random Forest:\t\t\t{random_forrest(file, y, x)}
    """)