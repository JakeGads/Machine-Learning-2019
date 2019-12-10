import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier

import data_help as dh


def knn(data, x_s: list, y_name: str, max_k: int, out_file: str):
    highest_score = dh.ClassificationScores.KNNScore(0, 0, 0, 0)
    file = open(out_file, 'a+')
    for x_name in x_s:
        x = data.loc[:, list(x_name)]
        y = data.loc[:, list(y_name)]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        sub_accuracy = dh.KNNScore(0, 0, 0, 0)

        for k in range(max_k + 1):
            knn_classifier = KNeighborsClassifier(n_neighbors=k)

            knn_classifier.fit(X_train, y_train)

            y_pred = knn_classifier.predict(X_test)

            current_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

            if current_accuracy > sub_accuracy.score:  # checks to see if the new accuarcy is bigger than the largest and reassings if it is
                sub_accuracy.__remake__(x_name, y_name, k, current_accuracy)

        file.write(sub_accuracy.__str__() + "\n")
        if sub_accuracy.score > highest_score.score:
            highest_score.__remake__(
                sub_accuracy.x,
                sub_accuracy.y,
                sub_accuracy.neighbors,
                sub_accuracy.score
            )

    file.close()
    return highest_score.__str__()


def logistic_regression(data, x_s: list, y_name: str, out_file: str):
    file = open(out_file, 'a+')
    high_score = dh.ClassificationScores.GeneralClassificationScores(0, 0, 0)

    for x_name in x_s:
        x = data.loc[:, list(x_name)]
        y = data.loc[:, list(y_name)]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        sub_score = dh.ClassificationScores.GeneralClassificationScores(
            x_name, y_name, metrics.accuracy_score(y_test, y_pred)
        )

        file.write(sub_score.__str__())
        if sub_score.score > high_score.score:
            high_score.__remake__(
                sub_score.x,
                sub_score.y,
                sub_score.score
            )

    file.close()
    return high_score.__str__()


def decision_tree(data, x_s: list, y_name: str, out_file: str):
    file = open(out_file, 'a+')
    high_score = dh.ClassificationScores.GeneralClassificationScores(0, 0, 0)

    for x_name in x_s:
        x = data.loc[:, list(x_name)]
        y = data.loc[:, list(y_name)]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

        classifier = DecisionTreeClassifier(criterion='entropy',
                                            random_state=0)  # Entropy is nothing but the measure of disorder.
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        sub_score = dh.ClassificationScores.GeneralClassificationScores(
            x_name, y_name, metrics.accuracy_score(y_test, y_pred))

        file.write(sub_score.__str__())

        if sub_score.score > high_score.score:
            high_score.__remake__(
                sub_score.x,
                sub_score.y,
                sub_score.score
            )

    file.close()
    return high_score.__str__()


def random_forest(data, x_s: list, y_name: str, out_file: str):
    file = open(out_file, 'a+')
    high_score = dh.ClassificationScores.GeneralClassificationScores(0, 0, 0)

    for x_name in x_s:
        x = data.loc[:, list(x_name)]
        y = data.loc[:, list(y_name)]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

        classifier = RandomForestClassifier(n_estimators=600, max_depth=300, max_features='sqrt')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        sub_score = dh.ClassificationScores.GeneralClassificationScores(
            x_name, y_name, metrics.accuracy_score(y_test, y_pred))

        file.write(sub_score.__str__())

        if sub_score.score > high_score.score:
            high_score.__remake__(
                sub_score.x,
                sub_score.y,
                sub_score.score
            )

    file.close()
    return high_score.__str__()