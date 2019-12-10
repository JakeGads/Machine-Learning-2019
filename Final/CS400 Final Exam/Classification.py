import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import data_help as dh


def knn(data, x_s: list, y_name: str, max_k: int, out_file: str):
    highest_score = dh.KNNScore(0, 0, 0, 0)
    file = open(out_file, 'a+')
    for x_name in x_s:
        x = data.loc[:, list(x_name)]
        y = data.loc[:, y_name]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        sub_accuracy = dh.KNNScore(0, 0, 0, 0)

        for k in range(max_k + 1):
            knn_classifier = KNeighborsClassifier(n_neighbors=k)

            knn_classifier.fit(X_train, y_train)

            y_pred = knn_classifier.predict(X_test)

            current_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

            if current_accuracy > sub_accuracy.accuracy_score:  # checks to see if the new accuarcy is bigger than the largest and reassings if it is
                sub_accuracy.__remake__(x_name, y_name, k, current_accuracy)

        file.write(sub_accuracy.__str__() + "\n")
        if sub_accuracy.score > highest_score.score:
            highest_score.__remake__(
                sub_accuracy.x,
                sub_accuracy.y,
                sub_accuracy.neighbors,
                sub_accuracy.score
            )

    return highest_score.__str__()
