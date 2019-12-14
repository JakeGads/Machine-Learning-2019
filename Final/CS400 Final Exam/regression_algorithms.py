from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

import data_help as dh


def linear(data: DataFrame, x_s: list, y_name: str, out_file: str):
    high_score = dh.RegressionScores.GeneralRegressionScores(-1, -1, 0)

    file = open(out_file, 'a+')

    for i in tqdm(range(len(x_s))):
        x_name = x_s[i]

        if y_name in list(x_name):
            continue

        x = data.loc[:, list(x_name)].values
        y = data.loc[:, [y_name]].values

        model = LinearRegression()
        model.fit(x, y)

        score = dh.RegressionScores.GeneralRegressionScores(x_name, y_name, model.score(x, y))

        file.write(score.__str__() + "\n")

        if high_score.score < score.score:
            high_score.__remake__(score.x, score.y, score.score)

    file.close()
    return high_score.__str__()


def polynomial(data: DataFrame, x_s: list, y_name: str, polys: int, out_file: str):
    high_score = dh.RegressionScores.PolyRegressionScores(-1, -1, -1, 0)
    file = open(out_file, "a+")

    for i in tqdm(range(len(x_s))):
        x_name = x_s[i]
        x = data.loc[:, list(x_name)].values
        y = data.loc[:, [y_name]].values

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=.2, random_state=0
        )

        for h in range(polys):
            poly_reg = PolynomialFeatures(degree=h)

            try:
                X_ = poly_reg.fit_transform(x)
                X_test = poly_reg.fit_transform(X_test)

                lin_reg = LinearRegression()
                lin_reg.fit(X_, y)

                _score = dh.RegressionScores.PolyRegressionScores(x_name,  y_name, i, lin_reg.score(X_, y))

                if _score.score > .95:
                    continue

                file.write(_score.__str__() + "\n")

                if _score.score > high_score.score:
                    high_score.__remake__(_score.x, _score.y, _score.poly, _score.score)
            finally:
                None

        file.close()
        return high_score.__str__()


