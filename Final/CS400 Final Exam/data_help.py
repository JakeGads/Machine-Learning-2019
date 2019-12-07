import pandas as pd
import numpy as np

from itertools import permutations


class KNNScore:
    def __init__(self, x, y, neighbors, score):
        if isinstance(x, tuple):
            self.x = str(x).replace(",", "")
        else:
            self.x = x
        self.y = y
        self.neighbors = neighbors
        self.score = score

    def __remake__(self, x, y, neighbors, score):
        self.x = x
        self.y = y
        self.neighbors = neighbors
        self.score = score

    def __str__(self):
        return f"{self.x}, {self.y}, {self.neighbors}, {self.score}"


def clean_data(file: str):
    data = pd.read_csv(file)
    data = data.dropna()
    data = data.reset_index(drop=True)

    for col in data.columns:
        if data[col].dtype == np.float64:
            continue
        if data[col].dtype == np.int64:
            continue

        count = 0
        temp = {}
        for unique in data[col].unique():
            temp.update({unique: count})
            count += 1
        data[col] = [temp[item] for item in data[col]]

    return data


def gen_permutations(data):
    x_s = []
    for length in range(len(data) - 1):
        x_s += permutations(data.columns, length + 1)
    return x_s
