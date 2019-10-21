import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("titanic.csv")

g = sns.catplot(x = "who", y="survived", col="class", data=data, kind="bar", ci=None, aspect=1.0)