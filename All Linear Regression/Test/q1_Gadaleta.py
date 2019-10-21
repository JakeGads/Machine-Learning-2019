import numpy as np

# number 1
# a
print("Exclusive\t", np.arange(5,25))
print("Inclusive\t", np.arange(5,25+1))
vector = np.arange(5,25)
# b
vector =    np.flip(vector)
print(vector)

# c
from statistics import mean

vector = np.random.rand(30)
print(f"""
Vector: {vector}
Mean : {vector.mean()}  
""")

# 2
import pandas as pd
df = pd.read_csv("titanic.csv")
pt = pd.pivot_table(df, index=["sex", "age"])
