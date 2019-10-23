import numpy as np

# number 1
# a
print("Exclusive\t", np.arange(5,25)) # generates 5 to 24
print("Inclusive\t", np.arange(5,25+1)) # generates 5 to 25
vector = np.arange(5,25) # saved for later
# b
vector =    np.flip(vector) # flips the vector
print(vector)

# c
vector = np.random.rand(30) # generates a 1 demensional 30 value array
# prints out the new vector and the mean of the vector
print(f"""
Vector: {vector} 

Mean : {vector.mean()}  
""")

# 2
import pandas as pd
df = pd.read_csv("titanic.csv") # reads
age = pd.cut(df['age'], [0, 18, 80]) # creates a div based on age
df.pivot_table("survial", ['sex', age]) # pushes them together to generate a piviot table
