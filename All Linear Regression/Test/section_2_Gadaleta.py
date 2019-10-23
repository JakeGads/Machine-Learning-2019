import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt

# 1
df = pd.read_csv("iris.csv") # reading

plt.pie(
    df["species"].value_counts(), # counts the number of apperences of each unique values in a col 
    labels=df["species"].unique() # pulls each individual aspect (becuase of the natural of each being 1/3rd I don't really care where it goes)
)
plt.savefig("iris_species_gadaleta.png") # displays

# 2
df = pd.read_csv("flights.csv")  # reading
df["combo"] = df["month"] + " " + str(df["year"]) # this produces a combo  of month and year
plt.plot(df["year"], df["passengers"]) # plot regardless of the month
plt.savefig("flights_by_year_gadaleta.png")
