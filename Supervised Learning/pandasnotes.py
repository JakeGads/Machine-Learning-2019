# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:07:16 2019

@author: Student
"""
#PANDAS
import pandas as pd

series = pd.Series([1,2,3,4,5], index =['a','b','c','d','c'])
print(series)

print(series[2])

print(series.iloc[2]) #iloc only works with numbers

print(series['b'])

print(series['c'])

print(series.loc['c'])

print(series[2:])

#Pandas Dataframe
#uses df to mean dataframe
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,4),
                  columns= list('ABCD'))

print(df) #prints table with row numbers and column titles

print(df.values) #prints values in a matrix of only the values

print(df.describe()) #shows information about each column like mean and std

print(df.mean(0)) #does for down the column

print(df.mean(1)) #does for across the row

df.head(3) #prints first 3 rows

df.tail(3) #prints last 3 rows

df.head() #prints first half of rows

df.tail() #prints second half of rows

print(df.A) #prints column A
print(df['A']) #both do the same thing

print(df[['A','C']]) #prints columns A and C

print(df[2:5]) #will print rows 2 - 4

print(df.iloc[2:4, 1:4])

print(df.iloc[[2,4],[1,3]])

print(df.loc[1:5, 'A':'C']) #prints rows 1-5 of columns A-C

print(df.at[2,'B']) #prints value at that exact location

#only give cells where both conditions are true
print(df[(df.A > 0.5) & (df.B > 0.4)])

print(df.transpose())
df.T

#sorting data
print(df.sort_index(axis=0, ascending=False)) #sorting by index of rows

print(df.sort_index(axis=1, ascending=False)) #sorting by index of columns

print(df.sort_values('A', axis=0))

print(df.sort_values(5,axis=1))


import math
#can add conditions like the if statment
sq_root = lambda x: math.sqrt(x) if x > 0 else x
sq = lambda x: x ** 2

print(df.B.apply(sq_root))

print(df.B.apply(sq))

for column in df:
    df[column] = df[column].apply(sq_root)
print(df)

print(df.apply(np.sum, axis=0))
print(df.apply(np.sum, axis=1))

data = {'name': ['Nad', 'June', 'Amy'],
        'year': [2012, 2018, 2012],
        'programs': [12, 1, 10]}

df = pd.DataFrame(data, index = ['Singapore', 'Japan', 'USA'])

schools = np.array(['Oxford', 'MIT', 'Oxford'])
df['school'] = schools

print(df)

#removing a row

df1 = df.drop(['Japan'])

#dropping a column
df2 = df.drop('programs', axis=1)

print(df1)
print(df2)

df3 = df.drop(df.columns[[1,3]], axis = 1)
print (df3)

#CROSSTAB
dfTeam = pd.DataFrame({'Gender': ['Male', 'Male', 'Female'],
                       'Team': [1,2,3]})

print(dfTeam)

print(pd.crosstab(dfTeam.Gender, dfTeam.Team)) #how many of each are on a team
print(pd.crosstab(dfTeam.Team, dfTeam.Gender)) #how many of a team to each gen.

