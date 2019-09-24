import pandas as pd
import numpy as np
import re
print('1. Import pandas and print the version')
print(pd.show_versions())

print('\n2. Create a pandas series from a list of numbers (created using the numpy package).')
a = pd.Series(np.random.randn(10), index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print(a)

print('\n3. Convert the series created in the last question into a dataframe. \
This will require researching functions for series from the pandas package.')
a = pd.DataFrame(a)
print(a, '\n\n', type(a))

print('\n4. Create another series and combine this series with the one created in the previous question to form a \
dataframe. Series 1 will represent column 1 and Series 2 will represent column 2.')
a[1] = pd.Series(np.random.randn(10), index=a.index)

print(a)

print('\n5. Create a regular expression to find phone numbers')
phones = pd.Series(['12345', '(610) 786-9089', '610 786-9089', '(610) 786 9089'])
# \d{3}-\d{3}-\d{4}
# r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'
regex = re.compile(r'\(\d{3}\) \d{3}-\d{4}')
for i in phones:
    print(regex.findall(i), end='\t')
print()
print('\n6. Compute the euclidean distance between series p and q, without using a python packaged formula.') # return
p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
print(p-q)# is this what you want

print('\n7. Import the dataset (Housing.csv available in Blackboard) into a dataframe df. Make sure the column names are \
used in the dataframe.')

df = pd.read_csv('Housing.csv')
print(df.head())

print('\n8. Check is df has any missing values.')
print('isnull().values.any(): ', df.isnull().values.any())

print('\n9. Calculate the frequency of distinct values in df.')

for col in df.columns:
    print(df.groupby(col).size())

