print('# 1) Import NumPy as np and print the version number')
import numpy as np
print(f'NumPy Version: {np.version.version}')

print('\n\n# 2) Create a one-dimensional array of numbers from 0 to 9')
a = np.arange(10)
print(a)


print('\n\n# 3) Create a rank 2 (2x2) array containing random numbers.')
a = np.random.rand(2, 2)
print(a)

print('\n\n# 4) Extract all event numbers from an array created in question 2.')
for array in a:
    for i in array:
        print(i, end=' ')
print()

print('\n\n# 5) Convert the array created in question 2 to a rank 2 array with 2 rows')
a = np.ndarray(shape=(2, 2))
print(a)

print('''\n\n
# 6) Find a function from the NumPy package that helps in finding the common elements between
# two arrays (one-dimensional). Then use the function in an example to show the use of the
# function. ''')
b = np.array([1, 2, 3, 4, 5])
c = np.array([5, 6, 7, 8, 1])

print(f'NumPy.Intersect1d: {np.intersect1d(b, c)}')
print('''\n\n
# 7) Use the “where” function from the NumPy package to find the positions (index) where elements
# of two arrays are equal. ''')

array1 = np.array([2, 3, 4, 6, 7])
array2 = np.array([2, 4, 3, 6, 7])

print('NumPy.where(array1 == array2): ', np.where(array1 == array2))


print('''\n\n
# 9) Create a rank 2 (3x3) array containing random numbers. Swap columns 1 and 2 in the two dimensional array and
# print the results. ''')
a = np.random.random(9).reshape(3, 3)
print('\na:\n', a,)
a[[0, 1]] = a[[1, 0]]
print('a (swapped):\n', a)

print('\n\n# 10) Using the array in question 9, display only column 1 and 2 and row 1 of the array as output.')
print('\n', a[:1, :2])

print('\n\n# 11) Swap rows 1 and 2 in the two-dimensional array created in question 9 and print the results.')
a[[0, 1]] = a[[1, 0]]
print('\n', a)

print('''\n\n
# 12) Explore the set_printoptions function in the NumPy package. There are multiple options available
# with this function such as precision and threshold. Create a one-dimensional array and show the
# use of precision and threshold.
#
# Q. From the array a, replace all values greater than 30 to 30 and less than 10 to 10. ''')
a = np.array([0, 5, 7, 30, 55, 6, 16, 18])
print('\n\n', a)

myLamda = lambda x: 10 if x < 10 else (30 if x > 30 else x)
# according to PEP 8 this is to long to be a lamda and instead should be a localized function

vectorizer = np.vectorize(myLamda)

a = vectorizer(a)

print(a)

print('''\n\n
# 13) This question will require the sorting functions available in the NumPy package. Given two arrays
# persons = np.array([‘Mary’, ‘Peter’, ‘Joe’])
# ages = np.array([34, 12, 23])
# Sort the persons array based on alphabetical order. Based on the sort of the persons array, respectively
# rearrange the ages array elements that match the persons with the ages.
''')
print('\n')

persons = np.array(['Mary', 'Peter', 'Joe'])
ages = np.array([34, 12, 23])


total = np.array([persons, ages])
total_sort = total[:,total[0].argsort()]

print(total_sort)
