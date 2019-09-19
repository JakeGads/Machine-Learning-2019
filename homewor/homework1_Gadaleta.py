# 1) Import NumPy as np and print the version number
import numpy as np
print(np.version.version)

# 2) Create a one-dimensional array of numbers from 0 to 9
a = np.arange(10)
print(a)

# 3) Create a rank 2 (2x2) array containing random numbers.
a = np.random.rand(2, 2)
print(a)

# 4) Extract all event numbers from an array created in question 2.
for array in a:
    for i in array:
        print(i, end=' ')
print()

# 5) Convert the array created in question 2 to a rank 2 array with 2 rows
a = np.ndarray(shape=(2, 2))
print(a)

# 6) Find a function from the NumPy package that helps in finding the common elements between
# two arrays (one-dimensional). Then use the function in an example to show the use of the
# function.
b = np.array([1, 2, 3, 4, 5])
c = np.array([5, 6, 7, 8, 1])

print(np.intersect1d(b, c))

# 7) Use the “where” function from the NumPy package to find the positions (index) where elements
# of two arrays are equal.

array1 = np.array([2, 3, 4, 6, 7])
array2 = np.array([2, 4, 3, 6, 7])

print(np.where(array1 == array2))

# 8) Create a range of numbers from 0-15 named array1 (15 included). Create another array named
# array2 which extracts numbers between 4 and 6 from array1 and store them in array2. Print array2.
array1 = np.arange(16)
array2 = array1[4:6]
# so I pull elements from 4:6 which is 4,5 4:7 whould also pull 6 if that is what you are looking for
print(array2)

# 9) Create a rank 2 (3x3) array containing random numbers. Swap columns 1 and 2 in the twodimensional array and
# print the results.
a = np.random.random(9).reshape(3, 3)
print('\n', a, '\n')
a[[0, 1]] = a[[1, 0]]
print(a)

# 10) Using the array in question 9, display only column 1 and 2 and row 1 of the array as output.
print('\n', a[:1, :1])

# 11) Swap rows 1 and 2 in the two-dimensional array created in question 9 and print the results.
a[[0, 1]] = a[[1, 0]]
print('\n', a)

# 12) Explore the set_printoptions function in the NumPy package. There are multiple options available
# with this function such as precision and threshold. Create a one-dimensional array and show the
# use of precision and threshold.
#
# Q. From the array a, replace all values greater than 30 to 30 and less than 10 to 10.
a = np.array([0, 5, 7, 30, 55, 6, 16, 18])
a = a
print('\n\n', a)

# 13) This question will require the sorting functions available in the NumPy package. Given two arrays
# persons = np.array([‘Mary’, ‘Peter’, ‘Joe’])
# ages = np.array([34, 12, 23])
# Sort the persons array based on alphabetical order. Based on the sort of the persons array, respectively
# rearrange the ages array elements that match the persons with the ages.