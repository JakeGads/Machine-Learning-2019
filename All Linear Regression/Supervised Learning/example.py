import numpy as np

# creating a list
list1 = [1,2,3,4,5,6,7,8,9,10]

list2 = [1, True, "hi"]


a1 = np.arange(10)
print(a1)

print(a1.shape)

a1 = np.arange(0,10,2)

twoDimenArray = np.zeros((2,3))

print(twoDimenArray)

fillArray = np.full((2,3), 8)
print(fillArray)
