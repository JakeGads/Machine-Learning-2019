numbers = []
greeting = []

numbers.append(1)
numbers.append(2)
numbers.append(3)

greeting.append('hello')
greeting.append('world')


print('The Length of the numbers list is {length}'.format(length = len(numbers)))

'''
Given the list
data = ("John", "Doe", 53.44) # This is not list notation
Write a format string which prints out the data using the following syntax: 
'''

data = ("John", "Doe", 53.44)
print('Hello {First} {Last}, Your Current Balance is ${cash}'.format(First = data[0], Last = data[1], cash = round(data[2], 2)))
print('Hello %s %s, Your Current Balance is $%2f'%(data[0], data[1], data[2]))
