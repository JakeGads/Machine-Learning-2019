stringValue = "hello"
floatValue = 10.0
integerValue = 20

print('{var}\t{type}\t{val}'.format(var = 'string value', type = type(stringValue), val = stringValue))
print('{var}\t{type}\t{val}'.format(var = 'float value', type = type(floatValue), val = floatValue))
print('{var}\t{type}\t{val}'.format(var = 'integer value', type = type(integerValue), val = integerValue))
