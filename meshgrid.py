import numpy as np

 
x = np.linspace(-4, 4, 9)

y = np.linspace(-5, 5, 11)
print(x)
print(y)
x_1, y_1 = np.meshgrid(x, y)
 
print("x_1 = ")
print(x_1)
print("y_1 = ")
print(y_1)