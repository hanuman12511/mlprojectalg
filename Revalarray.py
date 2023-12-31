import numpy as np
 
array = np.arange(15).reshape(3, 5)
print("Original array : \n", array)

print("\nravel() : ", array.ravel())
 
print("\nnumpy.ravel() == numpy.reshape(-1)")
print("Reshaping array : ", array.reshape(-1))