import numpy as np

A = np.array([[1], [2], [3]])
print(A.shape)
A = A[:, :,np.newaxis]
print(A.shape)
