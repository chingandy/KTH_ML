import numpy as np

x = np.array([1,1,1,2,2,3,4,5,6])
print x
idx = np.where(x == 1)[0]
print idx
