import numpy as np
import math as m

n = np.zeros((5,3))
for i in range(5):
    for j in range(3):
        n[i][j] = i + j
