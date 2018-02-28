import numpy as np
import math as m

sig= np.array([2,4,5,6])

v1= sig*sig.reshape(-1,1)
v2= np.power(sig,2)

print v1,v2
