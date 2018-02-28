import numpy as np

A = np.array([[1,2,3],[2,3,5],[4,6,8]])
print 'A = '
print A
x , y = A.shape
print A.shape[0],A.shape[1]

s = np.zeros(A.shape[1])
print s

for i in range(0,A.shape[0]):
    s = s + A[i,:]
print s
print x
print y
