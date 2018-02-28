import numpy as np
X = np.array([[1,0, 0],[1 ,2 ,3] , [2,3,4], [5,6,7],[1,1,1]])
y = np.array([1,1,2,3,3])
#X,y = getData()
classes = np.unique(y)
Npts,Ndims = np.shape(X)
Nclasses = np.size(classes)
mu = np.zeros((Nclasses,Ndims))
sigma = np.zeros((Nclasses, Ndims, Ndims))
print X
for jdx,class1 in enumerate(classes):
    print 'class = ',class1
    idx = np.where(y==class1)[0]
    print 'idx_2=',idx
    xlc =X[idx,:]
    print xlc

    # mu matrix: C x d matrix of class means
    s = np.zeros(X.shape[1])
    for i in range(0,xlc.shape[0]):
        s = s + xlc[i,:]

    print 'mu = ',s/xlc.shape[0]
    mu[jdx,:]=s/xlc.shape[0]


    # sigma matrix: C x d x d matrix
    print 'xlc - mu = '
    print (xlc - mu[jdx,:])
    sig = (xlc - mu[jdx,:])
    sum = np.zeros((Ndims,Ndims))
    for i in range(np.shape(sig)[0]):
        sum += sig[i]*sig[i].reshape(-1,1)
    print 'sum ='
    print sum
    print 'sum/Nk ='
    print sum/np.shape(sig)[0]
    sigma[jdx,:] = sum/np.shape(sig)[0]

print 'mu='
print mu
print 'sigma = '
print sigma
print 'end'
