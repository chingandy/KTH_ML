from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math


#Generating Test Data
classA = [(random.normalvariate(-2, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]

classB = [(random.normalvariate(0, 0.5), random.normalvariate(-1, 0.5), -1.0) for i in range(10)]

data = classA + classB
random.shuffle(data)




pylab.hold(True)
pylab.plot([ p[0] for p in classA ], [ p[1] for p in classA ], 'bo')
pylab.plot([ p[0] for p in classB ], [ p[1] for p in classB ], 'ro')



# matrix P
w, h = 20, 20;
P = [[0 for x in range(w)] for y in range(h)]
for i in range(20):
    for j in range(20):
        P[i][j]= (numpy.inner(data[i][0:2], data[j][0:2])+ 1 )*data[i][2]*data[j][2]

# vector q

q = numpy.array([-1 for x in range(20)])
q = q.astype(numpy.double)


# vector h
c = 10000
h1 = numpy.array([0 for x in range(20)])
h2 = numpy.array([c for x in range(20)])
h = numpy.append(h1, h2)
h = h.astype(numpy.double)

# matrix G
G1 = numpy.diag([-1 for x in range(20)])
G2 = numpy.diag([1 for x in range(20)])
G = numpy.r_[G1, G2]
G = G.astype(numpy.double)

# call qp

r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])




# indicator function
def indicator(x, y):
    alpin = []
    xi = []
    for i in range(20):
        if alpha[i] > pow(10,-5):
            alpin.append(alpha[i])
            xi.append(data[i])
    sum = 0
    for i in range(len(xi)):
        sum += alpin[i]*xi[i][2]*(x * xi[i][0]+ y * xi[i][1] + 1)
    return sum


#plotting the Decision Boundary

xrange = numpy.arange(-4 , 4 , 0.05)
yrange = numpy.arange(-4 , 4 , 0.05)

grid = matrix([[indicator(x,y) for y in yrange] for x in xrange])

pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))
pylab.show()
pylab.show()
