from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math


#Generating Test Data
classA = [(random.normalvariate(-0.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + [(random.normalvariate(1, 1), random.normalvariate(0, 1), 1.0) for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

data = classA + classB
random.shuffle(data)




pylab.hold(True)
pylab.plot([ p[0] for p in classA ], [ p[1] for p in classA ], 'bo')
pylab.plot([ p[0] for p in classB ], [ p[1] for p in classB ], 'ro')



# matrix P
w, h = 10, 10;
P = [[0 for x in range(w)] for y in range(h)]
for i in range(10):
    for j in range(10):
        P[i][j]= (numpy.inner(data[i][0:2], data[j][0:2])+ 1 )**5*data[i][2]*data[j][2]

# vector q

q = numpy.array([-1 for x in range(10)])
q = q.astype(numpy.double)


# vector h

h = numpy.array([0 for x in range(10)])
h = h.astype(numpy.double)

# matrix G
G = numpy.diag([-1 for x in range(10)])
G = G.astype(numpy.double)

# call qp

r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])




# indicator function
def indicator(x, y):
    alpin = []
    xi = []
    for i in range(10):
        if alpha[i] > pow(10,-5):
            alpin.append(alpha[i])
            xi.append(data[i])
    sum = 0
    for i in range(len(xi)):
        sum += alpin[i]*xi[i][2]*(x * xi[i][0]+ y * xi[i][1] + 1)**5
    return sum


#plotting the Decision Boundary

xrange = numpy.arange(-4 , 4 , 0.05)
yrange = numpy.arange(-4 , 4 , 0.05)

grid = matrix([[indicator(x,y) for y in yrange] for x in xrange])

pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))
pylab.show()
pylab.show()
