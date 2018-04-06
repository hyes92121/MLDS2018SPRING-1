import math
import numpy as np
import matplotlib.pyplot as plt

from numpy import arange


class Coordinates(object):
    def __init__(self, range, step):
        self.start  = range[0]
        self.max    = range[1]
        self.step   = step


    def myFunc(self, x):
        y = (6*x)-2
        exp     = math.tanh(y)
        sinc    = (math.sin(y*math.pi))/(math.pi*y)

        return exp*sinc

    def myFunc2(self, x):
        y = (6*x)-5

        sinc = (math.sin(math.pi*y))/(math.pi*y)
        abso = abs(y)
        f    = (11*math.cos(y)-6*math.cos( (11*y)/6) )

        return sinc * abso * f


    def counter(self):
        c = 0
        while c <= self.max:
            yield c
            c += self.step


    def get_coordinates(self):
        for i in arange(self.start, self.max, self.step):
            yield (round(i, 5), round(self.myFunc2(i), 5))



c = Coordinates((0, 1), 0.00005)

def myFunc(x):
    y   = 6*(x-1)
    f1  = np.vectorize(math.exp)
    f2  = np.vectorize(math.sin)
    exp = f1(y)
    sin = f2(y)

    return (exp*sin)+0.3

X, Y = [], []
for x, y in c.get_coordinates():
    X.append(x)
    Y.append(y)

X, Y = np.array(X), np.array(Y)




x = arange(0.0, 1.0, 0.0005)

plt.plot(X, Y)
plt.ylabel("By Coordinates")
plt.show()
"""
plt.plot(x, myFunc(x))
plt.ylabel("By arange")
plt.show()



with open("data_train2.csv", "w") as f:
    f.write("x,y\n")
    for x, y in c.get_coordinates():
        f.write("{},{}\n".format(x, y))
"""

