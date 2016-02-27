from math import *
import random
from numpy import *
import matplotlib.pyplot as plt

import KNNDigits

waitForEnter=False

def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]

def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]


def computeDistances(data, dimensions):
    D = dimensions
    N = len(data)
    dist = []
    for n in range(N):
        for m in range(n):
            dist.append(KNNDigits.exampleDistance(data[n][0], data[m][0]) / sqrt(D))
    return dist

Dims = [784]   # dimensionalities to try
Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF']
Bins = arange(0, 1, 0.02)

plt.xlabel('distance / sqrt(dimensionality)')
plt.ylabel('# of pairs of points at that distance')
plt.title('dimensionality versus digits data point distances')

for i, d in enumerate(Dims):
    data = KNNDigits.loadDigitData('data/1vs2.all', 100000000)
    distances = computeDistances(data, d)

    print "D=%d, average distance=%g" % (d, mean(distances) * sqrt(d))
    plt.hist(distances,
             Bins,
             histtype='step',
             color=Cols[i])
    if waitForEnter:
        plt.legend(['%d dims' % d for d in Dims])
        plt.show(False)
        x = raw_input('Press enter to continue...')

plt.legend(['%d dims' % d for d in Dims])
plt.savefig('fig.pdf')
plt.show()