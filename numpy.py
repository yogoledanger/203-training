import numpy as np

'''
N-dimensional array
'''
# Array creation

a = np.array([1,2,3,4])

b = np.array([(1.5,2,3), (4,5,6)])
b.dtype
np.zeros((3,4))

np.ones((2,3,4))

np.arange( 10, 30, 5 )
np.arange( 0, 2, 0.3 )                 # it accepts float arguments

from numpy import pi
x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points

c = np.arange(24).reshape(2,3,4)         # 3d array
