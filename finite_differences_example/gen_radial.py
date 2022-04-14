#!/usr/bin/env python3

import numpy as np
import csv
import pandas as pd
import random

# Generates a random number between
# a given positive range
from matplotlib import pyplot as plt

r1 = random.uniform(10, 15)

x0, y0, z0 = r1, 0., 0. # RANDOM such that `r0` is large enough (>10)
path = "K_FD_column.csv"
K_FD_column_data = np.genfromtxt(path, delimiter=',')
K_FD_values = len(K_FD_column_data)

K_0func = []
for i in range(K_FD_values):
    r1 = random.uniform(10, 15)
    x0, y0, z0 = r1, 0., 0.
    r0 = (r1 ** 2 + y0 ** 2 + z0 ** 2) ** 0.5
    print(r0)
    k0 = 2 / r0
    K_0func.append(k0)

print(K_0func)

df = pd.DataFrame({'data':K_FD_column_data, 'kappa':K_0func})
df.to_csv(r'2Kappa.csv', index=False, header=False)



plt.scatter(K_0func, K_FD_column_data)
plt.xlabel(r'$\kappa_\mathrm{0}$')
plt.ylabel(r'$\kappa_\mathrm{FD}$')
plt.savefig('K_FD_over_K_0.pdf')



'''


#---------------
x0, y0, z0 = r1, 0., 0. # RANDOM such that `r0` is large enough (>10)
h = 1.
u = np.zeros((3, 3, 3))
for iz in [0, 1, 2]:
    for iy in [0, 1, 2]:
        for ix in [0, 1, 2]:
            x = x0 + h * (ix - 1)
            y = y0 + h * (iy - 1)
            z = z0 + h * (iz - 1)
            u[ix, iy, iz] = (x ** 2 + y ** 2 + z ** 2) ** 0.5

u = np.transpose(u, (2, 1, 0))
u = u.flatten()
r0 = (x0 ** 2 + y0 ** 2 + z0 ** 2) ** 0.5
k0 = 2 / r0

print(u)
print(k0)
#-----------

'''
