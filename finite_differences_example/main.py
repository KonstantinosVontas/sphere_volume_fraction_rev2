#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

path = "../overlap_curvature.csv"

data = np.genfromtxt(path, delimiter=',')

uu = data[:, :27]
h = 1.

def grad(u):
    u = u.reshape((3, 3, 3))
    u = np.transpose(u, (2, 1, 0))
    g = np.zeros((3, 2, 3)) # (face direction, face side, component of gradient)
    '''
    -------------
   2|   |   |   |
    -------------
   1|   |   |   |
    -------------
   0|   |   |   |
    -------------
      0   1   2
    '''
    # x-component of gradient
    g[0, 0, 0] = (u[1, 1, 1] - u[0, 1, 1]) / h
    g[0, 1, 0] = (u[2, 1, 1] - u[1, 1, 1]) / h
    g[1, 0, 0] = ((u[2, 1, 1] + u[2, 0, 1]) * 0.5 -
                  (u[0, 1, 1] + u[0, 0, 1]) * 0.5) / (2 * h)
    g[1, 1, 0] = ((u[2, 2, 1] + u[2, 1, 1]) * 0.5 -
                  (u[0, 2, 1] + u[0, 1, 1]) * 0.5) / (2 * h)
    g[2, 0, 0] = ((u[2, 1, 1] + u[2, 1, 0]) * 0.5 -
                  (u[0, 1, 1] + u[0, 1, 0]) * 0.5) / (2 * h)
    g[2, 1, 0] = ((u[2, 1, 2] + u[2, 1, 1]) * 0.5 -
                  (u[0, 1, 2] + u[0, 1, 1]) * 0.5) / (2 * h)
    # y-component of gradient
    g[1, 0, 1] = ( u[1, 1, 1] - u[1, 0, 1]) / h
    g[1, 1, 1] = ( u[1, 2, 1] - u[1, 1, 1]) / h
    g[0, 0, 1] = ((u[1, 2, 1] + u[0, 2, 1]) * 0.5 -
                  (u[1, 0, 1] + u[0, 0, 1]) * 0.5) / (2 * h)
    g[0, 1, 1] = ((u[2, 2, 1] + u[1, 2, 1]) * 0.5 -
                  (u[2, 0, 1] + u[1, 0, 1]) * 0.5) / (2 * h)
    g[2, 0, 1] = ((u[1, 2, 1] + u[1, 2, 0]) * 0.5 -
                  (u[1, 0, 1] + u[1, 0, 0]) * 0.5) / (2 * h)
    g[2, 1, 1] = ((u[1, 2, 2] + u[1, 2, 1]) * 0.5 -
                  (u[1, 0, 2] + u[1, 0, 1]) * 0.5) / (2 * h)
    # x-component of gradient
    g[2, 0, 2] = ( u[1, 1, 1] - u[1, 1, 0]) / h
    g[2, 1, 2] = ( u[1, 1, 2] - u[1, 1, 1]) / h
    g[0, 0, 2] = ((u[1, 1, 2] + u[0, 1, 2]) * 0.5 -
                  (u[1, 1, 0] + u[0, 1, 0]) * 0.5) / (2 * h)
    g[0, 1, 2] = ((u[2, 1, 2] + u[1, 1, 2]) * 0.5 -
                  (u[2, 1, 0] + u[1, 1, 0]) * 0.5) / (2 * h)
    g[1, 0, 2] = ((u[1, 1, 2] + u[1, 0, 2]) * 0.5 -
                  (u[1, 1, 0] + u[1, 0, 0]) * 0.5) / (2 * h)
    g[1, 1, 2] = ((u[1, 2, 2] + u[1, 1, 2]) * 0.5 -
                  (u[1, 2, 0] + u[1, 1, 0]) * 0.5) / (2 * h)
    return g

def normal(g):
    n = np.zeros_like(g)
    for i in range(3):
        for j in range(2):
            l = sum(g[i, j, :]**2)**0.5
            if l:
                n[i, j, :] = -g[i, j, :] / l
    return n

def curv(n):
    k = (n[0, 1, 0] - n[0, 0, 0]) / h + \
        (n[1, 1, 1] - n[1, 0, 1]) / h + \
        (n[2, 1, 2] - n[2, 0, 2]) / h
    return k

N = uu.shape[0]
kk_fd = np.zeros(N)
for i in range(N):
    g = grad(uu[i])
    n = normal(g)
    kk_fd[i] = curv(n)

kk_exact = data[:, -1]

plt.scatter(kk_exact, kk_fd)
plt.xlabel(r'$\kappa_\mathrm{exact}$')
plt.ylabel(r'$\kappa_\mathrm{FD}$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('a.pdf')
