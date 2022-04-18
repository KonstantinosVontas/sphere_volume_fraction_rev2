#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


def grad(u):
    u = u.reshape((3, 3, 3)) # z is most frequent, u[x, y, z]
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




def gen_smooth(N, h=1, rmin=4, rmax=25):
    res = np.empty((N, 28))
    for i in range(N):
        # Center of central cell.
        x0 = np.random.uniform(rmin, rmax)
        y0 = np.random.uniform(rmin, rmax)
        z0 = np.random.uniform(rmin, rmax)
        r0 = (x0**2 + y0**2 + z0**2)**0.5
        k_ref = 2 / r0
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    x = x0 + h * (ix - 1) # Center of neighboring cell.
                    y = y0 + h * (iy - 1)
                    z = z0 + h * (iz - 1)
                    u = -(x**2 + y**2 + z**2)**0.5
                    res[i, ix * 9 + iy * 3 + iz] = u
        res[i, -1] = k_ref
    return res

#path = "../overlap_curvature.csv"
#k_ref_data = np.genfromtxt(path, delimiter=',')

k_ref_data = gen_smooth(2000, rmin=4, rmax=25)

uu = k_ref_data[:, :27]
h = 1.

N = uu.shape[0]
kk_fd = np.zeros(N)
for i in range(N):
    g = grad(uu[i])
    n = normal(g)
    kk_fd[i] = curv(n)

kk_ref = k_ref_data[:, -1]




'''



plt.scatter(kk_ref, kk_fd, s=1)
k = np.linspace(kk_ref.min(),kk_ref.max())
plt.plot(k, k)
plt.xlabel(r'$\kappa_\mathrm{ref}$')
plt.ylabel(r'$\kappa_\mathrm{FD}$')
#plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.savefig('K_FD_overK_ref.pdf')

#exit()
'''
print(kk_fd)


f = open('K_FD_column.csv', 'w')

# create the csv writer
writer = csv.writer(f)
#kk_fd2 = np.transpose(kk_fd)
# write a row to the csv file
writer.writerow(kk_fd)

pd.read_csv('K_FD_column.csv', header=None).T.to_csv('K_FD_final.csv', header=False, index=False)

f.close()






path = "../overlap_curvature_h1.csv"
data = np.genfromtxt(path, delimiter=',')

K_myCode = data[:, 27]
print('The curvature of my code is: \n', K_myCode)

print('\nThe curvature of kk_fd is: \n', kk_fd)

print('\nThe curvature of kk_ref is: \n', kk_ref)



plt.plot(K_myCode, kk_fd, 'o', color='black')
plt.title('$\kappa_{FD}$ over $\kappa_{myCode}$')
plt.xlabel(r'$\kappa_{myCode}$')
plt.ylabel(r'$\kappa_{fd}$')
plt.savefig('curvature of myCode and K_FD.pdf')


