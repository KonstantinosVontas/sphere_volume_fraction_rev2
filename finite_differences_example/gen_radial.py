#!/usr/bin/env python3

import numpy as np

x0, y0, z0 = 10., 0., 0. # RANDOM such that `r0` is large enough (>10)

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
