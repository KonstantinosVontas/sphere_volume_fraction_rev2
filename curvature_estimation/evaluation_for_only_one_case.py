from asyncore import write
from cgi import test
from random import random
from matplotlib import projections
import overlap
import numpy as np
import csv
from tqdm import tqdm
import plotly.graph_objects as go

def divergence(f,sp):
    """ Computes divergence of vector field 
    f: array -> vector field components [Fx,Fy,Fz,...]
    sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], sp[i], axis=i) for i in range(num_dims)])


def random_normal_generation():
    v = np.random.normal(size = (1,3))
    v = v / np.sqrt(np.sum(v ** 2))

    return v

def generate_random_point_in_center_cell():
    maximum_coordinate_value = 1

    points = (np.random.rand(1, 3) * (2 * maximum_coordinate_value)) - maximum_coordinate_value

    return points

def generate_random_sphere():
    normal = random_normal_generation()
    radius = np.random.uniform(1, 10)

    point = generate_random_point_in_center_cell()

    center = point - radius * normal

    return overlap.Sphere(tuple(center[0]), radius)

def generate_cube_from_center(point):

    h = 2

    cube_vertices = np.array((
        (point[0] - h/2, point[1] - h/2, point[2] - h/2),
        (point[0] + h/2, point[1] - h/2, point[2] - h/2),
        (point[0] + h/2, point[1] + h/2, point[2] - h/2),
        (point[0] - h/2, point[1] + h/2, point[2] - h/2),
        (point[0] - h/2, point[1] - h/2, point[2] + h/2),
        (point[0] + h/2, point[1] - h/2, point[2] + h/2),
        (point[0] + h/2, point[1] + h/2, point[2] + h/2),
        (point[0] - h/2, point[1] + h/2, point[2] + h/2),
    ))
    return cube_vertices

def points_of_regular_grid_generation():
    axis_values = [-2, 0, 2]

    points = np.array([0,0,0])

    for x in axis_values:
        for y in axis_values:
            for z in axis_values:
                points = np.vstack((points, np.array([x,y,z])))

    return points[1:, :]

def find_volume_fraction(cubicle_overlap):
    maximum_volume = 8

    volume_fraction = cubicle_overlap / maximum_volume
    
    if volume_fraction > 1:
        volume_fraction = 1

    return volume_fraction

points = points_of_regular_grid_generation()

hexahedra = np.zeros(shape = (points.shape[0], 8, 3))

for hexahedron_index in range(hexahedra.shape[0]):
    hexahedra[hexahedron_index, :, :] = generate_cube_from_center(points[hexahedron_index, :])


def estimate_curvature(a):
    """
    Estimate the curvature via divergence of gradient of relative overlaps

    a: list of relative overlaps
    """
    points = points_of_regular_grid_generation()

    hexahedra = np.zeros(shape = (points.shape[0], 8, 3))

    for hexahedron_index in range(hexahedra.shape[0]):
        hexahedra[hexahedron_index, :, :] = generate_cube_from_center(points[hexahedron_index, :])

    # Transforming the list to the 3D dimension that np.gradient will use
    roverlaps_3d = np.zeros((3,3,3))

    count = 0

    for x_index in range(3):
        for y_index in range(3):
            for z_index in range(3):
                roverlaps_3d[x_index, y_index, z_index] = a[count]
                count += 1

    grads = np.gradient(roverlaps_3d)

    normals = np.stack((grads[0], grads[1], grads[2]))

    for x in range(3):
        for y in range(3):
            for z in range(3):
                if np.linalg.norm(normals[:, x, y, z]) != 0.0:
                    normals[:, x, y, z] = normals[:, x, y, z] / np.linalg.norm(normals[:, x, y, z])

    h_x = np.gradient(normals[0,:,:,:])[0]
    h_y = np.gradient(normals[1,:,:,:])[1]
    h_z = np.gradient(normals[2,:,:,:])[2]

    # getting the unit value
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if np.linalg.norm(h_x) != 0.0:
                    h_x = h_x / np.linalg.norm(h_x)

                if np.linalg.norm(h_y) != 0.0:
                    h_y = h_y / np.linalg.norm(h_y)

                if np.linalg.norm(h_z) != 0.0:
                    h_z = h_z / np.linalg.norm(h_z)

    k = np.abs(h_x + h_y + h_z)

    return k[1,1,1]