from asyncore import write
from cgi import test
from random import random
import overlap
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

NUMBER_OF_SAMPLES = 2000


def random_normal_generation():
    v = np.random.normal(size=(1, 3))
    v = v / np.sqrt(np.sum(v ** 2))

    return v


def generate_random_point_in_center_cell():
    maximum_coordinate_value = 1

    points = (np.random.rand(1, 3) * ((1 * maximum_coordinate_value) - maximum_coordinate_value))

    return points


def generate_random_sphere():
    normal = random_normal_generation()
    radius = np.random.uniform(1, 10)

    point = generate_random_point_in_center_cell()

    center = point - radius * normal

    return overlap.Sphere(tuple(center[0]), radius)


def generate_cube_from_center(point):

    h = 1

    cube_vertices = np.array((
        (point[0] - h / 2, point[1] - h / 2, point[2] - h / 2),
        (point[0] + h / 2, point[1] - h / 2, point[2] - h / 2),
        (point[0] + h / 2, point[1] + h / 2, point[2] - h / 2),
        (point[0] - h / 2, point[1] + h / 2, point[2] - h / 2),
        (point[0] - h / 2, point[1] - h / 2, point[2] + h / 2),
        (point[0] + h / 2, point[1] - h / 2, point[2] + h / 2),
        (point[0] + h / 2, point[1] + h / 2, point[2] + h / 2),
        (point[0] - h / 2, point[1] + h / 2, point[2] + h / 2),
    ))
    return cube_vertices

def points_of_regular_grid_generation():
    axis_values = [-1, 0, 1] # was [-2, 0, 2]

    points = np.array([0, 0, 0])

    for x in axis_values:
        for y in axis_values:
            for z in axis_values:
                points = np.vstack((points, np.array([x, y, z])))

    return points[1:, :]


def find_volume_fraction(cubicle_overlap):
    maximum_volume = 1

    volume_fraction = cubicle_overlap / maximum_volume

    if volume_fraction > 1:
        volume_fraction = 1

    return volume_fraction


points = points_of_regular_grid_generation()

hexahedra = np.zeros(shape=(points.shape[0], 8, 3))

for hexahedron_index in range(hexahedra.shape[0]):
    hexahedra[hexahedron_index, :, :] = generate_cube_from_center(points[hexahedron_index, :])

with open('overlap_curvature_h1.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)

    for sample_num in range(int(NUMBER_OF_SAMPLES)):
        sphere = generate_random_sphere()
        curvature = 2 / sphere.radius

        row = []

        for hexahedron in hexahedra:
            row.append(find_volume_fraction(overlap.overlap(sphere, overlap.Hexahedron(hexahedron))))

        row.append(curvature)

        writer.writerow(row)



