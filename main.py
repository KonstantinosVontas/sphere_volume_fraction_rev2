from asyncore import write
from cgi import test
from random import random
import overlap
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from tqdm import tqdm

NUMBER_OF_SAMPLES = 2000

#normalList = []

def random_normal_generation():
    #n = np.array([1,0,0])

    #nx, ny, 0  ::
    #n = np.zeros((1, 3))
    #n[0, 0:2] = np.random.normal(size=(1, 2))
    #n = n / np.sqrt(np.sum(n ** 2))

    #nx, ny, nz ::
    n = np.random.normal(size=(1, 3))
    n = n / np.sqrt(np.sum(n ** 2))

    #normalList.append(n)


    return n


def generate_random_point_in_center_cell():

    points = (np.random.rand(1, 3)) -0.5

    #print('The points from generate_random_point_in_center_cell are: ', points)


    return points


coordinates = []
volume_fraction = []

def generate_random_sphere():
    normal = random_normal_generation()
    radius = np.random.uniform(8, 40)

    point = generate_random_point_in_center_cell()

    center = point - radius * normal

    coordinates = np.vstack(center)
    #print(coordinates)

    return center,radius,overlap.Sphere(tuple(center[0]), radius)


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

    #print("This is the volume fraction", volume_fraction)
    return volume_fraction

for ii in range(10):

    points = points_of_regular_grid_generation()

    hexahedra = np.zeros(shape=(points.shape[0], 8, 3))

    for hexahedron_index in range(hexahedra.shape[0]):
        hexahedra[hexahedron_index, :, :] = generate_cube_from_center(points[hexahedron_index, :])
        #print('this is alpha', aa)

    with open('overlap_curvature_h1.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)

        for sample_num in range(int(NUMBER_OF_SAMPLES)):
            center,radius,sphere = generate_random_sphere()
            curvature = 2 / sphere.radius

            row = []

            for hexahedron in hexahedra:
                row.append(find_volume_fraction(overlap.overlap(sphere, overlap.Hexahedron(hexahedron))))

            row.append(curvature)

            writer.writerow(row)


    ind = []
    for i in range(len(points)):
        if points[i, 2] == 0:
            ind.append(i)

    print(points[ind])


    row=np.array(row)
    a=row[ind]



    y=np.linspace(-1,1,3)
    z=np.linspace(-1,1,3)
    Y,Z=np.meshgrid(y,z)
    Y=Y.astype(int)
    Z=Z.astype(int)

    a=np.zeros(Z.shape)

    for i in range(3):
        for j in range(3):
            a[i,j]=row[ind][i+3*j]
    a=np.flip(a,0)


    zpos=0 # since we are showing the normal on the z axis plane




    cz=center[0,2]

    r_cut=np.sqrt(radius**2-(cz-zpos)**2) # radius of cut sphere

    circle=plt.Circle((center[0,0],center[0,1]),r_cut,color='black',fill=False, linewidth=2.5) # draw a sphere
    fig,ax=plt.subplots()

    im = ax.imshow(a, interpolation='nearest', extent=[-1.5, 1.5, -1.5, 1.5])
    ax.add_patch(circle)
    fig.colorbar(im)
    ax.scatter(points[0:9,1],points[0:9,2])
    ax.set_xlim([-1.6, 1.6])
    ax.set_ylim([-1.6, 1.6])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Volume Fraction')
    plt.savefig('volume fraction 2D plot-{:}.pdf'.format(ii))
    #plt.show()
    plt.draw()
    plt.pause(0.1)
    plt.close()



