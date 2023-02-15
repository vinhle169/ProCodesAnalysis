import skimage
from skimage.future import graph
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, segmentation
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import spatial
from sklearn.neighbors import KDTree

import scipy
import networkx
from imageio.v2 import imread
from skimage.measure import regionprops
from tqdm import tqdm

# what to do:
# load in cell masks
# grab the centroids from each cell
# flip them s.t. x,y are the right values and add border points to them as well [[0,0],[0,2048],[2048,0],[2048,2048]]
# call Voronoi on the centroids, find shared vertices (which means an edge is shared)
# use networkx greedy coloring to 4-color the graph
# assign numbers to channels

file = np.load('data/cell_mask_test.npz')
mask = file['arr_0']
img = imread('data/test.png')


def get_vor_centroids(mask):
    boundary_centroids = {}
    for region in regionprops(mask):
        boundary_centroids[region.label] = np.array(region.centroid).astype('int32')

    centroids = np.array([[x,y] for y,x in boundary_centroids.values()] +[[0,0],[0,2048],[2048,0],[2048,2048]])
    return centroids

def vor_and_create_graph(centroids):
    vor = Voronoi(centroids)
    centroids_to_vertices = {}
    check_valid = lambda x,y: 0<x<=2047 and 0<y<=2047
    for i in range(len(vor.point_region)):
        corresponding_coord = tuple(vor.points[i].astype('int32'))
        vertices_indices = vor.regions[vor.point_region[i]]
        if not check_valid(corresponding_coord[0], corresponding_coord[1]):
            continue
        centroids_to_vertices[corresponding_coord] = set()
        for idx in vertices_indices:
            x,y = vor.vertices[idx].astype('int32')
            if not idx:
                continue
            else:
                centroids_to_vertices[corresponding_coord].add((x,y))

    seen = set((i,i) for i in centroids_to_vertices.keys())
    edges = {k:[] for k in centroids_to_vertices.keys()}
    for k,v in centroids_to_vertices.items():
        for k1,v1 in centroids_to_vertices.items():
            if (k,k1) in seen or (k1, k) in seen:
                continue
            seen.add((k,k1))
            seen.add((k1,k))
            if v.intersection(v1):
                edges[k].append(k1)
                edges[k1].append(k)
    graph = networkx.from_dict_of_lists(edges)
    return graph


def graph_color():
    coloring = networkx.greedy_color(graph)
    test = np.zeros((2048, 2048, 3)).astype('int32')
    for region in regionprops(mask):
        centroid = np.array(region.centroid).astype('int32')
        color_int = coloring[(centroid[1], centroid[0])]
        color = color_dict[color_int]
        test[tuple(region.coords.T)] = color
