import os
import json
import torch
import networkx
import numpy as np
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from imageio.v2 import imread
from skimage.measure import regionprops
from skimage import transform
from utils import make_plotable
from PIL import Image
from torchvision.transforms import Resize
from utils import normalize_array, normalize_array_t

# what to do:
# load in cell masks
# grab the centroids from each cell
# flip them s.t. x,y are the right values and add border points to them as well e.g. [[0,0],[0,2048],[2048,0],[2048,2048]]
# call Voronoi on the centroids, find shared vertices (which means an edge is shared)
# use networkx greedy coloring to 4-color the graph
# assign numbers to channels


def vor_and_create_graph(centroids, x_max, y_max):
    vor = Voronoi(centroids)
    # voronoi_plot_2d(vor)
    # plt.savefig('vor.png')
    centroids_to_vertices = {}
    check_valid = lambda x,y: 0<x<=x_max and 0<y<=y_max
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


def graph_color(cmi, shape, with_coloring=False):
    '''
    :param cmi: cell mask image
    :param shape: the image shape
    :return:
    '''

    boundary_centroids = {}
    for region in regionprops(cmi):
        boundary_centroids[region.label] = np.array(region.centroid).astype('int32')

    # grabbing y,x because its row/col for the location
    centroids = np.array(
        [[x, y] for x, y in boundary_centroids.values()] + [[0, 0], [0, shape[1]], [shape[0], 0], [shape[0], shape[1]]])

    graph = vor_and_create_graph(centroids, shape[0]-1, shape[1]-1)
    # plt.clf()
    # pos = {i: i for i in graph}
    # networkx.draw_networkx(graph, pos)
    # plt.savefig('graph.png')
    coloring = networkx.greedy_color(graph, strategy='saturation_largest_first')
    # print(coloring)
    # channel, height, width
    final_image = torch.zeros((4, shape[0], shape[1]))
    label_to_color = {'miscolors':0}
    for region in regionprops(cmi):
        # color_int is the channel the image will be in
        centroid = boundary_centroids[region.label]
        color_int = coloring[(centroid[0], centroid[1])]
        if color_int > 3:
            color_int = np.random.choice([0, 1, 2, 3])
            label_to_color['miscolors'] += 1
        label_to_color[int(region.label)] = int(color_int)
        final_image[color_int][tuple(region.coords.T)] = 1
    if with_coloring:
        return final_image, label_to_color
    return final_image

def make_plotable_4chan(img, train=False):
    '''
    function to make a 4 channeled image plotable as RGB image
    :param img: torch tensor image, in shape 4 x M x N
    :param train: whether or not it is a training image
    :return:
    '''
    # this line to prevent
    img = img.clone()
    if train:
        return make_plotable(img[:3])
    nonzero_idx = torch.nonzero(img[3] > 0.01)
    for pair in nonzero_idx:
        org_val = img[(3,)+tuple(pair)]
        img[(0,)+tuple(pair)], img[(1,)+tuple(pair)], img[(2,)+tuple(pair)] = org_val, org_val, org_val
    return make_plotable(img[:3,:,:])


def hpa_kaggle_graph_color(cell_path, nuclei_path, org_path, metadata_path, new_train_path, new_truth_path, img_size=(4, 2048, 2048)):
    '''
    Given 3 paths, cell path and nuclei path leading to segmentation masks for original images
    This function will use these three types of images to make training data, on given paths
    Train -> cell nuclei will be colored in, everything else grayscale
    Truth -> cell nuclei not colored, everything else colored
    :param cell_path: path to cell outlines
    :param nuclei_path: path to cell nuclei
    :param org_path: path to the original images
    :param metadata_path: path to metadata
    :param new_train_path: new path for training images
    :param new_truth_path: new path for truth images
    :param img_size: image sizes for output
    :return: none, create data images
    '''
    # metadata will contain {file->{regions->colors}}
    metadata = {}
    miscolors = 0
    aug = Resize((img_size[1], img_size[2]), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=False)
    for filename in tqdm(os.listdir(cell_path)):
        # grab the unique part of the filename
        f_name = filename[:36]
        segmentation_mask_cell_path = cell_path+filename
        segmentation_mask_nuclei_path = nuclei_path+filename
        # get the corresponding matrix for segmentation masks
        # just the nuclei
        segmentation_mask_nuclei = np.load(segmentation_mask_nuclei_path)
        smn = segmentation_mask_nuclei[segmentation_mask_nuclei.files[0]].astype(np.int32)
        if smn.shape != img_size[1:]:
            smn = np.array(aug(torch.from_numpy(smn).view(1, smn.shape[0], smn.shape[1])))[0]
        # the cell
        segmentation_mask_cell = np.load(segmentation_mask_cell_path)
        smc = segmentation_mask_cell[segmentation_mask_cell.files[0]].astype(np.int32)
        if smc.shape != img_size[1:]:
            smc = np.array(aug(torch.from_numpy(smc).view(1, smc.shape[0], smc.shape[1])))[0]
        # get the image of the cells as a grayscale image so we can get the intensities of every pixel
        img_cell = imread(org_path+f_name+'_y.png', as_gray=True)
        if img_cell.shape != img_size[1:]:
            img_cell = aug(torch.from_numpy(img_cell).view(1, img_cell.shape[0], img_cell.shape[1]))[0]
        img_cell = normalize_array_t(img_cell)
        max_clip = torch.max(img_cell)
        # stack for elementwise multiplication
        img_cell = torch.stack([img_cell,img_cell,img_cell,img_cell])

        # graph color the cells to ensure no adjacent cells are the same color

        colored, coloring = graph_color(smc, img_size[1:], with_coloring=True)
        miscolors += coloring['miscolors']
        metadata[f_name] = coloring
        # multiply the colored in version with the grayscale version to get the right intensities for each pixel
        output_img = torch.mul(colored, img_cell)
        # clamp our output to ensure its in the correct range
        output_img = torch.clamp(output_img, min=0, max=max_clip)
        # delete these to save space
        del smc
        del img_cell
        truth = output_img
        # create our corresponding training image, in which the cell body will be grayscale
        train = torch.clone(output_img)
        grayscale = train.mean(0)
        train[0, :, :] = train[1, :, :] = train[2, :, :] = grayscale
        for region in regionprops(smn):
            color = coloring[region.label]
            train[(color,)+tuple(region.coords.T)] = .33
        del smn

        torch.save(train, new_train_path + f_name + '.pt')
        torch.save(truth, new_truth_path + f_name + '.pt')
        del train
        del truth

    print(miscolors)
    with open(metadata_path+'metadata.json', 'w') as fp:
        json.dump(metadata, fp)
    return None



if __name__ == '__main__':
    cell_path = '/nobackup/users/vinhle/data/hpa_data/hpa_cell_mask/'
    nuclei_path = '/nobackup/users/vinhle/data/hpa_data/hpa_nuclei_mask/'
    org_path = '/nobackup/users/vinhle/data/hpa_data/hpa_original/'
    org_path2 = '/nobackup/users/vinhle/data/hpa_data/hpa_original_test/'
    new_train_path = '/nobackup/users/vinhle/data/hpa_data/hpa_train/train_gc_512/'
    new_truth_path = '/nobackup/users/vinhle/data/hpa_data/hpa_train/truth_gc_512/'
    metadata_path = '/nobackup/users/vinhle/data/hpa_data/hpa_train/'
    # img_size = (4, 512, 512)
    # hpa_kaggle_graph_color(cell_path, nuclei_path, org_path, metadata_path, new_train_path, new_truth_path,
    #                           img_size=img_size)
    # path = 'data/'
    # trainp = path + 'train_test_hpa.pt'
    # truthp = path + 'truth_test_hpa.pt'
    # fig, ax = plt.subplots(1,2)
    # print(torch.load(trainp).shape)
    # print(torch.load(truthp).shape)
    # train = make_plotable_4chan(torch.load(trainp), train=True)
    # truth = make_plotable_4chan(torch.load(truthp))
    # ax[0].imshow(train)
    # ax[1].imshow(truth)
    # plt.show()
    # needs to be width x height x channel for plot
    # needs to be channel x height x width

    # get the corresponding matrix for segmentation masks
    # just the nuclei
    # file_path = 'data/cell_mask_test.npz'
    # file = np.load(file_path)
    # mask = file[file.files[0]].astype(np.int32)
    # mask = np.array(Image.fromarray(mask).resize((512, 512), Image.Resampling.NEAREST))
    # aug = Resize((512,512), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=False)
    # mask = np.array(aug(torch.from_numpy(mask).view(1,2048,2048)))[0]
    # colored = graph_color(mask, (512, 512))
    # colored_plotable = make_plotable_4chan(colored)
    # colored_plotable = make_plotable(colored)
    # print(colored_plotable[:,:,:3].shape)
    # plt.imshow(colored_plotable)
    # plt.savefig('colored_plot.png')
    # segmentation_mask_cell_path = 'data/wrong2.npz'
    # aug = Resize((img_size[1], img_size[2]), interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    #              antialias=False)
    # segmentation_mask_cell = np.load(segmentation_mask_cell_path)
    # smc = segmentation_mask_cell[segmentation_mask_cell.files[0]].astype(np.int32)
    # print(smc.max())
    # if smc.shape != img_size[1:]:
    #     smc = np.array(aug(torch.from_numpy(smc).view(1, smc.shape[0], smc.shape[1])))[0]
    # print(smc.max())
    # plt.imshow(smc)
    # plt.gca().invert_yaxis()
    # plt.show()
    # colored = graph_color(smc, (512, 512))
    # colored_plotable = make_plotable_4chan(colored)
    # plt.imshow(colored_plotable)
    # plt.show()
    x = torch.load('results/train_512.pt')
    x = make_plotable_4chan(x, train=True)
    plt.imshow(x)
    plt.show()
    plt.clf()
    x = torch.load('results/truth_512.pt')
    x = make_plotable_4chan(x)
    plt.imshow(x)
    plt.show()


