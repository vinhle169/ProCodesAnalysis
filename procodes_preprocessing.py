import time
import torch
import pandas as pd
from imageio.v2 import volread
import seaborn as sns
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing, remove_small_objects
import matplotlib.ticker as ticker
from utils import make_plotable

def load_tif(path):
    img = volread(path)
    img = img.astype(np.float32) / img.max((-1, -2), keepdims=True)
    return img


def load_codebook(path, values=True):
    codebook = pd.read_csv(path, sep='.', index_col=0)
    if values:
        codebook = codebook.values.copy().T
    return codebook


def display_codebook(path):
    codebook = pd.read_csv(path, sep='.', index_col=0)
    sns.heatmap(codebook)
    plt.show()


def prepare_mp_input(img, codebook, markers):
    '''
    :param img:
    :param codebook:
    :param markers:
    :return:
    '''
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    if img.shape[0] != len(markers.index.tolist()):
        img = img.transpose(1, 0, 2, 3)
    assert img.shape[0] == len(markers.index.tolist()), 'image input should be transposed to CZYX'

    ## Background Subtraction ## ALL CHANNELS except DNA channels
    for ch in range(4, img.shape[0]):
        r = ch % 4
        if r == 1:  # 488 channel # green
            img[ch, ...] = img[ch, ...] - img[1, ...]  # subtract background

        elif r == 2:  # 555 channel # yellow
            img[ch, ...] = img[ch, ...] - img[2, ...]  # subtract background

        elif r == 3:  # 637 channel # red
            img[ch, ...] = img[ch, ...] - img[3, ...]  # subtract background

        img[img < 0] = 0  # clip negatives

    img = img[markers['marker_name'].isin(codebook.index)]

    # calculate normalization factors
    img_min = np.quantile(img, 0.75, (1, 2, 3), keepdims=True)  # most of each channel is probably background
    img_max = np.quantile(img, 0.999, (1, 2, 3), keepdims=True)
    img = (img.astype(np.float32) - img_min) / (img_max - img_min)
    img = img.clip(0, 1)  # clip negatives
    return img


# approximately solve min_z ||zA-x||^2_2 st ||z||_0 <= max_iters
def matching_pursuit(x, A, max_iters, thr=1):
    '''
    :param x: Image
    :param A: codebook values transposed
    :param max_iters: number of iterations aka the number of components
    :param thr: scaling factor to otsu threshold
    :return z: deconvolution of the image
    '''
    # this is necessary if doing more than one iteration, for the residuals update to work
    A = A / np.linalg.norm(A, axis=1)[:, None]
    z = np.zeros((A.shape[0], *x.shape[1:]))
    x = x.copy()  # initialize "residuals" to the image itself

    # mask for whether each pixel has converged or not; we already know background's zero
    active_set = np.ones(x.shape[1:], dtype=bool)
    x_norm = np.linalg.norm(x, axis=0, ord=2)

    # pick it using the otsu threshold, as estimate of noise level
    max_norm = threshold_otsu(x_norm)
    max_norm *= thr  # hack; otsu just not great. see 'enhance neurites' in CellProfiler?
    active_set[x_norm <= max_norm] = False

    for t in tqdm(range(max_iters)):
        # project dictionary on residual image
        Ax = A @ x[:, active_set]
        # pick index with max projection
        k_max = Ax.argmax(0)
        # set it to active
        z[k_max, active_set] = Ax[k_max, range(len(k_max))]
        # subtract off contribution
        x[:, active_set] -= A[k_max].T * z[k_max, active_set]
        # mark pixels with sufficiently small residual norm as done
        x_norm = np.linalg.norm(x, axis=0, ord=2)
        active_set[x_norm < max_norm] &= False

    return z


def prepare_grayscale_mask(img, footprint_size=(12, 12)):
    '''
    :param img:
    :param footprint_size:
    :return:
    '''
    im = img.copy()
    img_channels = []
    footprint = np.ones(footprint_size)
    for img_chan in tqdm(im):
        area_closed = closing(img_chan, footprint)
        opened = opening(area_closed, footprint)
        img_chan[img_chan == 0] = np.nan
        opened = np.where(opened > np.nanquantile(img_chan, 0.90), 1, 0)
        opened_label = label(opened, connectivity=2)
        img_channels.append(opened_label > 0)
    return np.stack(img_channels, axis=0)



def make_grayscale(img, footprint_size=(12, 12)):
    img_maxed = img.max(axis=1)
    del img
    masks = prepare_grayscale_mask(img_maxed, footprint_size=footprint_size)
    assert img_maxed.shape == masks.shape, "Shape of Z-Flattened Image must match Stack of Masks"
    # get grayscale values
    # img_maxed = torch.from_numpy(img_maxed)
    grayscale = img_maxed.mean(0)
    input_img = np.zeros_like(img_maxed)
    # place the grayscale values in each channel
    for i in tqdm(range(img_maxed.shape[0])):
        # mask_i = masks[i]
        # img_i = img_maxed[i]
        input_img[i,:,:] = grayscale

    # mask to place color where it should be colored
    input_img[masks] = img_maxed[masks]
    return input_img


if __name__ == '__main__':
    # start_time = time.time()
    img = volread('data/F05.tif')
    # codebook = pd.read_csv('codebook.csv',sep='.',index_col=0)
    # markers = pd.read_csv('markers.csv')
    # idx = codebook.index.tolist()
    # idx.pop(2)
    # codebook = codebook.reindex(idx + ['FLAG'])
    # grab the correct rows
    # codebook = codebook[['A2', 'AA5', 'D4']]
    # our input to create train data
    # print('preparing input')
    # x = prepare_mp_input(img, codebook, markers)
    # print('making grayscale')
    # np.save('prepped_F05',x)
    f = np.load('prepped_F05.npy')
    nuclear_channel = img.max(0)[0]
    # train_image = make_grayscale(x, footprint_size=(8,8))
    # train_image = torch.from_numpy(train_image)
    # torch.save(train_image, '/nobackup/users/vinhle/data/procodes_data/unet_train_single/train/single.pt')
    # print(train_image.shape)

    # f1_test = f1[1:4]
    # f1_test = make_plotable(f1_test, numpy_like=True)
    # fig, ax = plt.subplots(1,2)
    # fig.set_figheight(15)
    # fig.set_figwidth(15)
    # ax[0].imshow(f_test)
    # ax[1].imshow(f1_test)
    # ax[0].axis('off')
    # ax[1].axis('off')
    # plt.savefig('test_data.png')
    # plt.clf()
    print(f.shape)
    for i in range(len(f)):
        fig,ax = plt.subplots(1,2)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        fmax = f[i].max(0)
        ax[0].imshow(nuclear_channel, vmin=np.quantile(nuclear_channel, 0.05), vmax=np.quantile(nuclear_channel, 0.95))
        ax[1].imshow(fmax, vmin=np.quantile(fmax, 0.05), vmax=np.quantile(fmax, 0.95))
        plt.show()
        plt.close()

    # opened = f1max
    # footprint = np.ones((8, 8))
    # area_closed = closing(opened, footprint)
    # footprint = np.ones((8, 8))
    # opened = opening(area_closed, footprint)
    # f1max[f1max==0] = np.nan
    # opened = np.where(opened > np.nanquantile(f1max, 0.90), 1, 0)
    # ax[1].matshow(opened)
    #
    # opened_label, num = label(opened, return_num=True, connectivity=2)
    # opened_label = remove_small_objects(opened_label)
    # ax[2].matshow(opened_label)
    # plt.show()
    # plt.tight_layout()
    # plt.savefig("test.png", bbox_inches='tight')

    # A = codebook.values.copy().T
    # print(x.shape)
    # print(A.shape)
    # max_components = 3  # assuming maximum of three overlapping neurites..
    # fudge_factor = 0.25
    #
    # # run MP
    # z = matching_pursuit(x, A, max_components, fudge_factor)
    # z = z.clip(0, 1)
    # z = z.max(1)
    # print(z.shape)
    # z = torch.from_numpy(z)
    # torch.save(z, '/nobackup/users/vinhle/data/procodes_data/unet_train_single/truth/single.pt')
    # print('total time: ', time.time() - start_time)
    # cm = ListedColormap(['red','green','blue'])
    # cm.set_bad('black')
    # plt.figure(figsize=(12, 12))
    # plt.gca().matshow(np.where(z.max(1).max(0) == 0, np.nan, z.max(1).argmax(0)), cmap=cm)
    # plt.savefig('matching_pursuit_test.png')
    # 7 channels but each has a unique combination of the 3 channels
    # with these combinations we can fill in the training data.