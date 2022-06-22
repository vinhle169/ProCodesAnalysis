import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imageio import volread as imread
from skimage.filters import threshold_otsu, rank

def load_tif(path):
    img = imread(path)
    # equalizes/normalizes channels mark
    img = img.astype(np.float32) / img.max((1, 2), keepdims=True)
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


def color_blobs(img, blob_radius=1, num_blobs=1):
    '''

    :param img:
    :return: grayscale image where n blobs are filled in with the right color
    '''
    def get_neighbors(img, index):
        # up, down, left, right: do this later
        pass
    possible_blobs = np.transpose(np.nonzero(img > 0))
    used = set()
    print(np.array([2, 752]) in possible_blobs)
    for i in range(num_blobs):
        idx = np.random.randint(0, possible_blobs.shape[0])
        blob_index = possible_blobs[idx]
        used.add(blob_index)
    # return coordinates that are in the blob(s)
    return used


# approximately solve min_z ||zA-x||^2_2 st ||z||_0 <= max_iters
def matching_pursuit(x, A, max_iters, thr=1):
    '''
    :param x: Image
    :param A: codebook values transposed (7 x 19)
    :param max_iters: number of iterations aka the number of components
    :param thr: scaling factor to otsu threshold
    :return:
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
    active_set[x_norm < max_norm] = False

    for t in range(max_iters):
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

if __name__ == '__main__':
    x = load_tif('F000_max.tif')
    markers = pd.read_csv('markers.csv')
    A = load_codebook('codebook.csv', values=False)
    x = x[markers['marker_name'].isin(A.index)]
    A = A.values.copy().T
    z = matching_pursuit(x, A, 3)
    print(z.shape)
    test_img = z[1]
    print(test_img.shape)
    print(np.min(test_img))
    print(np.max(test_img))
    color_blobs(test_img, 2)