import pandas as pd
from imageio.v2 import volread
import seaborn as sns
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

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
    img = volread('data/F05.tif')
    # img = np.amax(img, axis=0)
    img = img[0]
    img = img.astype(np.float32) / img.max((1, 2), keepdims=True)  # equalize channels, so it better matches our assumptions on codebook
    print(img.shape)
    markers = pd.read_csv('markers.csv')[:24]
    codebook = pd.read_csv('codebook.csv', sep='.', index_col=0)
    img = img[markers['marker_name'].isin(codebook.index)]
    A = codebook.values.copy().T
    x = img.copy()
    max_components = 3
    fudge_factor = 0.8
    z = matching_pursuit(x, A, max_components, fudge_factor)

    cm = plt.get_cmap('tab20').copy()
    cm.set_bad('k')
    plt.figure(figsize=(12, 12))
    plt.gca().matshow(np.where(z.max(0) == 0, np.nan, z.argmax(0)), cmap=cm)
    plt.savefig('tif.png')