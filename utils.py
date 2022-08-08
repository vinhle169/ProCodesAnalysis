import os
import seaborn as sns
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import volread as imread
from skimage.filters import threshold_otsu, rank, threshold_local
from skimage import img_as_ubyte


def random_channel(image, multi_image=False):
    # assuming channels in the 1st dimension
    shape = image.shape
    channel_i = np.random.randint(0, shape[0])
    if torch.is_tensor(image):
        y = image.clone().detach()
        im = np.delete(y, channel_i, 0)
    else:
        im = np.delete(image.copy(), channel_i, 0)
    return image[channel_i,:,:], im, channel_i

def load_tif(path):
    img = imread(path)
    # equalizes/normalizes channels mark
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

# noinspection PyTypeChecker
def color_blobs(img, blob_radius=50, num_blobs=1):
    '''
    :param img: in the shape -> channel x height x width
    :return: grayscale image where n blobs are filled in with the right color
    '''
    def set_blob(img, index):
        # assumes 2d height x width
        i, j = index
        img[i - blob_radius:i + blob_radius + 1, j - blob_radius:j + blob_radius + 1] = 1
        return img

    img_c = torch.clone(img)
    for channel in range(img_c.shape[0]):
        img_i = img[channel]
        # possible_blobs = torch.nonzero(img_i > 0)
        threshold = img_i >= (torch.max(img_i) * .3)
        possible_blobs = torch.nonzero(threshold > 0)
        blob_mask = torch.zeros(img_i.shape)
        for i in range(num_blobs):
            # choose a random center for a blob
            idx = np.random.randint(0, possible_blobs.shape[0])
            blob_index = possible_blobs[idx]
            i, j = blob_index
            blob_mask = set_blob(blob_mask, blob_index)
        img_c[channel] = blob_mask

    # return coordinates that are in the blob(s)
    return img, img*img_c


def plot_loss(train_losses, title, savefig=False):
    x = [i for i in range(len(train_losses))]
    y = train_losses
    data = {"Epochs":x, "Loss":y}
    sns.lineplot(data = data, x = "Epochs", y = "Loss")
    if savefig:
        plt.savefig("Loss Graph.png").set(title=title)
    return None

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
    print(A)
    A = A / np.linalg.norm(A, axis=1)[:, None]
    print(A)
    z = np.zeros((A.shape[0], *x.shape[1:]))
    print(A.shape, x.shape)
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
    # path = '/nobackup/users/vinhle/data/'
    # for filename in os.listdir(path + 'slices/'):
    #     img = torch.load(path + 'slices/' + filename)
    #     img, blob = color_blobs(img, num_blobs=3, blob_radius=200)
    #     combined = torch.amax(img, 0).view((1,2048,2048))
    #     inp = torch.cat((combined, blob))
    #     torch.save(inp, path + 'blobs/' + filename)

    #
    # path = 'data/'
    # filename = 'F031_trim_manual_1.pt'
    # img = torch.load(path + 'slices/' + filename)
    # img, blob = color_blobs(img, num_blobs=3, blob_radius=200)
    # combined = torch.amax(img, 0)
    # sns.heatmap(combined, xticklabels=100, yticklabels=200, vmin=0, vmax=1)
    # inp = torch.cat((combined, blob))
    # print(blob.shape)
    # fig, axs = plt.subplots(3,2)
    # fig.set_figheight(15)
    # fig.set_figwidth(15)
    # axs[0][0].set_title('Images With Blobs Sampled')
    # sns.heatmap(blob[0], ax=axs[0,0], xticklabels=100, yticklabels=200, vmin=0, vmax=1)
    # sns.heatmap(blob[1], ax=axs[1, 0], xticklabels=100, yticklabels=200, vmin=0, vmax=1)
    # sns.heatmap(blob[2], ax=axs[2, 0], xticklabels=100, yticklabels=200, vmin=0, vmax=1)
    # axs[0][1].set_title('Original Images')
    # sns.heatmap(img[0], ax=axs[0, 1], xticklabels=100, yticklabels=200, vmin=0, vmax=1)
    # sns.heatmap(img[1], ax=axs[1, 1], xticklabels=100, yticklabels=200, vmin=0, vmax=1)
    # sns.heatmap(img[2], ax=axs[2, 1], xticklabels=100, yticklabels=200, vmin=0, vmax=1)
    # rows = ['Flag', 'C', 'S']
    # for ax, row in zip(axs[:, 0], rows):
    #     ax.set_ylabel(row, rotation=0, size='large')
    # plt.show()
    pass



