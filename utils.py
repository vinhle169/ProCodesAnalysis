import os
import seaborn as sns
import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageio import volread as imread
from skimage.filters import threshold_otsu, rank, threshold_local
from skimage import img_as_ubyte
from torchvision import transforms


def otsu_threshold_channelwise(image):
    mask = []
    image = image.numpy()
    for channel in image:
        thresh = threshold_otsu(channel)
        new_ch = channel > thresh
        mask.append(new_ch)
    mask = np.array(mask)
    new_img = np.multiply(image, mask)
    running_diff = np.sum(new_img > 0)/np.sum(image > 0)
    return torch.Tensor(new_img), running_diff


def random_channel(image, multi_image=False):
    # assuming channels in the 1st dimension
    shape = image.shape
    channel_i = np.random.randint(0, shape[0])
    if torch.is_tensor(image):
        y = image.clone().detach()
        im = np.delete(y, channel_i, 0)
    else:
        im = np.delete(image.copy(), channel_i, 0)
    return image[channel_i, :, :], im, channel_i


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


def normalize_array(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def plot_procode_torch(path, img_shape=(2048, 2048, 3), amplify=0, normalize=False, filename=None):
    img = torch.load(path)
    if amplify:
        print('amplified')
        img *= amplify
        # normalize to max of 1
    if normalize:
        img /= img.max()
    if img.size()[0] == 1:
        img = img[0]
    fig, axs = plt.subplots(1, 4)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    if img.size()[0] == 4:
        for channel in range(4):
            one_channel = np.zeros(img_shape)
            mini = img[channel].view(img_shape[0:2]).detach()
            mini = mini.numpy()
            if channel == 0:
                axs[channel].imshow(mini)
                continue
            one_channel[:, :, channel-1] = mini
            axs[channel].imshow(one_channel)

    elif img.size()[0] == 3:
        combined = []
        for channel in range(3):
            one_channel = np.zeros(img_shape)
            mini = img[channel].view(img_shape[0:2]).detach()
            mini = mini.numpy()
            one_channel[:, :, channel] = mini
            axs[channel + 1].imshow(one_channel)
            mini = np.reshape(mini, img_shape[0:2])
            combined.append(mini)
        combined = np.stack(combined, -1)
        axs[0].imshow(combined)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()




def display_codebook(path):
    codebook = pd.read_csv(path, sep='.', index_col=0)
    sns.heatmap(codebook)
    plt.show()


def fixed_blobs(img_shape, v_stride: int, w_stride: int, blob_len: int, v_padding: int = 0,
                w_padding: int = 0):
    """
    :param img_shape: tuple/list (h x w)
    :param v_stride: stride vertically between top-left points of blobs
    :param w_stride: stride horizontally between top-left points of blobs
    :param blob_len: length of the sides of the blobs, they are square
    :param v_padding: padding top and bottom where there shouldn't be blobs
    :param w_padding: padding left and right where there shouldn't be blobs
    :
    """

    # make sure parameters are correct
    assert blob_len < img_shape[0] and blob_len < img_shape[1]
    assert v_stride < img_shape[0] and v_padding < img_shape[0]
    assert w_stride < img_shape[1] and w_padding < img_shape[1]
    zeros = torch.zeros(img_shape)
    flag = True
    # i, horizon index ; j, vert index
    i, j = 0, 0
    while flag:
        if i == 0:
            i += w_padding
        if j == 0:
            j += v_padding
        if i > (img_shape[1]) - w_padding:
            i = 0
            j += v_stride
            if j > (img_shape[0]) - v_padding:
                break

        else:
            end_w = i + blob_len
            if end_w > (img_shape[1]) - w_padding:
                end_w = (img_shape[1]) - w_padding
            end_v = j + blob_len
            if end_v > (img_shape[0]) - v_padding:
                end_v = (img_shape[0]) - v_padding
            zeros[j: end_v, i: end_w] = 1
            i += w_stride
    return zeros


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
    return img, img * img_c


def plot_loss(train_losses, title, savefig=False):
    x = [i for i in range(len(train_losses))]
    y = train_losses
    data = {"Epochs": x, "Loss": y}
    sns.lineplot(data=data, x="Epochs", y="Loss").set(title=title)
    if savefig:
        plt.savefig("Loss Graph.png")
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
    # blob_mask = fixed_blobs(img_shape=(2048, 2048), v_stride=100, w_stride=100, blob_len=50, v_padding=100,
    #                         w_padding=100)
    # blob_mask = torch.Tensor(blob_mask)
    # running_diff = 0
    # for filename in tqdm(os.listdir(path + 'slices/')):
    #     img = torch.load(path + 'slices/' + filename)
    #     img_o, r_diff = otsu_threshold_channelwise(img)
    #     blobs = torch.mul(img_o, blob_mask)
    #     combined = torch.amax(img, 0).view((1, 2048, 2048))
    #     inp = torch.cat((combined, blobs))
    #     running_diff += r_diff
    #     torch.save(inp, path + 'fixed_blobs_otsu/' + filename)
    # print(running_diff / len(list(os.listdir(path + 'slices/'))) * 100,
    #       ' average fraction of image kept with thresholding')

    # img = Image.open("doggo.png")
    # print(img.size)
    # convert_tensor = transforms.Compose([transforms.PILToTensor()])
    # img = convert_tensor(img)
    # torch.save(img, 'dog.pt')
    # filename = 'F030_trim_manual_0.pt'
    # img = torch.load('/nobackup/users/vinhle/data/one_image_truth_downsamp/F030_trim_manual_0.pt')
    # new_img, running_diff = otsu_threshold_channelwise(img)
    # torch.save(new_img, '/nobackup/users/vinhle/data/one_img_truth_downsamp_otsu/F030_trim_manual_0.pt')
    # print(running_diff)
    plot_procode_torch('/nobackup/users/vinhle/data/one_img_truth_downsamp_otsu/F030_trim_manual_0.pt',
                       img_shape=(256,256,3), filename='otsu_gt.png')
    # print(img.size())
    # resized_img = transforms.Resize(size=(256, 256))(img)
    # torch.save(resized_img, 'F030_resize.pt')

    # img = torch.load('F050_trim_manual_7.pt')
    # x,y,z = img.numpy()
    # fig, axs = plt.subplots(1,3)
    # for i,j in zip(range(len(axs)),[x,y,z]):
    #     axs[i].imshow(j, cmap='inferno')
    # plt.show()

    # img = torch.load('F031_trim_manual_4.pt')
    # nimg = otsu_threshold_channelwise(img)
    # x1, y1, z1 = img.numpy()
    # x, y, z = nimg
    # fig = plt.figure(constrained_layout=True)
    # fig.suptitle('Blob Mask')
    # images = [[x1, y1, z1], [x, y, z]]
    # subfigs = fig.subfigures(nrows=2, ncols=1)
    # titles = ['Original', 'Masked']
    # for row, subfig in enumerate(subfigs):
    #     subfig.suptitle(titles[row])
    #     axs = subfig.subplots(nrows=1, ncols=3)
    #     for col, ax in enumerate(axs):
    #         i = ax.imshow(images[row][col] * 100, cmap='inferno')
    #         ax.set_title(f'Channel {col}')
    # plt.show()
    # blob_mask = fixed_blobs(img_shape=(2048, 2048), v_stride=100, w_stride=100, blob_len=50, v_padding=100,
    #                         w_padding=100)
    # blob_mask = torch.Tensor(blob_mask)
    # blobs = torch.mul(img, blob_mask)
    # x, y, z = blobs.numpy()
    # x1, y1, z1 = img.numpy()
    # fig = plt.figure(constrained_layout=True)
    # fig.suptitle('Fixed Blob Mask')
    # images = [[x1, y1, z1], [x, y, z]]
    # subfigs = fig.subfigures(nrows=2, ncols=1)
    # titles = ['Original', 'Masked']
    # for row, subfig in enumerate(subfigs):
    #     subfig.suptitle(titles[row])
    #     axs = subfig.subplots(nrows=1, ncols=3)
    #     for col, ax in enumerate(axs):
    #         i = ax.imshow(images[row][col] * 100, cmap='inferno')
    #         ax.set_title(f'Channel {col}')
    # plt.show()

