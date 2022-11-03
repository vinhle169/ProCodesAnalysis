import os
import random
import json
import seaborn as sns
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageio import volread, imread
from skimage.filters import threshold_otsu
from unet import create_pretrained
from torchvision import transforms
import skimage
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
from skimage import morphology
import itertools

def otsu_threshold_channelwise(image, thresh_scale: float = 1.0):
    mask = []
    if image.size()[0] == 1:
        image = image[0]
    image = image.detach()
    image = image.cpu()
    image = image.numpy()
    for channel in image:
        thresh = threshold_otsu(channel) * thresh_scale
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
    img = volread(path)
    # equalizes/normalizes channels mark
    img = img.astype(np.float32) / img.max((-1, -2), keepdims=True)
    return img


def load_codebook(path, values=True):
    codebook = pd.read_csv(path, sep='.', index_col=0)
    if values:
        codebook = codebook.values.copy().T
    return codebook


def normalize_array_t(array):
    return (array - torch.min(array)) / (torch.max(array) - torch.min(array))


def normalize_array(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def plot_procode_torch(path, img_shape=(2048, 2048, 3), amplify=0, normalize=False, filename=None):
    if type(path) is str:
        img = torch.load(path)
    else:
        img = path
        path = 'No Name'
    if amplify:
        print('amplified')
        img *= amplify
        # normalize to max of 1
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
            if normalize:
                mini = normalize_array(mini)
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
            if normalize:
                mini = normalize_array(mini)
            one_channel[:, :, channel] = mini
            axs[channel + 1].imshow(one_channel)
            axs[channel + 1].axis('off')
            mini = np.reshape(mini, img_shape[0:2])
            combined.append(mini)
        combined = np.stack(combined, -1)
        axs[0].imshow(combined)
        axs[0].axis('off')
    plt.tight_layout()
    plt.suptitle(path, y=0.80, fontsize='xx-large')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()




def display_codebook(path):
    codebook = pd.read_csv(path, sep='.', index_col=0)
    sns.heatmap(codebook)
    plt.show()


def fixed_blobs(img_shape, v_stride: int, w_stride: int, blob_len: int, v_padding: int = 0,
                w_padding: int = 0, boolean: bool = False):
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
    if boolean:
        # returns a bool tensor works in torch version > 1.4
        return zeros > 0
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


def three_channel_grayscale(img, mask):
    assert img.size() == mask.size()
    grayscale = img.mean(2)
    input_img = torch.zeros_like(img)
    input_img[:,:,0] = input_img[:,:,1] = input_img[:,:,2] = grayscale
    input_img[mask] = img[mask]
    return input_img


def classification_accuracy(img, label, mask_over_zero):
    img_max = img.max(axis=1, keepdim=True).indices
    label_max = label.max(axis=1, keepdim=True).indices
    acc = img_max == label_max
    acc = acc.float()
    acc *= mask_over_zero.float()
    return torch.sum(acc)/torch.sum(mask_over_zero)


def preprocess_and_create_data(data_path: str, output_path: str, output_size: tuple, preprocess, thresh_scale: float=1.0):
    '''
    Function to preprocess and create new training data
    :param data_path:
    :param output_path:
    :param output_size:
    :param preprocess: a preprocessing function
    :return:
    '''
    mask = fixed_blobs(output_size[::-1], int(output_size[1]/16), int(output_size[1]/16),
                       int(output_size[1]/32), int(output_size[1]/64), int(output_size[1]/64), boolean=True)
    resizer = transforms.Resize(size=output_size[1:])
    for filename in tqdm(os.listdir(data_path)):
        org_img = torch.load(data_path+filename)
        img = resizer(org_img)
        label = torch.clone(img)
        # get mask covering each of the non-zero points
        abs = torch.abs(label)
        abs, _ = otsu_threshold_channelwise(abs)
        mask_over_zero = abs.sum(axis=0, keepdim=True)
        # make three channel grayscale image
        img = torch.stack([normalize_array_t(i) for i in img])
        buffer = torch.stack([i for i in img], -1)
        tcg = three_channel_grayscale(buffer, mask)
        # put images through preprocessing
        tcg = tcg.numpy()
        # tcg = preprocess(tcg)
        tcg = torch.Tensor(np.stack([tcg[:,:,i] for i in range(3)], 0))
        torch.save(img, output_path + 'truth/' + filename)
        torch.save(tcg, output_path + 'train/' + filename)
        torch.save(mask_over_zero, output_path + 'classification_mask/' + filename)
    print('~Finished!~')


def preprocess_main(data_path, output_path, output_size, model='resnet50', dataset='swsl', thresh_scale=1):
    preprocess_fn = create_pretrained(model, dataset, preprocess_only=True)
    try:
        os.mkdir(output_path[:-1])
        os.mkdir(output_path + 'train')
        os.mkdir(output_path + 'truth')
        os.mkdir(output_path + 'classification_mask')
    except:
        pass

    preprocess_and_create_data(data_path, output_path, output_size, preprocess_fn, thresh_scale)


def connected_components(image, min_size: int):
    '''
    needs image to be of size [channel, h, w]
    :param image:
    :return:
    '''
    if image.size()[0] == 1:
        image = image[0]
    image = image.detach()
    image = image.cpu()
    image = image.numpy()
    result = []
    object_features = dict()
    for c in range(len(image)):
        channel = image[c]
        blurred_img = skimage.filters.gaussian(channel, sigma=0.25)
        thresh = threshold_otsu(blurred_img)
        binary_mask = blurred_img > thresh
        object_mask = morphology.remove_small_objects(binary_mask, min_size)
        labeled_image, count = skimage.measure.label(object_mask, connectivity=2, return_num=True)
        obj_feat = skimage.measure.regionprops(labeled_image)
        result.append(labeled_image)
        obj_feats = [(objf["label"], objf["area"], objf["coords"]) for objf in obj_feat]
        obj_feats = sorted(obj_feats, key = lambda x: x[1], reverse=True)
        object_features[c] = obj_feats
    resulting_image = np.stack([i for i in result], -1)
    return resulting_image, object_features


def random_pixel_data(data_path: str, output_path: str, output_size: tuple, min_size: int, preprocess):

    resizer = transforms.Resize(size=output_size[1:])
    for filename in tqdm(os.listdir(data_path)):
        org_img = torch.load(data_path+filename)
        img = resizer(org_img)
        # make three channel grayscale image
        img = torch.stack([normalize_array_t(i) for i in img])
        _, obj_f = connected_components(img, min_size)
        buffer = torch.stack([i for i in img], -1)
        mask = torch.zeros_like(buffer)
        # do single pixel coloring based off largest component from each channel
        for i in range(3):
            coords = obj_f[i][0][2]
            row, col = coords[np.random.choice(len(coords))]
            mask[row][col][i] = 1

        # to get a bool array
        mask = mask > 0
        tcg = three_channel_grayscale(buffer, mask)
        # put images through preprocessing
        tcg = tcg.numpy()
        tcg = preprocess(tcg)
        tcg = torch.Tensor(np.stack([tcg[:,:,i] for i in range(3)], 0))
        torch.save(tcg, output_path + 'train_pixel/' + filename)
    print('~Finished!~')


def test_images(path, img_name='F030.pt', output_path='testest.png'):
    _, ax = plt.subplots(1,2)

    train = torch.load(path + 'train/' + img_name)
    train = torch.stack([normalize_array_t(i) for i in train], -1).numpy()
    ax[0].imshow(train)
    truth = torch.load(path + 'truth/' + img_name)
    truth = torch.stack([i for i in truth], -1).numpy()
    ax[1].imshow(truth)

    plt.savefig(output_path)


def remove_outliers(img):
    top = np.quantile(img, .99)
    img = np.where(img<top, img, 0)
    return img


def make_plotable(img, remove=False):
    img = img.detach().cpu().numpy()
    if remove:
        img = remove_outliers(img)
    img = np.stack([normalize_array(i) for i in img], -1)
    return img


def save_image(image, output_path):
    img = make_plotable(image)
    plt.imshow(img)
    plt.savefig(output_path)


def channel_transform(data_path_train, data_path_truth, new_data_path, num_channels):
    try:
        print('Creating output paths')
        os.mkdir(new_data_path[:-1])
        os.mkdir(new_data_path + 'train')
        os.mkdir(new_data_path + 'truth')
    except:
        print('Output paths already exist, permuting channels now')
    permutations = list(itertools.permutations(range(num_channels), num_channels))
    perm_name = [''.join([str(j) for j in i]) for i in permutations]
    print('Begin permuting channels of train data')
    for filename in tqdm(os.listdir(data_path_train)):
        org_img = torch.load(data_path_train+filename)
        clone = torch.clone(org_img)
        for i in range(len(permutations)):
            torch.save(clone[permutations[i],:,:], new_data_path + 'train/' + perm_name[i] + filename)
    print('Begin permuting channels of truth data')
    for filename in tqdm(os.listdir(data_path_truth)):
        org_img = torch.load(data_path_truth + filename)
        clone = torch.clone(org_img)
        for i in range(len(permutations)):
            torch.save(clone[permutations[i], :, :], new_data_path + 'truth/' + perm_name[i] + filename)


def kaggle_als_renamer(path, output_path, output_path_j):
    name_to_id = dict()
    id_to_numbers = dict()
    id_to_name = dict()
    i = 0
    for org_name in tqdm(os.listdir(path)):
        idx = org_name.rfind('_')
        buffer = org_name[:idx]
        idx2 = buffer.rfind('_')
        name = buffer[:idx2]
        number = buffer[idx2+1:]
        if name_to_id.get(name, -1) < 0:
            name_to_id[name] = i
            id_to_name[i] = name
            i += 1
        id_to_numbers.setdefault(name_to_id[name], [])
        id_to_numbers[name_to_id[name]].append(number)
        img = imread(path+org_name)
        img_t = torch.Tensor(img)
        torch.save(img_t, output_path+str(name_to_id[name])+'_'+number+'.pt')
    json_dict = {'name_to_id':name_to_id,
                 'id_to_name':id_to_name,
                 'id_to_numbers':id_to_numbers}
    with open(output_path_j, "w") as outfile:
        json.dump(json_dict, outfile)
    return json_dict


# def create_synthetic_dataset_als(meta_data, path, output_path, max_images=20000, cells_per_chan=4, image_shape=(3, 1024, 1024)):

def create_synthetic_dataset_als(path, max_images=20000, cells_per_chan=6, image_shape=(1024, 1024, 3)):
    def generate_x_y(points):
        x,y = random.randint(0,900), random.randint(0,900)
        if not points:
            return x,y
        diff = lambda p1, p2: (p1-p2 < -192) or (p1-p2 > 192)
        xlist = [x]*len(points)
        ylist = [y]*len(points)
        if not all(map(diff, points, xlist)) or not all(map(diff, points, ylist)):
            return generate_x_y(points)
        return x,y
    key_list = 1
    filenames = list(os.listdir(path))
    curr_chan = 0
    num_cells = 0
    num_images = 0
    while num_images < max_images:
        new = np.zeros(image_shape)
        for channel in range(image_shape[-1]):
            empty = np.zeros(image_shape[:2])
            points = []
            for i in range(cells_per_chan):
                x,y = generate_x_y(points)
                name = random.sample(filenames, 1)[0]
                img = imread(path+name)

                empty[x:min(x+256,1023), y:min(y+256,1023)] = normalize_array(img[:min(1023-x, 256),:min(1023-y, 256)])
                chan = np.maximum(new[:,:,channel],empty)
                new[:,:,channel] = chan
        break
    plt.imshow(new)
    plt.show()




if __name__ == '__main__':
    # loop through ground truth samples, compute mean, make the grayscale3chan, normalize g.t and normalize grayscale3chan
    # data_path = '/nobackup/users/vinhle/data/z_max_slices/'
    train = '/nobackup/users/vinhle/data/256/train/'
    truth = '/nobackup/users/vinhle/data/256/truth/'
    output = '/nobackup/users/vinhle/data/256_channel_permutation/'
    # output_size = (3, 256, 256)
    # preprocess_main(data_path, output_path, output_size, thresh_scale = .5)
    # test_images(output_path, data_path, 'F030.pt')
    # channel_transform(train, truth, output, 3)
    # test_images(output,'012F030.pt')
    # test_images(output, '102F030.pt')
    path = '/nobackup/users/vinhle/data/hpa_data/renamed/'
    path = 'data/single_cell/train_tiles/nuclear_bodies/'
    # kaggle_als_renamer(path, 'data/single_cell/train_tiles/new_nuclear/', 'data/single_cell/train_tiles/metadata.json')
    kaggle_als_renamer('/nobackup/users/vinhle/data/hpa_data/nuclear_bodies/',
                       '/nobackup/users/vinhle/data/hpa_data/nuclear_bodies_renamed/', '/nobackup/users/vinhle/data/hpa_data/metadata.json')

    # create_synthetic_dataset_als('data/single_cell/train_tiles/nuclear_bodies/')
    # mask = fixed_blobs(output_size[::-1], int(output_size[1] / 16), int(output_size[1] / 16),
    #                    int(output_size[1] / 32), int(output_size[1] / 64), int(output_size[1] / 64), boolean=True)
    # code to test connected components
    # model = 'resnet50'
    # dataset = 'swsl'
    # preprocess_fn = create_pretrained(model, dataset, preprocess_only=True)
    # try:
    #     os.mkdir(output_path + 'train_pixel')
    # except:
    #     pass






