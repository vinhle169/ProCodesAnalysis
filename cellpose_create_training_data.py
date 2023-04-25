import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from cellpose import models
from imageio import volread
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import label, square, opening, closing, remove_small_objects


def normalize_array(array, min_max: bool = True):

    if min_max:
        denom = np.max(array) - np.min(array)
        # avoid divide by 0 error
        if denom == 0:
            denom = 1
        return (array - np.min(array)) / denom
    else:
        bottom, top = 0, np.percentile(array, 100)
        # avoid casting everything to 0
        if not bottom < top:
            return array
        return np.clip(array, a_min=bottom, a_max=top)


def preprocess_with_resize(img, model):
    # normalize, resize, close (see skimage morphology closing for more details)
    # percentle clip before resize then use cellpose
    img_i = normalize_array(img, min_max=False)
    img_i = cv2.resize(img_i, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    img_i = closing(img_i, footprint=np.ones((2, 2)))
    # create our masks
    mask, _, _, _ = model.eval(img_i, diameter=6, flow_threshold=0.5, cellprob_threshold=0.05)
    return img_i, mask


def preprocess_with_skimage(img):
    def get_truth_and_label(img_0):
        io = normalize_array(img_0, min_max=True)
        io = cv2.resize(io, dsize=(256 * 2, 256 * 2), interpolation=cv2.INTER_AREA)
        # things will be saved as '{row}{col}':array

        thresh = threshold_otsu(io) * 1.2
        bw = opening(io > thresh, square(3))

        # label image regions
        label_image = label(bw, connectivity=1)
        label_image = remove_small_objects(label_image > 0, min_size=20)
        return io, label_image

    truths = []
    labels = []
    for i in range(img.shape[0]):
        img_i = img[i]
        truth, label_img = get_truth_and_label(img_i)
        truth = normalize_array(truth, min_max=True)
        truths.append(truth)
        labels.append(label_img)
    truths = np.stack(truths, axis=0).astype(np.float64)
    labels = np.stack(labels, axis=0).astype(np.float64)

    return truths, labels


def main(directory, paths, train_directory, resize=False, gray_at_end=False):
    if resize:
        model = models.Cellpose(gpu=True, model_type='nuclei')

    for p in paths:
        path = directory+p+'/mp_score_max/'
        for filename in tqdm(os.listdir(path)):
            img = volread(path+filename)
            if resize:
                truth, masks = [], []
                for channel in range(img.shape[0]):
                    img_i = img[channel].copy()
                    img_i, mask = preprocess_with_resize(img_i, model)
                    # normalize to 0-1 before saving as data
                    img_i = normalize_array(img_i, min_max=True)
                    truth.append(img_i)
                    masks.append(mask)
                del img
                # need to be double so can be casted to torch
                truth = np.stack(truth, axis=0).astype(np.float64)
                masks = np.stack(masks, axis=0).astype(np.float64)
                # now create our grayscale train data
                grayscale = truth.mean(0)
                train = np.zeros_like(truth).astype(np.float64)
                nuclei = masks > 0
                if gray_at_end:
                    train[nuclei] = truth[nuclei]
                    train = torch.cat([i for i in train] + [grayscale], 0)
                else:
                    for i in range(truth.shape[0]):
                        train[i, :, :] = grayscale
                    train[nuclei] = truth[nuclei]
                # convert to torch tensor for future training use
                truth, masks, train = torch.from_numpy(truth).half(), torch.from_numpy(masks).half(), torch.from_numpy(train).half()
                torch.save(truth, train_directory + f'truth/{p + "_" + filename[:-4]}.pt')
                torch.save(train, train_directory + f'train/{p + "_" + filename[:-4]}.pt')
                torch.save(masks, train_directory + f'masks/{p + "_" + filename[:-4]}.pt')
            else:
                patches_truth, patches_train = {}, {}
                truth, mask = preprocess_with_skimage(img)
                del img
                # cut up images into patches
                for i in range(0, truth.shape[1], 256):
                    for j in range(0, truth.shape[1], 256):
                        # row then column
                        file_prefix = f'{int(i/256)}{int(j/256)}'
                        patch_tru = truth[:, i:i+256, j:j+256]
                        patch_mask = mask[:, i:i+256, j:j+256]
                        patches_truth[file_prefix] = patch_tru
                        grayscale = patch_tru.mean(0)
                        patch_tra = np.zeros_like(patch_tru).astype(np.float64)
                        nuclei = patch_mask > 0
                        if gray_at_end:
                            patch_tra[nuclei] = patch_tru[nuclei]
                            patch_tra = np.stack([i for i in patch_tra] + [grayscale], 0)
                        else:
                            for k in range(patch_tra.shape[0]):
                                patch_tra[k, :, :] = grayscale
                            patch_tra[nuclei] = patch_tru[nuclei]
                        patches_train[file_prefix] = patch_tra
                for f_prefix in list(patches_truth.keys()):
                    # convert to torch tensor for future training use
                    truth, train = torch.from_numpy(patches_truth[f_prefix]).half(), \
                                          torch.from_numpy(patches_train[f_prefix]).half()
                    torch.save(truth, train_directory + f'truth/{p + "_" + f_prefix + "_" + filename[:-4]}.pt')
                    torch.save(train, train_directory + f'train/{p + "_" + f_prefix + "_" + filename[:-4]}.pt')
        print(f'finished {path}')



if __name__ == '__main__':
    directory = '/nobackup/users/vinhle/data/procodes_data/unet_train/original_data/coverslip'
    paths = ['1', '2', '3', '4', '5']
    # paths = ['3','4','5']
    train_directory = '/nobackup/users/vinhle/data/procodes_data/unet_train/23chan/'
    main(directory, paths, train_directory, gray_at_end=True)