import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from cellpose import models
from imageio import volread
import matplotlib.pyplot as plt
from skimage.morphology import closing

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


def main(directory, paths, train_directory):
    model = models.Cellpose(gpu=True, model_type='nuclei')

    for p in paths:
        path = directory+p+'/mp_score_max/'
        for filename in tqdm(os.listdir(path)):
            # percentle clip before resize then use cellpose
            img = volread(path+filename)
            truth, masks = [], []
            # cycle through to normalize, resize, close (see skimage morphology closing for more details)
            # and create our masks
            for channel in range(img.shape[0]):
                img_i = img[channel].copy()
                img_i = normalize_array(img_i, min_max=False)
                img_i = cv2.resize(img_i, dsize=(256, 256), interpolation=cv2.INTER_AREA)
                img_i = closing(img_i, footprint=np.ones((2, 2)))
                mask, _, _, _ = model.eval(img_i, diameter=6, flow_threshold=0.5, cellprob_threshold=0.05)
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
            for i in range(truth.shape[0]):
                train[i,:,:] = grayscale
            nuclei = masks > 0
            train[nuclei] = truth[nuclei]
            # convert to torch tensor for future training use
            truth, masks, train = torch.from_numpy(truth), torch.from_numpy(masks), torch.from_numpy(train)
            torch.save(truth, train_directory + f'truth/{p + "_" + filename[:-4]}.pt')
            torch.save(train, train_directory + f'train/{p + "_" + filename[:-4]}.pt')
            torch.save(masks, train_directory + f'masks/{p + "_" + filename[:-4]}.pt')
        print(f'finished {path}')




if __name__ == '__main__':
    directory = '/nobackup/users/vinhle/data/procodes_data/unet_train/original_data/coverslip'
    paths = ['1','2','3','4','5']
    train_directory = '/nobackup/users/vinhle/data/procodes_data/unet_train/'
    main(directory, paths, train_directory)