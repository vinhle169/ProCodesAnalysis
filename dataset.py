import os
import torch
import numpy as np
from torchvision import transforms
from utils import *
import numpy.random as npr
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class ProCodes(torch.utils.data.Dataset):

    def __init__(self, paths, transforms=None, data_type='unet'):
        """
        :param path: path to the folder containing all the files
        :param transforms: optional transforms from the imgaug python package
        :return: None
        """
        print("Initializing data conversion and storage...")
        assert paths is not None, \
            "Path to data folder is required"
        super(ProCodes).__init__()
        self.paths = paths
        self.data_type = data_type
        self.transforms = transforms
        print("Done")

    def __getitem__(self, idx):
        """
        :param train_set: boolean where True means use the train set, False means test set, and None means entire set
        :param idx: index to index into set of data
        :param transform: boolean to decide to get transformed data or not
        :return: image as a tensor
        """
        # print("Grabbing file and converting to tensor...")
        if self.data_type == 'cnet':
            idx_path = self.paths[idx]
            image = torch.load(idx_path)
            image, label, _ = random_channel(image)
            # print("Performing image augmentation...")
            if self.transforms:
                image = self.transforms(image)
            image = torch.Tensor(image)
            label = torch.Tensor(label)
            image = image.view((1, 2048, 2048))
            return image, label
        elif self.data_type == 'unet':
            inp_path, mask_path = self.paths[0][idx], self.paths[1][idx]
            inp, mask = torch.load(inp_path), torch.load(mask_path)
            if self.transforms:
                inp = self.transforms(inp)
            inp = torch.Tensor(inp)
            mask = torch.Tensor(mask)
            inp = inp.view((4, 2048, 2048))
            mask = mask.view((3, 2048, 2048))
            return inp, mask

    def __len__(self):
        if self.data_type == 'cnet':
            return len(self.paths)
        else:
            return len(self.paths[0])


class ProCodesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 1, test_size: float = .3, data_type: str = 'unet'):
        '''
        :param data_dir:
        :param batch_size:
        :param test_size:
        :param data_type: if data type is unet make sure data_dir as an input is a list with
         input path aka blob path and then the mask path which in this case is slices
        '''
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()])

        self.data_type = data_type
        self.test_size = test_size
        self.data_dir = data_dir
        self.batch_size = batch_size
        if self.data_type == 'unet':
            self.items = [[directory + filename for filename in os.listdir(directory)] for directory in self.data_dir]
        else:
            self.items = [self.data_dir + filename for filename in os.listdir(self.data_dir)]
        assert data_dir is not None, \
            "Path to data folder is required"
        self.setup('fit')

    def setup(self, stage: str = None):
        if self.data_type == 'unet':
            self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.items[0], self.items[1], test_size=self.test_size)
            if stage in (None, "test"):
                self.test = ProCodes([self.xtest, self.ytest], data_type=self.data_type)
            if stage in (None, "fit"):
                self.train = ProCodes([self.xtrain, self.ytrain], transforms=self.transform, data_type=self.data_type)
        else:
            self.train, self.test = train_test_split(self.items, test_size=self.test_size)
            if stage in (None, "test"):
                self.test = ProCodes(self.test)
            if stage in (None, "fit"):
                self.train = ProCodes(self.train, transforms=self.transform, data_type=self.data_type)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    pass




