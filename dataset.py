import os
import torch
import numpy as np
from torchvision import transforms
from utils import *
import numpy.random as npr
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import multiprocessing as mp

from sklearn.model_selection import train_test_split


class ProCodes(torch.utils.data.Dataset):

    def __init__(self, paths, transform=None):
        """
        :param path: path to the folder containing all the files
        :param transform: optional transforms from the imgaug python package
        :return: None
        """
        print("Initializing data conversion and storage...")
        assert paths is not None, \
            "Path to data folder is required"
        super(ProCodes).__init__()
        self.paths = paths
        self.transforms = transform
        print("Done")

    def __getitem__(self, idx):
        """
        :param train_set: boolean where True means use the train set, False means test set, and None means entire set
        :param idx: index to index into set of data
        :param transform: boolean to decide to get transformed data or not
        :return: image as a tensor
        """
        inp_path, mask_path = self.paths[0][idx], self.paths[1][idx]
        i = inp_path.find("train")
        # zero_mask_path = f'{inp_path[:i]}/classification_mask/{inp_path[inp_path.rfind("/") + 1:]}'
        # zero_mask = torch.load(zero_mask_path)
        inp, mask = torch.load(inp_path), torch.load(mask_path)
        if self.transforms:
            inp = self.transforms(inp)
        inp = torch.Tensor(inp)
        mask = torch.Tensor(mask)
        size = inp.size()[1]
        inp = inp.view((3, size, size))
        mask = mask.view((3, size, size))
        # return inp, mask, zero_mask
        return inp, mask

    def __len__(self):
        return len(self.paths[0])


class ProCodesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 1, test_size: float = .3, transform = None):
        '''
        :param data_dir:
        :param batch_size:
        :param test_size:
         input path aka blob path and then the mask path which in this case is slices
        '''
        super().__init__()
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomVerticalFlip(0.5),
        #     transforms.RandomHorizontalFlip(0.5),
        #     transforms.ToTensor()])
        self.transform = transform
        self.test_size = test_size
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.items = [[directory + filename for filename in os.listdir(directory)] for directory in self.data_dir]
        assert data_dir is not None, \
            "Path to data folder is required"
        self.setup()

    def setup(self, stage: str = None):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.items[0], self.items[1], test_size=self.test_size)
        # want val size == test size
        self.xval, self.xtest, self.yval, self.ytest = train_test_split(self.xtest, self.ytest, test_size=0.5)
        print("VAL SET EXAMPLES: ", self.xval)
        print("TEST SET EXAMPLES: ", self.xtest)
        if stage in (None, "test"):
            self.test = ProCodes([self.xtest, self.ytest])
        if stage in (None, "fit"):
            self.train = ProCodes([self.xtrain, self.ytrain], transform=self.transform)
            self.val = ProCodes([self.xval, self.yval], transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def validation_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    pass




