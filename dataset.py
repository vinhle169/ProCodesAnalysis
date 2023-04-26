import os
import json
import torch
import numpy as np
from utils import *
import numpy.random as npr
import multiprocessing as mp
import datetime
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision import transforms


from sklearn.model_selection import train_test_split


class ProCodes(torch.utils.data.Dataset):

    def __init__(self, paths, image_size, transform=None, in_memory=False):
        """
        :param path: path to the folder containing all the files
        :param transform: optional transforms from the imgaug python package
        :return: None
        """
        print("Initializing data conversion and storage...")
        assert paths is not None, \
            "Path to data folder is required"
        super(ProCodes).__init__()
        # so we can grab a any certain file, knowing the filename
        self.name_to_idx = {paths[0][i][paths[0][i].rfind('/') + 1:]: i for i in range(len(paths[0]))}
        if in_memory:
            print("Loading in Training Input Images")
            path_x = torch.stack([torch.load(i) for i in paths[0]])
            print("Loading in Training Target Output Images")
            path_y = torch.stack([torch.load(i) for i in paths[1]])
            print(f"{path_x.shape[0]} samples loaded")
            self.paths = [path_x,path_y]
        else:
            self.paths = paths
        self.in_memory = in_memory
        self.image_size = image_size
        self.transforms = transform
        print("Done")

    def __getitem__(self, idx):
        """
        :param idx: index to index into set of data
        :param transform: boolean to decide to get transformed data or not
        :return: image as a tensor
        """
        inp, mask = self.paths[0][idx], self.paths[1][idx]
        # i = inp_path.find("train")
        # zero_mask_path = f'{inp_path[:i]}/classification_mask/{inp_path[inp_path.rfind("/") + 1:]}'
        # zero_mask = torch.load(zero_mask_path)
        if not self.in_memory:
            inp, mask = torch.load(inp), torch.load(mask)
        if self.transforms:
            inp = self.transforms(inp)
        # if self.image_size:
        #     padder = transforms.Pad([0,0,self.image_size[-1]-inp.shape[-1], self.image_size[-2]-inp.shape[-2]], padding_mode='edge')
        #     inp = padder(inp)
        #     mask = padder(mask)
        # inp = inp.clone().detach().type(torch.float16)
        # mask = mask.clone().detach().type(torch.float16)
        # size = inp.size()
        # inp = inp.view((4, size, size))
        # mask = mask.view((4, size, size))
        # return inp, mask, zero_mask
        return inp, mask

    def __len__(self):
        return len(self.paths[0])

    def get_item(self, name):
        idx = self.name_to_idx[name]
        return self[idx]



class ProCodesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 1, test_size: float = .3, transform = None, stage=None,
                 image_size=None, in_memory=False, metadata=False, load_metadata=None):
        '''
        :param data_dir: size 2 list of input directory and target output directory
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
        self.image_size = image_size
        self.items = [[directory + filename for filename in os.listdir(directory)] for directory in self.data_dir]
        assert data_dir is not None, \
            "Path to data folder is required"
        if in_memory: print("WARNING: ONLY ATTEMPT LOADING IN MEMORY IF THERE IS ENOUGH SPACE")
        self.setup(stage=stage,in_memory=in_memory,metadata=metadata, load_metadata=load_metadata)

    def setup(self, stage: str = None, in_memory: bool = False, metadata: bool = False, load_metadata: str = None):
        if load_metadata:
            print(f'Using {load_metadata} file')
            with open(load_metadata, 'r') as f:
                file_dict = json.load(f)
            self.xtrain = file_dict['train']
            self.xval = file_dict['val']
            self.xtest = file_dict['test']
            # hacky atm because old metadata files arent saved the new way yet
            self.ytrain = [i[:i.rfind('train')] + i[i.rfind('train'):] for i in file_dict['train']]
            self.yval = [i[:i.rfind('truth')] + i[i.rfind('truth'):] for i in file_dict['val']]
            self.ytest = [i[:i.rfind('truth')] + i[i.rfind('truth'):] for i in file_dict['test']]

        elif len(self.items[0]) == 1 or not self.test_size:
            self.xtrain = self.items[0]
            self.xval = self.items[0]
            self.ytrain = self.items[1]
            self.yval = self.items[1]
            self.xtest = self.items[0]
            self.ytest = self.items[1]
        else:
            self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.items[0], self.items[1], test_size=self.test_size)
            # want val size == test size
            self.xval, self.xtest, self.yval, self.ytest = train_test_split(self.xtest, self.ytest, test_size=0.5)

            if metadata:
                file_dict = {}
                # file_dict['paths'] = {'train':self.data_dir[0], 'truth':self.data_dir[1]}
                file_dict['train'] = self.xtrain
                file_dict['train'] = self.xtrain
                file_dict['val'] = self.xval
                file_dict['test'] = self.xtest
                idx = self.data_dir[0][:-1].rfind('/')
                save_path = self.data_dir[0][:idx+1]
                date = datetime.date.today().strftime('%y-%m-%d')
                filename = save_path + f'metadata_{date}.json'
                print(filename)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(file_dict, f, ensure_ascii=False, indent=4)
                print("Metadata file created and saved.")

            else:
                print("VAL SET EXAMPLES: ", self.xval[0:min(len(self.xval),5)])
                print("TEST SET EXAMPLES: ", self.xtest[0:min(len(self.xval),5)])
        if stage in (None, "test"):
            self.test = ProCodes([self.xtest, self.ytest], image_size=self.image_size)
        if stage in (None, "fit"):
            self.train = ProCodes([self.xtrain, self.ytrain], image_size=self.image_size, transform=self.transform, in_memory=in_memory)
            self.val = ProCodes([self.xval, self.yval], image_size=self.image_size, transform=self.transform, in_memory=in_memory)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def validation_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    data_path = ['/nobackup/users/vinhle/data/procodes_data/unet_train/train/','/nobackup/users/vinhle/data/procodes_data/unet_train/truth/']
    z = ProCodesDataModule(data_dir=data_path, batch_size=4,
                           test_size=.30, image_size=(256, 256), in_memory=False, metadata=True)
    train_loader = z.train_dataloader()




