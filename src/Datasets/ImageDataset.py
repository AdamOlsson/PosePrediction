import pandas as pd # easy load of csv
from skimage import io
import numpy as np

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, path_csv, path_root, transform=None):
        self._path_csv  = path_csv
        self._path_root = path_root
        self.annotations = pd.read_csv(path_csv)
        self.transform = transform


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self._path_root + self.annotations.iloc[idx,0]

        img = io.imread(img_name)
        label = self.annotations.iloc[idx,1]

        sample = {'image':img, 'label':label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
