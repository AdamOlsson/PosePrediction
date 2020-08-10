import pandas as pd # easy load of csv
import cv2
import numpy as np

import torch, torchvision
from torch.utils.data import Dataset

# debugging
import cv2

class VideoDataset(Dataset):
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

        vid_name = self._path_root + self.annotations.iloc[idx,0]

        vframes, _, _ = torchvision.io.read_video(vid_name, pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames
        label = self.annotations.iloc[idx,1]

        vframes = np.flip(vframes.numpy(), axis=3)

        sample = {'data':vframes, 'label':label, 'copy':np.copy(vframes), 'name':vid_name, 'type':'video'}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
