import pandas as pd # easy load of csv
import cv2
import numpy as np

import torch, torchvision
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, path_csv, path_root, transform=None, load_copy=False, frame_skip=0):
        self._path_csv  = path_csv
        self._path_root = path_root
        self.annotations = pd.read_csv(path_csv)
        self.transform = transform
        self.load_copy = load_copy
        self.frame_skip = frame_skip


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vid_name = self._path_root + self.annotations.iloc[idx,0]

        vframes, _, _ = torchvision.io.read_video(vid_name, start_pts=0, end_pts=1, pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames
        label = self.annotations.iloc[idx,1]

        vframes = np.flip(vframes.numpy(), axis=3)

        if frame_skip != 0:
            no_frames = vframes.shpe[0]
            selected_frames = np.linspace(0, no_frames, num=no_frames/self.frame_skip)
            vframes = v_frames[selected_frames]

        sample = {'data':vframes, 'label':label, 'name':vid_name, 'type':'video'}

        if self.load_copy:
            sample['copy'] = np.copy(vframes)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
