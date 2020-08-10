import torch
import numpy as np

class ToRTPoseInput(object):

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        
        data = sample['data']
        data = np.expand_dims(data, self.dim)

        sample['data'] = torch.from_numpy(data).float()

        return sample