import torch
import numpy as np

class ToRTPoseInput(object):

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        
        image = sample['image']
        image = np.expand_dims(image, self.dim)

        sample['image'] = torch.from_numpy(image).float()

        return sample