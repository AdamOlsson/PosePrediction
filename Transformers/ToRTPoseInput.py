import torch
import numpy as np

class ToRTPoseInput(object):

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, self.dim)

        return {'image':torch.from_numpy(image).float(), 'label':label}