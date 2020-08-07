import numpy as np

class ExpandDim(object):

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        
        image, label = sample['image'], sample['label']

        return {'image':np.expand_dims(image, self.dim), 'label':label}