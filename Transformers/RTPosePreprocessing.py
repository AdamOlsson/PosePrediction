import numpy as np

class RTPosePreprocessing(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        
        image = image.astype(np.float32)
        image = image / 256 - 0.5
        image = image.transpose((2, 0, 1)).astype(np.float32)

        sample['image'] = image

        return sample