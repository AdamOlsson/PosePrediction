import numpy as np

class RTPosePreprocessing(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        data = sample['data']
        
        data = data.astype(np.float32)
        data = data / 256 - 0.5
        data = data.transpose((2, 0, 1)).astype(np.float32)

        sample['data'] = data

        return sample