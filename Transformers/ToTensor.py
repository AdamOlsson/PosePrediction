import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        
        data, label = sample['data'], sample['label']

        # swap color axis because
        # numpy data: H x W x C
        # torch data: C X H X W
        data = data.transpose((2, 0, 1))
        return {'data':torch.from_numpy(data), 'label':label}