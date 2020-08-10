import numpy as np
import cv2

class FactorCrop(object):

    def __init__(self, factor, dest_size=368):
        self.factor = factor
        self.dest_size = dest_size

    def __call__(self, sample):
        data = sample['data']
        
        min_dimension = np.min(data.shape[:2])
        
        scale_factor = float(self.dest_size) / min_dimension

        data_resized = cv2.resize(data, None, fx=scale_factor, fy=scale_factor)

        h, w, c = data_resized.shape

        # TODO: find out the purpose of this
        h_new = int(np.ceil( h / self.factor))*self.factor
        w_new = int(np.ceil( w / self.factor))*self.factor

        data_cropped = np.zeros([h_new, w_new, c], dtype=data.dtype)
        data_cropped[:h, :w, :] = data_resized

        sample['data'] = data_cropped

        return sample