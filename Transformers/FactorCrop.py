import numpy as np
import cv2

class FactorCrop(object):

    def __init__(self, factor, dest_size=368):
        self.factor = factor
        self.dest_size = dest_size

    def __call__(self, sample):
        image = sample['image']
        
        min_dimension = np.min(image.shape[:2])
        
        scale_factor = float(self.dest_size) / min_dimension

        image_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        h, w, c = image_resized.shape

        # TODO: find out the purpose of this
        h_new = int(np.ceil( h / self.factor))*self.factor
        w_new = int(np.ceil( w / self.factor))*self.factor

        image_cropped = np.zeros([h_new, w_new, c], dtype=image.dtype)
        image_cropped[:h, :w, :] = image_resized

        sample['image'] = image_cropped

        return sample