from os.path import splitext, join, basename
import numpy as np
from torch import from_numpy
import torch

class WriteTensorToDisc(object):
    def __init__(self, write_loc, path_annotations):
        self.write_loc = write_loc          
        self.annotations = path_annotations

    def __call__(self, sample):
        
        name, ext = splitext(basename(sample["name"]))
        save_file = join("data", sample["label"], name + "_preprocessed" + ext)

        video_tensor = np.flip(sample["data"], axis=3).copy()
        
        filesystem_name = join(self.write_loc, sample["label"], name + "_preprocessed" + ext)

        torch.save(video_tensor, filesystem_name)

        with open(self.annotations, "a") as f:
            f.write("{},{},{}\n".format(save_file, sample["label"], video_tensor.shape))
        
        return sample

