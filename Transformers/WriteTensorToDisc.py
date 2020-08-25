from os.path import splitext, join, basename
import numpy as np
from torch import from_numpy
import torch

class WriteTensorToDisc(object):
    def __init__(self, write_loc, path_annotations):
        self.write_loc = write_loc          
        self.annotations = path_annotations

    def __call__(self, sample):
        
        name, _ = splitext(basename(sample["name"]))
        base = name + ".pt"
        save_file = join("data", sample["label"], base)
        
        filesystem_name = join(self.write_loc, sample["label"], base)

        torch.save(sample["data"], filesystem_name)

        with open(self.annotations, "a") as f:
            f.write("{},{},{}\n".format(save_file, sample["label"], "({},{},{},{})".format(
                sample["data"].shape[0],
                sample["data"].shape[1],
                sample["data"].shape[2],
                sample["data"].shape[3]
            )))
        
        return sample

