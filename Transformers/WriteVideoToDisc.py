from os.path import exists, join, basename, splitext
import numpy as np
from torch import from_numpy
import torchvision

class WriteVideoToDisc(object):

    def __init__(self, write_loc, path_annotations):
        self.write_loc = write_loc          # preprocessed/data     || /<label>/videoname
        self.annotations = path_annotations # preprocessed/annotations.csv

    def __call__(self, sample):
        
        metadata = sample["metadata"]

        name, ext = splitext(basename(sample["name"]))
        save_file = join(self.write_loc, sample["label"], name + "_preprocessed." + ext)

        video_tensor = np.flip(sample["data"], axis=3)

        #torchvision.io.write_video(save_file, from_numpy(video_tensor), int(metadata["video_properties"]["video_fps"]))
        
        with open(self.annotations, "a") as f:
            f.write("{},{},{}\n".format(save_file, sample["label"], video_tensor.shape[0]))
        
        return sample

