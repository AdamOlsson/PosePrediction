## 
## Due to memory constrainsts, preprocessing has to be done in steps. The script 
## assumes that there is a file called annotations.csv in the input directory.
## 
## The outout directory will have the following format:
## 
## <output dir>
##     annotations.csv
##     data/
##         <label1>
##             <video name>.json
##             ...
##         <label2>
##             <video name>.json
##             ...
## 
## The annotations file will have the follow format:
## 
## # path,label,subsets,subset_len
## <path1>,<label>,...
## <path2>,<label>,...
## ...
## 
## The output from this file is the path to the preprocessed datas root directory to allow for Linux
## based piping.
## 
## Usage:
## python factorcrop.py --data_dir <path to data root> --out_dir <path to output dir>

from torchvision.datasets.video_utils import VideoClips
from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput

from torchvision.transforms import Compose

from util.load_config import load_config

# native
import sys, getopt
from shutil import rmtree
from os.path import exists, join, basename, splitext
from os import makedirs, mkdir

import torch

# misc
import pandas as pd # easy load of csv

def main(input_dir, output_dir):
    
    config = load_config("config.json")

    annotations_in = join(input_dir, "annotations.csv")
    annotations_out = join(output_dir, "annotations.csv")

    annotations = pd.read_csv(annotations_in)
    labels = list(annotations.iloc[:,1])

    subset_size = 32
    video_names = [join(input_dir, annotations.iloc[i,0]) for i in range(len(annotations))]
    videoclips = VideoClips(video_names, clip_length_in_frames=subset_size, frames_between_clips=subset_size)

    transformers = [
        FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]),
        RTPosePreprocessing(),
        ToRTPoseInput(0),
        ]

    composed = Compose(transformers)
    counter, sample = {}, {}
    vframes = None
    old_video_idx = 0
    for i in range(len(videoclips)):
        vframes,_,_, video_idx = videoclips.get_clip(i)

        label = labels[video_idx]

        video_name = basename(video_names[video_idx])
        video_dir = join(output_dir, "data", label, video_name)

        if not exists(video_dir):
            mkdir(video_dir)
        
        if str(video_idx) in counter:
            counter[str(video_idx)] += 1
        else: 
            counter[str(video_idx)] = 0

        save_name = join(video_dir, str(counter[str(video_idx)]) + ".pt")

        sample["data"] = vframes.numpy()
        sample["type"] = "video"
        sample = composed(sample)
        vframes = sample["data"]

        # attempt to free some memory
        del sample
        sample = {}

        torch.save(vframes, save_name)

        if old_video_idx < video_idx or i == len(videoclips)-1:
            print("Done processing {}".format(video_names[old_video_idx]))
            with open(annotations_out, "a") as f:
                f.write("{},{},{},{}\n".format(join("data", label, video_name), label, counter[str(old_video_idx)]+1, subset_size))
        
        print(save_name)
        
        old_video_idx = video_idx



def setup(input_dir, output_dir):
    ## Setup structure of the output directory.
    annotations_path = join(input_dir, "annotations.csv")
    annotations = pd.read_csv(annotations_path)
    
    labels = annotations.iloc[:,1]
    unique_labels = list(set(labels))

    data_out_root = join(output_dir, "preprocessed")

    # delete data, start from clean slate
    if exists(data_out_root):
        rmtree(data_out_root)
    
    data_out = join(data_out_root, "data")
    makedirs(data_out) # recursive create of root and data dir

    for label in unique_labels:
        path = join(data_out, label)
        mkdir(path)

    annotations_out = join(data_out_root, "annotations.csv") 
    with open(annotations_out,'w+') as f:
        f.write("# filename,label,subsets,subset_len\n") # Header

    return data_out_root 



def parse_args(argv):
    try:
        opts, _ = getopt.getopt(argv, 'hi:o:', ['input_dir=', 'output_dir='])
    except getopt.GetoptError:
       sys.exit(2)
    input_dir = ""
    output_dir = ""
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-i", "--input"):
            input_dir = arg
        elif opt in ("-o", "--output"):
            output_dir = arg
    return input_dir, output_dir 


if __name__ == "__main__":
    # NOTE: Paths needs to relative, torchvision does not handle absolute paths
    input_dir, output_dir = parse_args(sys.argv[1:])
    output_dir = setup(input_dir, output_dir)
    main(input_dir, output_dir)