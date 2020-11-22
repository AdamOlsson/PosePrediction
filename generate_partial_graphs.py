## 
## Due to memory constrainsts, the video has to divided into subclips and then have
## graphs generate. After this script is done, there exists one additional script
## that merges all the partial graphs of a video to one single graph. This script
## assumes that there is a file called annotations.csv in the input directory.
## 
## The outout directory will have the following format:
## 
## <output dir>
##     annotations.csv
##     data/
##         <label1>
##             video name/
##                  0.json
##                  1.json
##                  ...
##         <label2>
##             video name/
##                  0.json
##                  1.json
##                  ...
##         ...
## 
## The annotations file will have the follow format:
## 
## # path,label,subsets,subset_len
## <path1>,<label>,...
## <path2>,<label>,...
## ...
## 
## 
## Usage:
## NOTE: Paths needs to relative
## python generate_graphs.py -i <path to data root> -o <path to output dir>

from model.PoseModel import PoseModel

from torchvision.datasets.video_utils import VideoClips
from Datasets.VideoClipsWrapper import VideoClipsWrapper
from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput

from torchvision.transforms import Compose

from util.load_config import load_config
from paf.util import save_humans
from paf.paf_to_pose import paf_to_pose_cpp
from paf.body_part_construction import body_part_construction, body_part_translation


# native
import sys, getopt
from shutil import rmtree
from os.path import exists, join, basename, splitext
from os import makedirs, mkdir

import torch
from torch.utils.data import DataLoader

# misc
import pandas as pd # easy load of csv
from util.setup_directories import setup

def main(input_dir, output_dir):
    
    device = "cuda"
    config = load_config("config.json")

    annotations_in = join(input_dir, "annotations.csv")
    annotations_out = join(output_dir, "annotations.csv")

    annotations = pd.read_csv(annotations_in)
    labels = list(annotations.iloc[:,1])

    subset_size = 15
    video_names = [join(input_dir, annotations.iloc[i,0]) for i in range(len(annotations))]
    transformers = [
        FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]),
        RTPosePreprocessing(),
        ToRTPoseInput(0),
        ]

    composed = Compose(transformers)
    videoclips = VideoClipsWrapper(video_names, clip_length_in_frames=subset_size, frames_between_clips=subset_size, transform=composed)

    loader_batch_size = 16
    loader_num_workers = 1 #(2)
    dataloader = DataLoader(videoclips, batch_size=loader_batch_size, shuffle=False, num_workers=loader_num_workers)

    model = PoseModel()
    model = model.to(device)
    model.load_state_dict(torch.load("model/weights/vgg19.pth", map_location=torch.device(device)))
    
    counter = {}
    subpart_count = {}
    old_video_idx = 0
    for batch_no, batch in enumerate(dataloader):
        vframes_batch     = batch["data"]
        info_batch        = batch["info"]
        video_idx_batch   = batch["video_idx"]

        # iterate over the batch
        for i in range(len(vframes_batch)):
            vframes     = vframes_batch[i]
            info        = {"video_fps": info_batch["video_fps"][i].item()}
            video_idx   = video_idx_batch[i]

            label = labels[video_idx]

            video_name = basename(video_names[video_idx])
            video_dir = join(output_dir, "data", label, video_name)

            # make dir that contains all partial graphs for a video
            if not exists(video_dir):
                mkdir(video_dir)

            # keep track of frame count accross multiple files
            if str(video_idx) in counter:
                counter[str(video_idx)] += 1
            else: 
                counter[str(video_idx)] = 0

            # processing
            with torch.no_grad():
                (branch1, branch2), _ = model(vframes.to(device))

            del vframes
            vframes = None

            paf = branch1.data.cpu().numpy().transpose(0, 2, 3, 1)
            heatmap = branch2.data.cpu().numpy().transpose(0, 2, 3, 1)

            # Construct humans on every frame
            no_frames = len(paf[:])
            frames = []
            for frame in range(no_frames):
                humans = paf_to_pose_cpp(heatmap[frame], paf[frame], config)
                frames.append(humans)

            # attempt to free some memory
            del paf
            del heatmap
            paf = []
            heatmap = []
        
            metadata = {
                "filename": video_names[video_idx],
                "body_part_translation":body_part_translation,
                "body_construction":body_part_construction,
                "label":labels[video_idx],
                "video_properties":info,
                "subpart":counter[str(video_idx)]
            }

            # save graph file is json format
            save_name = join(video_dir, str(counter[str(video_idx)]) + ".json")
            save_humans(save_name, frames, metadata)

            dir_name = join("data", label, video_name)
            if dir_name in subpart_count:
                (l, n, ss) = subpart_count[dir_name]
                subpart_count[dir_name] = (l, n+1, ss)
            else:
                subpart_count[dir_name] = (label, 0, subset_size)

            print(save_name)        
    
    with open(annotations_out, "a") as f:
       for key, (l,n,ss) in subpart_count.items():
           f.write("{},{},{},{}\n".format(key, l, n, ss))


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
    input_dir, output_dir = parse_args(sys.argv[1:])
    output_dir = setup(input_dir, output_dir, "partial_graphs", "filename,label,subsets,subset_len")
    main(input_dir, output_dir)