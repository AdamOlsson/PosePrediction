
from model.PoseModel import PoseModel

from torchvision.datasets.video_utils import VideoClips
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

# misc
import pandas as pd # easy load of csv
from util.setup_directories import setup

def main(input_vid, output_dir):
    
    device = "cuda"
    config = load_config("config.json")

    # annotations_in = join(input_dir, "annotations.csv")
    # annotations_out = join(output_dir, "annotations.csv")

    # annotations = pd.read_csv(annotations_in)
    # labels = list(annotations.iloc[:,1])
    #labels = [(annotations.iloc[0,1])] # debug

    subset_size = 16
    # video_names = [join(input_dir, annotations.iloc[i,0]) for i in range(len(annotations))]
    #video_names = [join(input_dir, annotations.iloc[0,0])] # debug
    videoclips = VideoClips([input_vid], clip_length_in_frames=subset_size, frames_between_clips=subset_size)

    transformers = [
        FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]),
        RTPosePreprocessing(),
        ToRTPoseInput(0),
        ]

    composed = Compose(transformers)
    
    model = PoseModel()
    model = model.to(device)
    model.load_state_dict(torch.load("model/weights/vgg19.pth", map_location=torch.device(device)))
    
    counter, sample = {}, {}
    vframes = None
    # subpart_count = {}
    for i in range(len(videoclips)):
        vframes,_,info, video_idx = videoclips.get_clip(i)

        # label = labels[video_idx]

        # video_name = basename(video_names[video_idx])
        # video_dir = join(output_dir, "data", label, video_name)

        # if not exists(video_dir):
            # mkdir(video_dir)
        
        # if str(video_idx) in counter:
            # counter[str(video_idx)] += 1
        # else: 
            # counter[str(video_idx)] = 0


        sample["data"] = vframes.numpy()
        sample["type"] = "video"
        sample = composed(sample)
        vframes = sample["data"]

        # attempt to free some memory
        del sample
        sample = {}

        with torch.no_grad():
            (branch1, branch2), _ = model(vframes.to(device))

        del vframes
        vframes = None

        paf = branch1.data.cpu().numpy().transpose(0, 2, 3, 1)
        heatmap = branch2.data.cpu().numpy().transpose(0, 2, 3, 1)

        # Construct humans on every frame
        no_frames = len(paf[:]) # == len(heatmap[:])
        frames = []
        for frame in range(no_frames):
            humans = paf_to_pose_cpp(heatmap[frame], paf[frame], config)
            frames.append(humans)

        # attempt to free some memory
        del paf
        del heatmap
        paf = []
        heatmap = []
        
        # metadata = {
            # "filename": video_names[video_idx],
            # "body_part_translation":body_part_translation,
            # "body_construction":body_part_construction,
            # "label":labels[video_idx],
            # "video_properties":info,
            # "subpart":counter[str(video_idx)]
        # }

        save_name = join(output_dir, str(video_idx) + ".json")

        save_humans(save_name, frames, {})

        # dir_name = join("data", label, video_name)
        # if dir_name in subpart_count:
            # (l, n, ss) = subpart_count[dir_name]
            # subpart_count[dir_name] = (l, n+1, ss)
        # else:
            # subpart_count[dir_name] = (label, 0, subset_size)
        
        print(save_name)        
        # with open(annotations_out, "a") as f:
        # for key, (l,n,ss) in subpart_count.items():
            # f.write("{},{},{},{}\n".format(key, l, n, ss))

    # NOTE: At this point, partial graphs are generated at output_dir. Now to merge them



def parse_args(argv):
    try:
        opts, _ = getopt.getopt(argv, 'hi:', ['input_vid='])
    except getopt.GetoptError:
       sys.exit(2)
    input_vid = ""
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-i", "--input"):
            input_vid = arg
    return input_vid

if __name__ == "__main__":
    input_dir = parse_args(sys.argv[1:])
    # output_dir = setup(input_dir, output_dir, "partial_graphs", "filename,label,subsets,subset_len")
    output_dir = "_tmp_vid"
    mkdir(output_dir)
    main(input_dir, output_dir)