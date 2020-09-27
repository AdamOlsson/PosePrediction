
from model.PoseModel import PoseModel

from torchvision.datasets.video_utils import VideoClips
from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput

from torchvision.transforms import Compose

from util.load_config import load_config
from paf.util import save_humans, load_humans
from paf.common import draw_humans
from paf.paf_to_pose import paf_to_pose_cpp
from paf.body_part_construction import body_part_construction, body_part_translation

# native
import sys, getopt, json, shutil
from shutil import rmtree
from os.path import join
from os import mkdir, listdir

import torch, torchvision

# misc
import numpy as np
from util.setup_directories import setup
from merge_partial_graphs import merge_dicts, load_partial_jsons


def main(input_vid, output_dir):
    
    device = "cuda"
    config = load_config("config.json")

    subset_size = 15

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
    
    sample = {}
    vframes = None
    for i in range(len(videoclips)):
        vframes,_,info, _ = videoclips.get_clip(i)

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

        metadata = {
            "filename": "",
            "body_part_translation":body_part_translation,
            "body_construction":body_part_construction,
            "label":0,
            "video_properties":info,
            "subpart":str(i)
        }


        save_name = join(output_dir, str(i) + ".json")

        save_humans(save_name, frames, metadata)
        
        print("Created subgraph:", save_name)        
   
    print("Merging subgraphs...")
    partial_graphs_names = listdir(output_dir)

    # merging partial dicts
    dicts = load_partial_jsons(output_dir, len(partial_graphs_names)-1)
    merged_dicts = merge_dicts(dicts)

    filename = join(output_dir, "merged" + ".json")
    with open(filename, 'w') as f:
        f.write(json.dumps(merged_dicts, indent=4, sort_keys=True))

    print("Rendering video...")
    # render video
    graph_data = load_humans(filename)
    metadata, humans = graph_data["metadata"], graph_data["frames"]
    vframes, _, _ = torchvision.io.read_video(input_vid, pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames
    vframes = np.flip(vframes.numpy(), axis=3)
    vframes = vframes[:len(humans)]
    
    for frame_idx in range(len(humans)):
        vframes[frame_idx] = draw_humans(np.float32(vframes[frame_idx]), humans[frame_idx])
    
    vframes = np.flip(vframes, axis=3).copy()
    save_path = join("skeleton.mp4")
    fps = 30
    torchvision.io.write_video(save_path, torch.from_numpy(vframes), fps)

    print("Video saved to {}".format(save_path))

    print("Removing tmp dir _tmp_vid.")
    shutil.rmtree(output_dir)



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