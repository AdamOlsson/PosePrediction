"""
This script generates graphs for human pose prediction. It assumes that the in the data directory,
there exists a file called "annotations.csv" that contains paths and labels to all videos in the dataset.

The output directory will have the following structure:

<output dir>
    annotations.csv
    data/
        <label1>
            <video name>.json
            ...
        <label2>
            <video name>.json
            ...

The annotations file will have the follow format:

# path, label
<path1>,<label>
<path2>,<label>
...


Usage:
python generate_graphs.py --data_dir <path to data root> --out_dir <path to output dir>
"""
# torch & torchvision
import torch
from torchvision.transforms import Compose

# model & datasets
from model.PoseModel import PoseModel
from Datasets.VideoDataset import VideoDataset
from Datasets.VideoPredictor import VideoPredictor

# Transformers
from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput

# util
from util.load_config import load_config
from paf.paf_to_pose import paf_to_pose_cpp
from paf.body_part_construction import body_part_construction, body_part_translation
from paf.util import save_humans

# native
import sys, getopt
from shutil import rmtree
from os.path import exists, join, basename, splitext
from os import makedirs, mkdir

# misc
import pandas as pd # easy load of csv

def generate_graphs(input_dir, output_dir, device="cpu"):
    def output_handler(outputs):
        """ 
        When splitting the video into batches the output from the VideoPredictor
        is on the form [((branch1, branch2),loss),....]. This method transforms said
        output to [branch1, branch2]. We do not care about loss.
        """
        branch1, branch2 = [], []
        for ((b1,b2),_) in outputs:
            branch1.append(b1)
            branch2.append(b2)

        return torch.cat(branch1, 0), torch.cat(branch2, 0)

    print("\n############   Starting graph generation    ############\n")

    config = load_config("config.json")

    annotations_file_input  = join(input_dir, "annotations.csv")
    annotations_file_output = join(output_dir, "graphs", "annotations.csv")
    data_out = join(output_dir, "graphs", "data")

    print("Loading dataset...")
    transformers = [FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]), RTPosePreprocessing(), ToRTPoseInput(0)]
    video_dataset = VideoDataset(annotations_file_input, input_dir, transform=Compose(transformers), load_copy=False)

    model = PoseModel()
    model = model.to(device)
    model.load_state_dict(torch.load("model/weights/vgg19.pth", map_location=torch.device(device)))

    print("Setup of video predictor to reduce RAM or VRAM (cpu or gpu) usage...")
    no_video_subsets = 64
    video_predictor = VideoPredictor(model, no_video_subsets, device, output_handler=output_handler)

    print("Starting generations of graphs...")
    for sample_idx in range(len(video_dataset)):
        sample = video_dataset[sample_idx]
        
        sample_name       = sample["name"]
        sample_label      = sample["label"]
        sample_properties = sample["properties"]
        with torch.no_grad():
            (branch1, branch2) = video_predictor.predict(sample["data"])

        del sample # free memory

        paf = branch1.data.cpu().numpy().transpose(0, 2, 3, 1)
        heatmap = branch2.data.cpu().numpy().transpose(0, 2, 3, 1)

        no_frames = len(paf[:]) # == len(heatmap[:])
        frames = []
        for frame in range(no_frames):
            humans = paf_to_pose_cpp(heatmap[frame], paf[frame], config)
            frames.append(humans)

        del paf, heatmap # free as much memory as possible
        # TODO Remove background humans

        name, _ = splitext(basename(sample_name))
        save_file = join(data_out, sample_label, name + ".json")
        metadata = {
            "filename": sample_name,
            "body_part_translation":body_part_translation,
            "body_construction":body_part_construction,
            "label":sample_label,
            "properties":sample_properties
        }

        save_humans(save_file, frames, metadata)
        print(save_file)
        
        # write to annotations file
        with open(annotations_file_output, "a") as f:
            f.write("{},{}\n".format(join("data", sample_label, name + ".json"), sample_label))
        exit()
    print("\n###############   Graph generation done   ##############\n")


def setup(input_dir, output_dir):
    """ Setup structure of the output directory. """
    print("\n##################   Starting setup   ##################\n")
    annotations_path = join(input_dir, "annotations.csv")
    print("Reading data annotations...")
    annotations = pd.read_csv(annotations_path)
    
    labels = annotations.iloc[:,1]
    unique_labels = list(set(labels))
    print("These are the labels that were found in the dataset:\n{}".format(unique_labels))

    data_out_root = join(output_dir, "graphs")

    # delete data, start from clean slate
    if exists(data_out_root):
        print("Found old graphs, deleting...")
        rmtree(data_out_root)
    
    data_out = join(data_out_root, "data")
    makedirs(data_out) # recursive create of root and data dir

    for label in unique_labels:
        path = join(data_out, label)
        print("Creating directory {}".format(path))
        mkdir(path)

    annotations_out = join(data_out_root, "annotations.csv") 
    print("Creating file {}".format(annotations_out))
    with open(annotations_out,'w+') as f:
        f.write("# filename,label\n") # Header
            
    print("\n##################   Setup Done   ##################\n")
    



def parse_args(argv):
    try:
        opts, _ = getopt.getopt(argv, 'hi:o:', ['input_dir=', 'output_dir='])
    except getopt.GetoptError:
       print('json_to_ndarray.py --input <data directory> --output <output directory>')
       sys.exit(2)

    input_dir = ""
    output_dir = ""
    for opt, arg in opts:
        if opt == '-h':
            print('generate_graphs.py -i <input directory> -o <output directory>')
            sys.exit()
        elif opt in ("-i", "--input"):
            input_dir = arg
        elif opt in ("-o", "--output"):
            output_dir = arg
    return input_dir, output_dir 


if __name__ == "__main__":
    input_dir, output_dir = parse_args(sys.argv[1:])
    setup(input_dir, output_dir)
    generate_graphs(input_dir, output_dir, device="cuda")