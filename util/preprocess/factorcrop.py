## 
## This script achieves exactly the same as Transformers/FactorCrop.py. Due to memory constrainsts, preprocessing
## has to be done in steps. The script assumes that there is a file called annotations.csv in the input directory.
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
## # path, label
## <path1>,<label>
## <path2>,<label>
## ...
## 
## The output from this file is the path to the preprocessed datas root directory to allow for Linux
## based piping.
## 
## Usage:
## python generate_graphs.py --data_dir <path to data root> --out_dir <path to output dir>

import sys
sys.path.append("../..") # append project root dir

from Datasets.VideoDataset import VideoDataset
from Transformers.FactorCrop import FactorCrop
from Transformers.WriteVideoToDisc import WriteVideoToDisc

from util.load_config import load_config

# native
import sys, getopt
from shutil import rmtree
from os.path import exists, join, basename, splitext
from os import makedirs, mkdir

# misc
import pandas as pd # easy load of csv

def main(input_dir, output_dir):
    
    config = load_config("../../config.json")

    annotations = join(input_dir, "annotations.csv")
    transformers = [
        FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]),
        WriteVideoToDisc(write_loc=join(input_dir, "data"), path_annotations=annotations)
        ]
    dataset = VideoDataset(annotations, input_dir, transform=transformers)

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
        f.write("# filename,label,frames\n") # Header

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
    input_dir, output_dir = parse_args(sys.argv[1:])
    output_dir = setup(input_dir, output_dir)
    main(input_dir, output_dir)