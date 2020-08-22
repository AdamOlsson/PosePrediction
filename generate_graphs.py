
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


import sys, getopt
from shutil import rmtree
from os.path import exists, join
from os import makedirs, mkdir
import pandas as pd # easy load of csv

def generate_graphs():
    pass

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
