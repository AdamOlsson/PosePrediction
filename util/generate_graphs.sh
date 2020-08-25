#!/bin/bash

#INPUT_DIR=$1
#OUTPUT_DIR=$2
INPUT_DIR="/home/adam/Documents/datasets/weightlifting/videos"
OUTPUT_DIR="/home/adam/Documents/datasets/weightlifting/videos"

PREPROCESSED_DIR=$(python preprocess/factorcrop.py -i "$INPUT_DIR" -o "$OUTPUT_DIR")


echo "$PREPROCESSED_DIR"