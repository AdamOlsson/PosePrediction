# torch & torchvision
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose

from model.PoseModel import PoseModel
from Datasets.ImageDataset import ImageDataset
from Datasets.VideoDataset import VideoDataset


# Transformers
from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput

# util
from util.load_config import load_config
from paf.util import save_humans, load_humans
from paf.paf_to_pose import paf_to_pose_cpp
from paf.common import draw_humans, CocoPart
from paf.body_part_construction import body_part_construction, body_part_translation

import matplotlib.pyplot as plt
import numpy as np
import cv2


if __name__ == "__main__":

    device = "cuda"

    image_path_data = "example_data/images/"
    image_path_annotations = "example_data/images/annotations.csv"

    video_path_data = "example_data/videos/"
    video_path_annotations = "example_data/videos/annotations.csv"

    config = load_config("config.json")

    no = 0
    fs = 6
    transformers = [FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]), RTPosePreprocessing(), ToRTPoseInput(0)]
    image_dataset = ImageDataset(image_path_annotations, image_path_data, transform=Compose(transformers), load_copy=True)
    video_dataset = VideoDataset(video_path_annotations, video_path_data, transform=Compose(transformers), load_copy=False, frame_skip=fs)

    dataset = video_dataset
    #dataset = image_dataset

    data = dataset[no]['data']
    # data_copy = dataset[no]['copy']

    model = PoseModel()
    model = model.to(device)
    model.load_state_dict(torch.load("model/weights/vgg19.pth", map_location=torch.device(device)))

    with torch.no_grad():
        (branch1, branch2), _ = model(data.to(device))

    paf = branch1.data.cpu().numpy().transpose(0, 2, 3, 1)
    heatmap = branch2.data.cpu().numpy().transpose(0, 2, 3, 1)

    no_frames = len(paf[:]) # == len(heatmap[:])
    frames = []
    for frame in range(no_frames):
        humans = paf_to_pose_cpp(heatmap[frame], paf[frame], config)
        frames.append(humans)

    save_file = "data/humans_{}.json".format(dataset[no]['type'])
    metadata = {"filename": dataset[no]['name'], "body_part_translation":body_part_translation, "body_construction":body_part_construction, "frame_skip":fs}
    save_humans(save_file, frames, metadata)
    print(save_file)

    # for images only
    #out = draw_humans(data_copy, frames[0])
    #cv2.imwrite('docs/result.png', out)

    # TODO: Look at Dawids Thesis
    # TODO: Add git lfs to setup
    # TODO: Remove background humans
    # TODO: Build a graph between frames in poses
    # TODO: Verify that downloading this package and using in another package works as intended, i.e imports.
