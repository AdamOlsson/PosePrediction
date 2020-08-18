# torch & torchvision
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose

from model.PoseModel import PoseModel
from Datasets.ImageDataset import ImageDataset
from Datasets.VideoDataset import VideoDataset
from Datasets.VideoPredictor import VideoPredictor

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

if __name__ == "__main__":

    device = "cuda"

    image_path_data = "data/images/"
    image_path_annotations = "data/images/annotations.csv"

    video_path_data = "data/videos/"
    video_path_annotations = "data/videos/annotations.csv"

    config = load_config("config.json")

    no = 0
    fs = 1
    transformers = [FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]), RTPosePreprocessing(), ToRTPoseInput(0)]
    #image_dataset = ImageDataset(image_path_annotations, image_path_data, transform=Compose(transformers), load_copy=True)
    video_dataset = VideoDataset(video_path_annotations, video_path_data, transform=Compose(transformers), load_copy=False, frame_skip=fs)

    #dataset = video_dataset

    model = PoseModel()
    model = model.to(device)
    model.load_state_dict(torch.load("model/weights/vgg19.pth", map_location=torch.device(device)))

    video_predictor = VideoPredictor(model, video_dataset, 64, device, output_handler=output_handler)

    with torch.no_grad():
        (branch1, branch2) = video_predictor.predict(no)

    # branch1.shape = torch.Size([1, 38, 82, 46])
    # branch2.shape = torch.Size([1, 19, 82, 46])

    paf = branch1.data.cpu().numpy().transpose(0, 2, 3, 1)
    heatmap = branch2.data.cpu().numpy().transpose(0, 2, 3, 1)

    no_frames = len(paf[:]) # == len(heatmap[:])
    frames = []
    for frame in range(no_frames):
        humans = paf_to_pose_cpp(heatmap[frame], paf[frame], config)
        frames.append(humans)

    save_file = "data/graphs/{}/humans_{}.json".format(video_dataset[no]['type'], video_dataset[no]['type'])
    metadata = {"filename": video_dataset[no]['name'], "body_part_translation":body_part_translation, "body_construction":body_part_construction, "frame_skip":fs, "label":video_dataset[no]['label'], "info":video_dataset[no]['info']}
    save_humans(save_file, frames, metadata)
    print(save_file)

    # for images only
    #out = draw_humans(data_copy, frames[0])
    #cv2.imwrite('results/result.png', out)

    # TODO: Look at Dawids Thesis
    # TODO: When the graphs are connected temporally, I might need to add a 3d dimension. Preferebly this would
    #       be time in seconds. Is there metadata about a video so I can get the recorded fps and this the time 
    #       between frames?
    # TODO: Remove background humans
    # TODO: Build a graph between frames in poses
    # TODO: Verify that downloading this package and using in another package works as intended, i.e imports.
