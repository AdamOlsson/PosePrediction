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
from util.load_save_humans import save_humans
from paf.paf_to_pose import paf_to_pose_cpp
from paf.common import draw_humans, CocoPart
from paf.body_part_construction import body_part_construction, body_part_translation

import matplotlib.pyplot as plt
import numpy as np
import cv2


def showRandomSample(dataset):
    """
    Sample and display 4 random samples from the dataset.
    """
    fig, ax = plt.subplots(1,4, figsize=(15,8))
 
    for i in range(4):
        rnd = np.random.randint(len(dataset))
        item = dataset[rnd]
        image, label = item['image'], item['label']

        if isinstance(item['image'], torch.Tensor):
            image = image.numpy().transpose((0, 2 ,3 ,1))
        
        ax[i].imshow(image[0,:])
        ax[i].set_title(item['label'], fontsize=20, fontweight='bold') 
 
    plt.show()

if __name__ == "__main__":

    image_path_data = "../exercise_prediction/data/images/"
    image_path_annotations = "../exercise_prediction/data/images/annotations.csv"
    
    video_path_data = "../exercise_prediction/data/videos/"
    video_path_annotations = "../exercise_prediction/data/videos/annotations.csv"

    config = load_config("config.json")

    no = 1
    transformers = [FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]), RTPosePreprocessing(), ToRTPoseInput(0)]
    image_dataset = ImageDataset(image_path_annotations, image_path_data, transform=Compose(transformers))
    video_dataset = VideoDataset(video_path_annotations, video_path_data, transform=Compose(transformers))
    
    dataset = video_dataset
    #dataset = image_dataset

    data = dataset[no]['data']
    data_copy = dataset[no]['copy']

    #showRandomSample(dataset)

    model = PoseModel()
    model.load_state_dict(torch.load("model/weights/vgg19.pth"))

    with torch.no_grad():
        (branch1, branch2), _ = model(data)

    paf = branch1.data.numpy().transpose(0, 2, 3, 1)
    heatmap = branch2.data.numpy().transpose(0, 2, 3, 1)

    #humans = paf_to_pose_cpp(heatmap, paf, config)

    no_frames = 2#len(paf[:]) # == len(heatmap[:])
    frames = []
    for frame in range(no_frames):
        humans = paf_to_pose_cpp(heatmap[frame], paf[frame], config)
        frames.append(humans)
    
    metadata = {"filename": dataset[no]['name'], "body_part_translation":body_part_translation, "body_construction":body_part_construction}

    save_humans(frames, metadata)
    out0 = draw_humans(data_copy[0], frames[0])
    out1 = draw_humans(data_copy[1], frames[1])

    cv2.imwrite('result0.png', out0)
    cv2.imwrite('result1.png', out1)


    # TODO: Look at Dawids Thesis
    # TODO: Do pose prediction for video
    # TODO: Remove background humans
    # TODO: Build a graph between frames in poses
    # TODO: Verify that downloading this package and using in another package works as intended, i.e imports.
