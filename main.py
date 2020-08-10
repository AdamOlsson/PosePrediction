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
from paf.paf_to_pose import paf_to_pose_cpp
from paf.common import draw_humans

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

    no = 10
    transformers = [FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]), RTPosePreprocessing(), ToRTPoseInput(0)]
#    image_dataset = ImageDataset(image_path_annotations, image_path_data, transform=Compose(transformers))
    video_dataset = VideoDataset(video_path_annotations, video_path_data, transform=Compose(transformers))
    
    image = image_dataset[no]['data']
    image_copy = image_dataset[no]['copy']
    
    #showRandomSample(dataset)

    model = PoseModel()
    model.load_state_dict(torch.load("model/weights/vgg19.pth"))

    with torch.no_grad():
        (branch1, branch2), _ = model(image)

    paf = branch1.data.numpy().transpose(0, 2, 3, 1)[0]
    heatmap = branch2.data.numpy().transpose(0, 2, 3, 1)[0]

    humans = paf_to_pose_cpp(heatmap, paf, config)

    out = draw_humans(image_copy, humans)
    cv2.imwrite('result.png', out)

    # TODO: Make pose on video
    # TODO: Look at Dawids Thesis
    # TODO: Do pose prediction for video
    # TODO: Remove background humans
    # TODO: Build a graph between frames in poses
    # TODO: Verify that downloading this package and using in another package works as intended, i.e imports.
