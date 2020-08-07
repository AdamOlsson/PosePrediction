# torch & torchvision
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose

from model.PoseModel import PoseModel
from Datasets.ImageDataset import ImageDataset

# Transformers
from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput

import matplotlib.pyplot as plt
import numpy as np


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

    path_data = "../exercise_prediction/data/images/"
    path_annotations = "../exercise_prediction/data/images/annotations.csv"


    transformers = [FactorCrop(8, dest_size=368), RTPosePreprocessing(), ToRTPoseInput(0)]
    dataset = ImageDataset(path_annotations, path_data, transform=Compose(transformers))
    
    #showRandomSample(dataset)

    model = PoseModel()
    model.load_state_dict(torch.load("model/weights/vgg19.pt"))

    with torch.no_grad():
        output, _ = model(dataset[0]['image'])

    print(output)

