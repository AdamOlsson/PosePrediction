# torch & torchvision
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose

from model.PoseModel import PoseModel
from Datasets.ImageDataset import ImageDataset

# Transformers
from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToTensor import ToTensor


if __name__ == "__main__":

    path_data = "../exercise_prediction/data/images/"
    path_annotations = "../exercise_prediction/data/images/annotations.csv"


    transformers = [FactorCrop(8, dest_size=368), RTPosePreprocessing(), ToTensor()]
    dataset = ImageDataset(path_annotations, path_data, transform=Compose(transformers))
    

    # model = PoseModel()
    # model.load_state_dict(torch.load("model/weights/vgg19.pt"))

    # img1 = cv2.imread("data/backsquat925.jpg")


