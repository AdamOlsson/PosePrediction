import torch
from model.PoseModel import PoseModel

if __name__ == "__main__":
    model = PoseModel()
    model.load_state_dict(torch.load("model/weights/vgg19_weights.pt"))