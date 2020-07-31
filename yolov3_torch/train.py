from utils import *
from darknet import *
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import matplotlib.patches as patches
import cv2
import pickle as pkl
import random


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FudanPedDataset("./data/PennFudanPed",416)
    trainloader = torch.utils.data.DataLoader(dataset , batch_size=2,
                  shuffle=True, num_workers=1,collate_fn= dataset.collate_fn)
    imgs,targets = next(iter(trainloader))
    imgs = imgs.to(device)
    targets = targets.to(device)
    
    net = Darknet("./cfg/yolov3_1class.cfg").to(device)
    detection,loss = net(imgs,targets,device)