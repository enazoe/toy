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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FudanPedDataset("./data/PennFudanPed",416)
    trainloader = torch.utils.data.DataLoader(dataset , batch_size= 8,
                  shuffle=True, num_workers=4,collate_fn= dataset.collate_fn)
    
    net = Darknet("./cfg/yolov3_1class.cfg").to(device)
    
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(100000):
        net.train()
        for batch_i ,(imgs,targets) in enumerate(trainloader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            detection,loss = net(imgs,targets,device)
            loss.backward()
            optimizer.step()
            if batch_i % 20 == 0:
                print("loss:",epoch*len(trainloader)+batch_i,",",loss.item())