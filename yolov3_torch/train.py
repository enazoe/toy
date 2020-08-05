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
    net.apply(weights_init_normal)
    net.load_weights("./cfg/darknet53.conv.74")
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(100000):
        net.train()
        for batch_i ,(imgs,targets) in enumerate(trainloader):
           # check_dataset(imgs,targets)
            optimizer.zero_grad()
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device),requires_grad = False)
            detection,loss = net(imgs,targets,device)
            loss.backward()
            optimizer.step()
            if batch_i % 10 == 0:
                print("epoch:",epoch," index:",batch_i,",",loss.item())
        if epoch % 10 == 0:
            torch.save(net.state_dict(),f"cfg/yolov3_ckpt_%d.pth" % epoch)
