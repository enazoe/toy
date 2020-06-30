from utils import *
from darknet import *
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import matplotlib.patches as patches

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":

    Darknet("./cfg/yolov3.cfg")
    #trans = transforms.Compose([transforms.ToTensor()])
    #dataset = FudanPedDataset("./data/PennFudanPed",trans)

    #trainloader = torch.utils.data.DataLoader(dataset , batch_size=2,
                  #shuffle=True, num_workers=1,collate_fn=collate_fn)
   
    #img,boxex = next(iter(trainloader))


   