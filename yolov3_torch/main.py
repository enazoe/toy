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

def collate_fn(batch):
    return tuple(zip(*batch))

def disp_result(img_raw,net_dim,prediction):
    colors = pkl.load(open("pallete", "rb"))
    h,w = img_raw.shape[0],img_raw.shape[1]
    print(w,h)
    s = np.max([w,h])/net_dim
    prediction = prediction.cpu()
    obj_nums = prediction.size(0)
    for r in range(obj_nums):
        color = random.choice(colors)
        obj = prediction[r,:]
        x1 = prediction[r,1]*s
        y1 = (prediction[r,2]-(net_dim-h/s)/2)*s
        x2 = prediction[r,3]*s
        y2 = (prediction[r,4]-(net_dim-h/s)/2)*s
        print(x1,y1,x2,y2)
        cv2.rectangle(img_raw,(x1,y1),(x2,y2),color,2)
    cv2.imshow("result",img_raw)
    cv2.waitKey()
if __name__ == "__main__":

    img = cv2.imread("data/dog-cycle-car.png")
    img_raw = img.copy()
    img = prep_image(img,416)
    net = Darknet("./cfg/yolov3.cfg")
    net.load_weights("./cfg/yolov3.weights")
    CUDA = 0
    if torch.cuda.is_available():
        CUDA = 1
        img = img.cuda()
        net.cuda()
    net.eval()

    with torch.no_grad():
        prediction = net(Variable(img),CUDA)
    output = post_process(prediction, 0.8, 80, 0.4)
    disp_result(img_raw,416,output)
    save_file(output)