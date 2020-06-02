import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import glob
import numpy as np
from PIL import Image
class FudanPedDataset(Dataset):
    def __init__(self,root="./",trans = None):
        self.img_list = glob.glob(root+"/PNGImages/*")
        self.label_list = glob.glob(root+"/PedMasks/*")
        self.len = len(self.img_list)
        self.trans = trans

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert("RGB")
        mask = Image.open(self.label_list[index])
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        img= self.trans(img)
        print(img.size)
        return (img,boxes)

    def __len__(self):
        return self.len