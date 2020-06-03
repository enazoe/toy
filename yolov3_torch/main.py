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

   trans = transforms.Compose([transforms.ToTensor()])
   dataset = FudanPedDataset("./data/PennFudanPed",trans)

   trainloader = torch.utils.data.DataLoader(dataset , batch_size=2,
                                       shuffle=True, num_workers=1,collate_fn=collate_fn)
   
   img,boxex = next(iter(trainloader))
  # print(img.shape,boxex.shape)
   img = torchvision.utils.make_grid(img[0])
   npimg = img.numpy()
   npimg = np.transpose(npimg, (1, 2, 0))
   h,w,_= npimg.shape
   # Create figure and axes
   fig,ax = plt.subplots(1)

   # Display the image
   ax.imshow(npimg)

   # Create a Rectangle patch
   #boxex = boxex.numpy()
   for i,bath in enumerate(boxex):
      if i>0:
         continue
      print(i)
      for box in bath:
         rect = patches.Rectangle((box[0]*w,box[1]*h),(box[2]-box[0])*w,(box[3]-box[1])*h,linewidth=1,edgecolor='r',facecolor='none')
         # Add the patch to the Axes
         ax.add_patch(rect)

   plt.show()
   