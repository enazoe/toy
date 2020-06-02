from utils import *

from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision

if __name__ == "__main__":
   trans = transforms.Compose([transforms.ToTensor()])
   dataset = FudanPedDataset("./data/PennFudanPed",trans)

   trainloader = torch.utils.data.DataLoader(dataset , batch_size=4,shuffle=True)
   
   dataiter= iter(trainloader)
   img,boxex = dataiter.next()
   
   img = torchvision.utils.make_grid(img)
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()
   