import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import glob
import numpy as np
from PIL import Image
import cv2
class FudanPedDataset(Dataset):
    def __init__(self,root="./",image_size = 416):
        self.img_list = glob.glob(root+"/PNGImages/*")
        self.label_list = glob.glob(root+"/PedMasks/*")
        self.len = len(self.img_list)
        self.image_size = image_size

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert("RGB")
        w,h=img.size
        mask = Image.open(self.label_list[index])
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)

        boxes_ = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            box = [xmin/float(w), ymin/float(h), xmax/float(w), ymax/float(h)]
            boxes_.append(torch.tensor([(box[0]+box[2])/2.,(box[1]+box[3])/2.,box[2]-box[0],box[3]-box[1]]))
        boxes_ = torch.stack(boxes_)
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)
        img = img.resize((self.image_size,self.image_size))
        img = transforms.ToTensor()(img)
        #img= self.trans(img)

        #extend boxes add batch position and class position
        boxes = torch.zeros((len(boxes_),6))
        boxes[:,2:] = boxes_;

        return (img,boxes)

    def __len__(self):
        return self.len

    def collate_fn(self,batch):
        imgs,targets = list(zip(*batch))
        imgs = torch.stack([img for img in imgs])
        for i,target in enumerate(targets):
            target[:,0]= i;
        targets = torch.cat(targets,0)
        return imgs,targets


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def save_file(tensor,file_name = "1.txt"):
    np.savetxt(file_name ,tensor.cpu())

def post_process(prediction,confidence,num_classes,nms_conf = 0.4):
    prediction_zero = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction =prediction * prediction_zero
    #左上、右下顶点
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)
    write = False
    for ind in range(batch_size):
        img_pred = prediction[ind]
        max_conf,max_conf_ind = torch.max(img_pred[:,5:5+num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_ind = max_conf_ind.float().unsqueeze(1)
        img_pred = torch.cat((img_pred[:,:5],max_conf,max_conf_ind),1)
        non_zero_ind =  (torch.nonzero(img_pred[:,4]))
        try:
            image_pred_ = img_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        if image_pred_.shape[0] == 0:
            continue 
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        for cls in img_classes:
            #perform NMS
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            #在结果前加一列batch index
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0