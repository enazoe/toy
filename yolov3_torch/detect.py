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

def disp_result(img_raw,net_dim,prediction):
    colors = pkl.load(open("pallete", "rb"))
    img_raw = np.array(img_raw)
    for pred in prediction:
        color = random.choice(colors)
        for rect in pred:
            box = rect.to("cpu")
            cv2.rectangle(img_raw,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,2)
    cv2.imshow("result",img_raw)
    cv2.waitKey()

def non_max_suppression(prediction, conf_thres=0.9, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Darknet("./cfg/yolov3_1class.cfg").to(device)
    net.load_state_dict(torch.load("./cfg/yolov3_ckpt_870.pth"))
    #net.load_weights("./cfg/yolov3.weights")
    net.eval()


    dataset = FudanPedDataset("./data/PennFudanPed",416)
    trainloader = torch.utils.data.DataLoader(dataset , batch_size= 1,
                    shuffle=True, num_workers=1,collate_fn= dataset.collate_fn)

    for batch_i ,(imgs,targets) in enumerate(trainloader):
        img_raw = imgs.clone()
        img_raw = img_raw.squeeze(0).permute(1,2,0).contiguous()
        imgs = imgs.to(device)
        with torch.no_grad():
            detections,loss = net(Variable(imgs),None,device)
            output =non_max_suppression(detections)
        disp_result(img_raw,416,output)