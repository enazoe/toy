import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
  
class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self,cfg_file):
        super(Darknet,self).__init__()
        self.blocks = self.parse_cfg(cfg_file)
        self.net_info,self.module_list = self.create_moudles(self.blocks)

    def forward(self,x,target = None,device="cuda"):
        modules = self.blocks[1:]
        output_cahce = {}
        write = 0
        losses = 0
        for i ,module in enumerate(modules):
            type = module["type"]

            if "convolutional" == type or type == "upsample":
                x = self.module_list[i](x)
            elif "shortcut" == type:
                pointer =int(module["from"])
                x = output_cahce[i-1] + output_cahce[i+pointer]
            elif "route" == type:
                layers = module["layers"]
                layers = [int(x) for x in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = output_cahce[i + layers[0]]
                else:
                    if layers[1] > 0 :
                        layers[1] = layers[1] - i
                    map1 = output_cahce[i + layers[0]]
                    map2 = output_cahce[i + layers[1]]
                    x = torch.cat((map1,map2),1)
            elif "yolo" == type:
                anchors = self.module_list[i][0].anchors
                input_dim = int (self.net_info["height"])
                num_classes = int (module["classes"])
                x,loss = self.parse_prediction(x,num_classes,anchors,input_dim,device,target)

                if not write:
                    write = 1
                    detection = x
                else:
                    detection = torch.cat((detection,x),1)
                losses += loss
            output_cahce[i] = x
        return detection,losses
            
    def parse_prediction(self,prediction,classes,anchors,input_dim,device,targets=None):
        
        if targets is not None:
            loss = self.comput_loss(prediction,targets,anchors,device)
        else:
            grid_size = prediction.size(2)
            batch_size = prediction.size(0)
            anchor_size = len(anchors)

            prediction = prediction.view(batch_size,anchor_size*(5+classes),grid_size*grid_size)
            prediction = prediction.transpose(1,2).contiguous()
            prediction = prediction.view(batch_size,anchor_size*grid_size*grid_size,5+classes)

            prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
            prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
            #confidence
            prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
            #add cx,cy to x,y
            grid = np.arange(grid_size)
            a,b = np.meshgrid(grid,grid)
            x_offset = torch.FloatTensor(a).view(-1,1)
            y_offset = torch.FloatTensor(b).view(-1,1)
            if "cuda" == device:
                x_offset = x_offset.cuda()
                y_offset = y_offset.cuda()
            x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,anchor_size).view(-1,2).unsqueeze(0)
            prediction[:,:,:2] += x_y_offset

            # w,h
            stride = input_dim // grid_size
            anchors = [(a[0]/stride,a[1]/stride) for a in anchors]
            anchors = torch.FloatTensor(anchors)
            if "cuda" == device:
                anchors = anchors.cuda()
            anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
            prediction[:,:,2:4] =torch.exp( prediction[:,:,2:4]) *anchors
        

            #classes
            prediction[:,:,5:5+classes] = torch.sigmoid((prediction[:,:,5:5+classes]))

            prediction[:,:,0:4] *= stride


        return prediction,loss

    def comput_loss(self,prediction,targets,anchors,device):
        loss = 0
        batch_size = prediction.size(0)
        channels = prediction.size(1)
        classes = int(prediction.size(1)/3-5)
        grid_size = prediction.size(2)
        anchor_size = len(anchors)
        #format x
        prediction = (prediction.view(batch_size,channels,grid_size*grid_size)
            .permute(0,2,1)
            .contiguous()
            .view(batch_size,grid_size*grid_size*anchor_size,5+classes))
        #format anchor
        anchors = [[anchor[0]/grid_size,anchor[1]/grid_size]for anchor in anchors]
        anchors = torch.FloatTensor(anchors)
        anchors_backup = anchors.clone();
        anchors_backup = anchors_backup.to(device)
        anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0).repeat(2,1,1)
        anchors = anchors.to(device)
        #x,y,w,h conf ,classes
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        prediction[...,2:4] = torch.exp(prediction[...,2:4])*anchors
        w = prediction[...,2]
        h = prediction[...,3]
        conf = torch.sigmoid(prediction[...,4])
        clas = torch.sigmoid(prediction[...,5:])

        #ground truch
        obj_mask = torch.zeros((batch_size,grid_size*grid_size*anchor_size,1))
        gbc = targets[:,2:]*grid_size
        gxy = gbc[...,0:2]
        gwh = gbc[...,2:4]
        iou_mat = self.compute_iou_matrix(gwh,anchors_backup)
        ious,best_index = iou_mat.max(1)
        gc,gr = gxy.long().t()
        for i in range(targets.size(0)):
            obj_mask[int(targets[i,0]),int(gc[i])+int(grid_size)*int(gr[i])+int(best_index[i]),0] = 1;
        print(obj_mask)
        return loss

    def bbox_wh_iou(self,wh1, wh2):
        wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
        return inter_area / union_area

    def compute_iou_matrix(self,targets,anchors):
        iou_matrix = []
        for i in range(targets.size(0)):
            ious = [self.bbox_wh_iou(targets[i,:],anchor) for anchor in anchors]
            ious = torch.stack(ious)
            iou_matrix.append(ious)
        iou_mat = torch.stack(iou_matrix,0)
        return iou_mat
    #使用sequential模块创建网络的子模块，like con_bn_leakly ....
    def create_moudles(self,blocks_):
        net_info = blocks_[0]
        module_list = nn.ModuleList()
        output_filters = []
        pre_filters = 3
        for index,block in enumerate(blocks_[1:]):
            module = nn.Sequential()

            if block["type"] == "convolutional": #卷积层
                #add conv
                filters = int(block["filters"])
                size = int(block["size"])
                stride = int(block["stride"])
                padding = int(block["pad"])
                if padding:
                    pad = (size-1)//2
                else:
                    pad = 0
                try:
                    normalization = int(block["batch_normalize"])
                    bias = False
                except:
                    normalization = 0
                    bias = True
                conv = nn.Conv2d(pre_filters,filters,size,stride,pad, bias = bias)
                module.add_module("conv_{0}".format(index),conv)

                #add normlization
                if normalization:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), bn)

                #add activation
                if block["activation"] == "leaky":
                    act = nn.LeakyReLU(0.1,True)
                    module.add_module("leaky_{0}".format(index),act)
                print(module)
            elif block["type"] == "shortcut": #res
                short = EmptyLayer() 
                module.add_module("short_{0}".format(index),short) 
                print(module)
            elif block["type"] == "upsample":
                scale = int(block["stride"])
                upsample_layer = nn.Upsample(scale_factor = scale,mode = "nearest")
                module.add_module("upsample_{0}".format(index),upsample_layer)
                print(module)
            elif block["type"] == "route":
                block["layers"] = block["layers"].split(",")
                #start layer index
                start = int(block["layers"][0])
                try:
                    end = int(block["layers"][1]) #做concate层的index
                except:
                    end = 0 #不做concate只是重定位网络流指针
                
                #Positive anotation
                if start > 0: 
                    start = start - index
                if end > 0:
                    end = end - index
                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters= output_filters[index + start]
                print(module)
            elif block["type"] == "yolo":
                mask = block["mask"].split(",")
                mask = [int(i) for i in mask]

                anchors = block["anchors"].split(",")
                anchors = [int(i) for i in anchors]
                anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
                anchors = [anchors[i] for i in mask]
                yolo = DetectionLayer(anchors)
                module.add_module("yolo_{0}".format(index),yolo)
                print(module)
            pre_filters = filters
            module_list.append(module)
            output_filters.append(filters)

        return (net_info,module_list)


    def parse_cfg(self,cfg_file):
        file = open(cfg_file, 'r')
        lines = file.read().split("\n")
        lines = [line for line in lines if len(line)>0]
        lines = [line for line in lines if line[0] !='#']
        lines = [line.rstrip().lstrip() for line in lines]

        block ={}
        blocks = []
        for line in lines:
            if line[0]=='[':
                if len(block)!=0:
                    blocks.append(block)
                    block={}
                block["type"]=line[1:-1].rstrip().strip()
            else:
                key,value = line.split('=')
                block[key.rstrip()] = value.lstrip()
    
        blocks.append(block)
        return blocks

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)