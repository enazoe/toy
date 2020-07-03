import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self,x,device):
        modules = self.blocks[1:]
        output_cahce = {}
        write = 0
        for i ,module in enumerate(modules):
            type = module["type"]

            if "convolution" == type:
                x = self.module_list[i](x)
            elif "shortcut" == type:
                pointer = module["from"]
                x = output_cahce[i-1] + output_cahce[i+pointer]
            elif "route" == module["route"]:
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
            elif "yolo" == module["yolo"]:
                anchors = self.module_list[i][0].anchors
                input_dim = int (self.net_info["height"])
                num_classes = int (module["classes"])
                x = self.parse_prediction(x,num_classes,anchors,input_dim,device)

                if not write:
                    write = 1
                    detection = x
                else:
                    detection = torch.cat((detection,x),1)

            output_cahce[i] = x
        return detection
            
    def parse_prediction(prediction,classes,anchors,input_dimm,CUDA):
        grid_size = prediction.size(2)
        batch_size = prediction.size(0)
        anchor_size = len(anchors)

        prediction = prediction.view(batch_size,anchor_size*(5+classes),grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size,anchor_size*grid_size*grid_size,5+classes)

        prediction = torch.sigmoid(prediction[:,:,:2])
        #add cx,cy to x,y
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid,grid)
        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)
        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()
        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
        prediction[:,:,:2] += x_y_offse

        # w,h
        stride = input_dim // grid_size
        anchors = [(a[0]/stride,a[1]/stride) for a in anchors]
        anchors = torch.FloatTensor(anchors)
        if CUDA:
            anchors = anchors.cuda()
        anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
        prediction[:,:,2:4] =torch.exp( prediction[:,:,2:4])
        prediction[:,:,2:4] = prediction[:,:,2:4]*anchors
        #confidence
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

        #classes
        prediction[:,:,5:5+classes] = torch.sigmoid((prediction[:,:,5:5+classes]))

        prediction[:,:,0:5] *= stride
        return prediction


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
                conv = nn.Conv2d(pre_filters,filters,size,stride,pad, bias)
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
