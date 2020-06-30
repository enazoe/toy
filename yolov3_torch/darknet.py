import torch
import torch.nn as nn
import torch.nn.functional as F


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
  


class Darknet(nn.Module):
    def __init__(self,cfg_file):
        super(Darknet,self).__init__()
        self.blocks = self.parse_cfg(cfg_file)
        self.create_moudles(self.blocks)

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
            pre_filters = filters
            module_list.append(module)
            output_filters.append(filters)



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
