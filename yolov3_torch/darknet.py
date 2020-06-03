import torch.nn as nn


def parse_cfg(cfg_file):
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
    print(blocks)
    return


class Darknet(nn.Module):
    def __init__(self,cfg_file):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfg_file)