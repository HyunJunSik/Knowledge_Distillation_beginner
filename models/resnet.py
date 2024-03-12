from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F

'''
This ResNet code from https://github.com/megvii-research/mdistiller/blob/master/mdistiller/models/cifar/resnet.py
'''

__all__ = ["resnet"]

def conv3x3(in_planes, out_planes, stride=1):
    '''
    3 x 3 convolution with padding
    '''
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )
    
class BasicBlock(nn.Module):
    '''
    BasicBlock : Conv층 2개로 이루어지며, 잔차가 포함된 block
    '''
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True) #inplace=True로 하면 들어가는 인수 값이 output과 동일하게 변동, 메모리 절약 효과
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out) 
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Downsample이 필요하다면?
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        preact = out
        out = F.relu(x) #init에 쓰는 nn.ReLU와 다르게 forward에서는 F.relu 쓰인다고함
        if self.is_last:
            return out, preact
        else:
            return out

class ResNet(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=100):
        super(ResNet, self).__init__()
    
        if block_name.lower() == "basicblock":
            assert(
                depth - 2
            ) % 6 == 0, "Basic block depth should be 6n+2, 20, 32, 44, 56, 110 등"
            n = (depth - 2) // 6
            block = BasicBlock
        else:
            raise ValueError("block_name should be Basicblock")
        
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        