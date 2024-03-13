import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["wrn"]

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.droprate = droprate
        self.equalIO = in_planes == out_planes
        self.convShortCut = (
            (not self.equalIO)
            and nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
            )
            or None
        )
    
    def forward(self, x):
        if not self.equalIO:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalIO else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalIO else self.convShortCut)
        