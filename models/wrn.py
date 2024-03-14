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
            x = self.relu(self.bn1(x))
            x = self.relu(self.bn1(x))
        else:
            out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.conv1(out if self.equalIO else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalIO else self.convShortCut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate)
            )
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
    
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, droprate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "depth should be 6n + 4"
        n = (depth - 4) // 6
        block = BasicBlock

        # network block 전 Conv layer 
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, droprate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, droprate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, droprate)

        # GAP, classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.stage_channels = nChannels
    
    def forward(self, x):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.reshape(-1, self.nChannels)
        f4 = out
        out = self.fc(out)

        f1_pre = self.block2.layer[0].bn1(f1)
        f2_pre = self.block3.layer[0].bn1(f2)
        f3_pre = self.bn1(f3)

        return out

def wrn(**kwargs):
    '''
    Wide Residual Networks
    '''
    model = WideResNet(**kwargs)
    return model

def wrn_40_2(**kwargs):
    model = WideResNet(depth=40, widen_factor=2, **kwargs)
    return model, "wrn_40_2"

def wrn_40_1(**kwargs):
    model = WideResNet(depth=40, widen_factor=1, **kwargs)
    return model, "wrn_40_1"

def wrn_16_2(**kwargs):
    model = WideResNet(depth=16, widen_factor=2, **kwargs)
    return model, "wrn_16_2"

def wrn_16_1(**kwargs):
    model = WideResNet(depth=40, widen_factor=1, **kwargs)
    return model, "wrn_16_1"

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    net, _ = wrn_40_2(num_classes=100)
    logit = net(x)

    print(logit.shape)