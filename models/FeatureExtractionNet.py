import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractionNet(nn.Module):
    def __init__(self):
        super(FeatureExtractionNet, self).__init__()
        self.inplanes = 32

        self.conv0 = nn.Conv2d(3, 8, 3, 1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(8)

        self.conv1 = nn.Conv2d(8, 8, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 5, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 32, 5, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x, inplace=True)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)

        x = self.conv7(x)
        
        return x