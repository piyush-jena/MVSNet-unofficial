import torch.nn as nn
import torch.nn.functional as F

class DepthMapRefineNet(nn.Module):
    def __init__(self):
        super(DepthMapRefineNet, self).__init__()
        self.conv0 = nn.Conv2d(4, 32, 3, 1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 1, 3, 1, 1, bias=False)
        #self.bn3 = nn.BatchNorm2d(1)
    
    def forward(self, img, depth_init):
        x = F.cat((img, depth_init), dim=1)
        
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
        #x = self.bn3(x)
        #x = F.relu(x, inplace=True)

        return x + depth_init