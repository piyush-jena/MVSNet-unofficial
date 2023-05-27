import torch.nn as nn
import torch.nn.functional as F

class CostVolumeRegularizationNet(nn.Module):
    def __init__(self):
        super(CostVolumeRegularizationNet, self).__init__()
        self.conv0 = nn.Conv3d(32, 8, 3, 1, 1, bias=False)
        self.bn0 = nn.BatchNorm3d(8)

        self.conv1 = nn.Conv3d(8, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 16, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(16)

        self.conv3 = nn.Conv3d(16, 32, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(32)

        self.conv4 = nn.Conv3d(32, 32, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(32)

        self.conv5 = nn.Conv3d(32, 64, 3, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm3d(64)

        self.conv6 = nn.Conv3d(64, 64, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm3d(64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True))

        self.conv10 = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        conv0 = x = F.relu(x, inplace=True)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        conv2 = x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = self.bn4(x)
        conv4 = x = F.relu(x, inplace=True)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)

        x = conv4 + self.conv7(x)
        x = conv2 + self.conv8(x)
        x = conv0 + self.conv9(x)
        x = self.conv10(x)

        return x