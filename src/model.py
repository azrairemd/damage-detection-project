import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        features = []

        x = self.stem(x)
        features.append(x)

        x1 = self.layer1(self.pool(x))
        features.append(x1)

        x2 = self.layer2(x1)
        features.append(x2)

        x3 = self.layer3(x2)
        features.append(x3)

        x4 = self.layer4(x3)
        features.append(x4)

        return features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.block4 = DecoderBlock(in_channels=1024, skip_channels=512, out_channels=256)
        self.block3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.block2 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64)
        self.block1 = DecoderBlock(in_channels=64, skip_channels=128, out_channels=32)

        self.final_upsample = nn.Upsample(scale_factor=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features):
        f0, f1, f2, f3, f4 = features

        x = self.block4(f4, f3)
        x = self.block3(x, f2)
        x = self.block2(x, f1)
        x = self.block1(x, f0)

        x = self.final_upsample(x)
        x = self.final_conv(x)
        return x


class SiameseUNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, pre, post):
        features_pre = self.encoder(pre)
        features_post = self.encoder(post)

        combined = []
        for f_pre, f_post in zip(features_pre, features_post):
            combined.append(torch.cat([f_pre, f_post], dim=1))

        return self.decoder(combined)