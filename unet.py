import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual block implementation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu=True):
        super(ResidualBlock, self).__init__()
        self.use_relu = use_relu
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = F.relu(out)
        out = self.bn2(self.conv2(out))
        if self.use_relu:
            out = F.relu(out)
        out = out + residual
        return out


# UNet with ResNet-style blocks
class ResNetUNet(nn.Module):
    def __init__(self, img_size=(512, 512), no_channels=3, start_neurons=32, dropout_rate=0.25):
        super(ResNetUNet, self).__init__()

        self.dropout_rate = dropout_rate

        # Downsampling layers
        self.conv1 = nn.Conv2d(no_channels, start_neurons * 1, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            ResidualBlock(start_neurons * 1, start_neurons * 1, use_relu=True),
            ResidualBlock(start_neurons * 1, start_neurons * 1, use_relu=False)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(start_neurons * 1, start_neurons * 2, kernel_size=3, padding=1)
        self.block2 = nn.Sequential(
            ResidualBlock(start_neurons * 2, start_neurons * 2, use_relu=True),
            ResidualBlock(start_neurons * 2, start_neurons * 2, use_relu=False)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(start_neurons * 2, start_neurons * 4, kernel_size=3, padding=1)
        self.block3 = nn.Sequential(
            ResidualBlock(start_neurons * 4, start_neurons * 4, use_relu=True),
            ResidualBlock(start_neurons * 4, start_neurons * 4, use_relu=False)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Middle layers
        self.middle = nn.Sequential(
            nn.Conv2d(start_neurons * 4, start_neurons * 8, kernel_size=3, padding=1),
            ResidualBlock(start_neurons * 8, start_neurons * 8, use_relu=True),
            ResidualBlock(start_neurons * 8, start_neurons * 8, use_relu=False)
        )

        # Upsampling layers
        self.deconv3 = nn.ConvTranspose2d(start_neurons * 8, start_neurons * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv3 = nn.Conv2d(start_neurons * 8, start_neurons * 4, kernel_size=3, padding=1)
        self.block_u3 = nn.Sequential(
            ResidualBlock(start_neurons * 4, start_neurons * 4, use_relu=True),
            ResidualBlock(start_neurons * 4, start_neurons * 4, use_relu=False)
        )

        self.deconv2 = nn.ConvTranspose2d(start_neurons * 4, start_neurons * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv2 = nn.Conv2d(start_neurons * 4, start_neurons * 2, kernel_size=3, padding=1)
        self.block_u2 = nn.Sequential(
            ResidualBlock(start_neurons * 2, start_neurons * 2, use_relu=True),
            ResidualBlock(start_neurons * 2, start_neurons * 2, use_relu=False)
        )

        self.deconv1 = nn.ConvTranspose2d(start_neurons * 2, start_neurons * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv1 = nn.Conv2d(start_neurons * 2, start_neurons * 1, kernel_size=3, padding=1)
        self.block_u1 = nn.Sequential(
            ResidualBlock(start_neurons * 1, start_neurons * 1, use_relu=True),
            ResidualBlock(start_neurons * 1, start_neurons * 1, use_relu=False)
        )

        # Output layer
        self.output_layer = nn.Conv2d(start_neurons * 1, 2, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Down
        conv1 = self.conv1(x)
        conv1 = self.block1(conv1)
        pool1 = F.dropout(self.pool1(conv1), p=self.dropout_rate)

        conv2 = self.conv2(pool1)
        conv2 = self.block2(conv2)
        pool2 = F.dropout(self.pool2(conv2), p=self.dropout_rate)

        conv3 = self.conv3(pool2)
        conv3 = self.block3(conv3)
        pool3 = F.dropout(self.pool3(conv3), p=self.dropout_rate)

        # Middle
        middle = self.middle(pool3)

        # Up
        deconv3 = self.deconv3(middle)
        uconv3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = self.uconv3(uconv3)
        uconv3 = self.block_u3(uconv3)

        deconv2 = self.deconv2(uconv3)
        uconv2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = self.uconv2(uconv2)
        uconv2 = self.block_u2(uconv2)

        deconv1 = self.deconv1(uconv2)
        uconv1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = self.uconv1(uconv1)
        uconv1 = self.block_u1(uconv1)

        # Output
        output = self.output_layer(uconv1)
        output = self.activation(output)
        return output


