# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 05:36:07 2024

@author: 17689
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """ Basic convolutional block with normalization and activation. """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AdaptiveResBlock(nn.Module):
    """ Residual block with dilated convolutions for maintaining spatial resolution. """
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(AdaptiveResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ScaleInvariantNetwork(nn.Module):
    def __init__(self):
        super(ScaleInvariantNetwork, self).__init__()
        self.entry = ConvBlock(1, 64)
        self.res1 = AdaptiveResBlock(64, 64, dilation_rate=1)
        self.res2 = AdaptiveResBlock(64, 64, dilation_rate=2)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.final_conv(x)
        return x

import matplotlib.pyplot as plt
# Initialize model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ScaleInvariantNetwork().to(device)
def generate_test_image(size=64):
    """ Generate a synthetic test image with a simple pattern. """
    image = torch.zeros((1, 1, size, size))
    image[0, 0, size//4:size//4*3, size//4:size//4*3] = 1  # Add a white square in the center
    return image

# Generate low and high resolution images
low_res_image = generate_test_image(64).to(device)
high_res_image = generate_test_image(128).to(device)

# Run model on both images
low_res_output = model(low_res_image)
high_res_output = model(high_res_image)

# Function to plot images
def plot_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image.squeeze().cpu().detach(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Display the outputs
plot_images([low_res_image, low_res_output, high_res_image, high_res_output],
            ['Low Resolution Input', 'Low Resolution Output', 'High Resolution Input', 'High Resolution Output'])

