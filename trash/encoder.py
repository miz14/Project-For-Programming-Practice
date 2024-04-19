import torch
import torch.nn as nn
import numpy as np

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 23 * 20, 512),
            nn.ReLU(),
            nn.Linear(512, 32)
        )

    def forward(self, x):
        x = x[25:, :, :]
        x = torch.tensor(np.moveaxis(x, -1, 0)).float()
        x = self.conv_layers(x)
        x = torch.flatten(x)
        x = self.fc_layers(x)
        return x