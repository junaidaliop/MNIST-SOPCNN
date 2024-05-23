# models/model.py

import torch.nn as nn
import torch.nn.functional as F

class SOPCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SOPCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = F.relu(self.conv3(x))
        x = self.conv3(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
