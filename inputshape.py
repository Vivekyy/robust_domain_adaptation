import torch
import torch.nn as nn
import torch.nn.functional as F

class testNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)

    def convs(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.dropout2d(F.max_pool2d(self.conv2(x), 2))
        return x

    def findInputShape(self):
        x = torch.randn(3,28,28).view(-1,3,28,28)
        inputshape = self.convs(x)
        return inputshape

model = testNet()
x = model.findInputShape()
print(x.size())
