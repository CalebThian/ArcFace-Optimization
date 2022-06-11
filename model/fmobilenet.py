import torch
import torch.nn as nn
import torch.nn.functional as F


#我们的第一个块叫Flatten，它的功能是将一个张量展平
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)