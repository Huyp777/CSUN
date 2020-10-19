import torch
from torch import nn

input_size = 4096
hidden_size = 512
kernel_size = 2
stride = 2

class video_pool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(video_pool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        return self.pool(self.conv(x.transpose(1, 2)).relu())

def build_fp():
    return video_pool(input_size, hidden_size, kernel_size, stride)
