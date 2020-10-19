import torch
from torch import nn
from .tcn import TemporalConvNet

LEN_FEATURE_V = 500
DROP_OUT = 0.3
LEN_FEATURE_B = 256
KERNEL_SIZE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 23, 28, 34, 41, 50, 61, 75, 92, 113, 139, 171, 211]
STRIDE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 6, 7, 9, 11, 13, 17, 21]

class VideoModule(nn.Module):
    def __init__(self):
        super(VideoModule, self).__init__()
        self.tcn = TemporalConvNet(LEN_FEATURE_V, [448, 416, 384, 352, 320, 288, 256], kernel_size=7, dropout=DROP_OUT)
        self.conv1d = nn.ModuleList([nn.Conv1d(in_channels=256,
                                               out_channels=256,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dilation=1)
                                     for stride, kernel_size in zip(STRIDE, KERNEL_SIZE)])
        self.net = nn.Sequential(
            nn.RReLU(),
            nn.Linear(256, LEN_FEATURE_B),
            nn.RReLU(),
            nn.Linear(LEN_FEATURE_B, LEN_FEATURE_B)
        )
        self.trans = nn.Sequential(
            nn.RReLU(),
            nn.Linear(3761, 64)
        )

    def forward(self, feed):
        feed = self.tcn(feed)
        out = []
        for kernel_size, conv1d in zip(KERNEL_SIZE, self.conv1d):
            if feed.shape[2] >= kernel_size:
                tmp = conv1d(feed)
                out.append(tmp)
        out = torch.cat(out, dim=2)
        out = self.trans(out)
        out = out.permute(0, 2, 1)
        out = self.net(out)
        return out

def build_VideoModule():
    return VideoModule()
