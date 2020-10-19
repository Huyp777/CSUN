import torch
from torch import nn

pooling_counts = [15, 8, 8, 8]
num_clips = 128

class get2dMap(nn.Module):
    def __init__(self, pooling_counts, N):
        super(get2dMap, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c):
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                mask2d[0, j] = 0
                mask2d[i, N - 1] = 0
                maskij.append((i, j))
            stride *= 2

        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        map2d[:, :, :, N - 1] = 0
        return map2d


def build_2dmap():
    return get2dMap(pooling_counts, num_clips)
