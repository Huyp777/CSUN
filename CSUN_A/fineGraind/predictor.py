import torch
from torch import nn

input_size = 512
hidden_size = 512
kernel_size = 9
num_stack_layers = 4


def mask2weight(mask2d, mask_kernel, padding=0):
    weight = torch.conv2d(mask2d[None, None, :, :].float(),
                          mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight


class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, k, num_stack_layers, mask2d):
        super(Predictor, self).__init__()

        mask_kernel = torch.ones(1, 1, k, k).to(mask2d.device)
        first_padding = (k - 1) * num_stack_layers // 2

        self.weights = [
            mask2weight(mask2d, mask_kernel, padding=first_padding)
        ]
        self.convs = nn.ModuleList(
            [nn.Conv2d(input_size, hidden_size, k, padding=first_padding)]
        )

        for _ in range(num_stack_layers - 1):
            self.weights.append(mask2weight(self.weights[-1] > 0, mask_kernel))
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k))
        self.pred = nn.Conv2d(hidden_size, 1, 1)

    def forward(self, x):
        for conv, weight in zip(self.convs, self.weights):
            x = conv(x).relu() * weight
        x = self.pred(x).squeeze_()
        return x


def build_predictor(mask2d):
    return Predictor(input_size, hidden_size, kernel_size, num_stack_layers, mask2d)
