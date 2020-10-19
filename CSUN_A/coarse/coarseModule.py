import torch
from torch import nn

from .sentence import build_Sentence
from .video import build_VideoModule
import torch.nn.functional as func



class coarse(nn.Module):
    def __init__(self):
        super(coarse, self).__init__()
        self.sentence = build_Sentence()
        self.video = build_VideoModule()

    def forward(self, batches):

        feat_s = self.sentence(batches.queries, batches.wordlens)
        feat_v = self.video(batches.feats.permute(0, 2, 1))
        output_s = feat_s.squeeze()
        output_v = feat_v
        coarse_loss = torch.sum(torch.pow(torch.matmul(output_s, output_v.permute(0, 2, 1)) - batches.sim * 256, 2))
        if self.training:
            return coarse_loss
        return torch.matmul(feat_s, feat_v.permute(0, 2, 1))
