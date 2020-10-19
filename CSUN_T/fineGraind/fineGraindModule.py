import torch
from torch import nn

from .pool import build_fp
from .getmap import build_2dmap
from .encode import build_encode
from .predictor import build_predictor
from .fineloss import build_fineloss

def modify_mask(input_mask, start):
    output_mask = input_mask
    output_mask[0:start, :] = 0
    return output_mask


class fine(nn.Module):
    def __init__(self):
        super(fine, self).__init__()
        self.fp = build_fp()
        self.f_2dmap = build_2dmap()
        self.query_encode = build_encode()
        self.predictor = build_predictor(self.f_2dmap.mask2d)
        self.fineLoss = build_fineloss(self.f_2dmap.mask2d)


    def forward(self, batches, ious2d=None, start=None):
        feats = self.fp(batches.feats)
        map_2d = self.f_2dmap(feats)
        map_2d = self.query_encode(batches.queries, batches.wordlens, map_2d)
        scores2d = self.predictor(map_2d)
        if self.training:
            return self.fineLoss(scores2d, ious2d)
        if start==None:
            return scores2d.sigmoid_() * self.f_2dmap.mask2d
        else:
            return scores2d.sigmoid_() * modify_mask(self.f_2dmap.mask2d, start)
