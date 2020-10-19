import os
from os.path import join, dirname
import json
import numpy as np
import torch

from ..utils.tools import video2feats, moment_to_iou2d, embedding
from ..coarse.utils import iou

KERNEL_SIZE = [15, 8, 8]
STRIDE = [1, 2, 4]


class ActivityNetDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, root, feat_file, num_pre_clips, num_clips, pre_query_size):
        super(ActivityNetDataset, self).__init__()

        with open(ann_file, 'r') as f:
            annos = json.load(f)
        self.annos = []
        ticks = []
        self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="activitynet")
        N, offset, stride = num_clips, 0, 1
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        for c in KERNEL_SIZE:
            for _ in range(c):
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
            stride *= 2
        tick0 = []
        tick1 = []
        for _ in range(N):
            if mask2d[0, _] == 1:
                tick0 += [0]
                tick1 += [_]
        for _ in range(N):
            if mask2d[_, N - 1] == 1:
                tick0 += [_]
                tick1 += [N - 1]
        for vid, anno in annos.items():
            duration = anno['duration']
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
                iou2d = moment_to_iou2d(moment, num_clips, duration)
                query = embedding(sentence)

                sim = np.zeros([1, len(tick0)], dtype=np.float32)
                timestamp0 = np.array(min(timestamp[0], timestamp[1]))
                timestamp1 = np.array(max(timestamp[0], timestamp[1]))
                sim = np.maximum(sim, iou(timestamp0, timestamp1, tick0*duration/num_clips, tick1*duration/num_clips))
                self.annos.append(
                    {
                        'vid': vid,
                        'moment': moment,
                        'iou2d': iou2d,
                        'sentence': sentence,
                        'query': query,
                        'wordlen': query.size(0),
                        'duration': duration,
                        'ticks': ticks,
                        'sim': sim,
                    }
                )

    def __getitem__(self, idx):
        anno = self.annos[idx]
        vid = anno['vid']
        return self.feats[vid], anno['query'], anno['wordlen'], anno['iou2d'], anno['ticks'], anno['sim'], idx

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]['duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['vid']
