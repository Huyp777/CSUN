import os
from os.path import join, exists
import h5py
import numpy as np
from terminaltables import AsciiTable
from tqdm import tqdm
import logging

import torch
import torchtext
from torch.functional import F

def embedding(sentence, vocabs=[], embedders=[]):
    if len(vocabs) == 0:
        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat(
            [vocab.vectors, torch.zeros(1, vocab.dim)],
            dim=0
        )
        vocabs.append(vocab)

    if len(embedders) == 0:
        embedder = torch.nn.Embedding.from_pretrained(vocab.vectors)
        embedders.append(embedder)

    vocab, embedder = vocabs[0], embedders[0]
    word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) \
                              for w in sentence.split()], dtype=torch.long)
    return embedder(word_idxs)

def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()
    scores = score2d[grids[:,0], grids[:,1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores

def moment_to_iou2d(moment, num_clips, duration):
    iou2d = torch.ones(num_clips, num_clips)
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d


def avgfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips + 1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips - 1)
    # To prevent a empty selection, check the idxs
    meanfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i + 1]
        if s < e:
            meanfeats.append(feats[s:e].mean(dim=0))
        else:
            meanfeats.append(feats[s])
    return torch.stack(meanfeats)


def video2feats(feat_file, vids, num_pre_clips, dataset_name):
    assert exists(feat_file)
    vid_feats = {}
    with h5py.File(feat_file, 'r') as f:
        for vid in vids:
            if dataset_name == "activitynet":
                feat = f[vid]['c3d_features'][:]
            else:
                feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat),dim=1)
            vid_feats[vid] = avgfeats(feat, num_pre_clips)
    return vid_feats

def nms(moments, scores, topk, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True

    return moments[~suppressed]


def evaluate(dataset, predictions, nms_thresh, recall_metrics=(1, 5), iou_metrics=(0.1, 0.3, 0.5, 0.7)):
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("tan.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))

    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    table = [['Rank@{},mIoU@{}'.format(i, j) \
              for i in recall_metrics for j in iou_metrics]]

    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)

    num_clips = predictions[0].shape[-1]
    for idx, score2d in tqdm(enumerate(predictions)):
        duration = dataset.get_duration(idx)
        moment = dataset.get_moment(idx)
        candidates, scores = score2d_to_moments_scores(score2d, num_clips, duration)
        moments = nms(candidates, scores, topk=recall_metrics[-1], thresh=nms_thresh)

        for i, r in enumerate(recall_metrics):
            mious = iou(moments[:r], dataset.get_moment(idx))
            bools = mious[:, None].expand(r, num_iou_metrics) > iou_metrics
            recall_x_iou[i] += bools.any(dim=0)

    recall_x_iou /= len(predictions)

    table.append(['{:.02f}'.format(recall_x_iou[i][j] * 100) \
                  for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics * num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)
