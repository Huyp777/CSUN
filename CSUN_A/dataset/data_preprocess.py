import logging
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
import importlib
import bisect
import importlib.util
import sys

num_pre_clips = 256
num_clips = 64
pre_query_size = 300
train_batch_size = 32
test_batch_size = 32
max_epoch = 5
dataset_train = ("activitynet_train", "activitynet_val")
dataset_test = ("activitynet_test")
path = os.path.join(os.path.dirname(__file__), "paths_catalog.py")


def import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module

class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)

@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: torch.tensor
    queries: torch.tensor
    wordlens: torch.tensor
    ticks: torch.tensor
    sim: torch.tensor

    def to(self, device):
        self.feats = self.feats.to(device)
        self.queries = self.queries.to(device)
        self.wordlens = self.wordlens.to(device)
        self.ticks = self.ticks.to(device)
        self.sim = self.sim.to(device)
        return self

class BatchCollator(object):

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        feats, queries, wordlens, ious2d, ticks, sim, idxs = transposed_batch
        return TLGBatch(
            feats=torch.stack(feats).float(),
            queries=pad_sequence(queries).transpose(0, 1),
            wordlens=torch.tensor(wordlens),
            ticks = torch.tensor(ticks),
            sim = torch.tensor(sim),
        ), torch.stack(ious2d), idxs

def build_dataset(dataset_list, dataset_catalog, is_train=True):
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = "ActivityNetDataset"
        args = data["args"]
        args["num_pre_clips"] = num_pre_clips
        args["num_clips"] = num_clips
        args["pre_query_size"] = pre_query_size
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    if not is_train:
        return datasets

    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    return [dataset]

def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_batch_data_sampler(dataset, sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )
    return batch_sampler

def make_data_loader(is_train=True, is_for_period=False):
    if is_train:
        batch_size = train_batch_size
        shuffle = True
    else:
        batch_size = test_batch_size
        shuffle = False

    paths_catalog = import_file(
        "paths_catalog", path, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = dataset_train if is_train else dataset_test
    datasets = build_dataset(dataset_list, DatasetCatalog, is_train=is_train or is_for_period)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle)
        batch_sampler = make_batch_data_sampler(dataset, sampler, batch_size)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=16,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(),
        )
        data_loaders.append(data_loader)
    if is_train or is_for_period:
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders

