"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "activitynet_train":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },
        "activitynet_val":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },
        "activitynet_test":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["video_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
        )
        if "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))
