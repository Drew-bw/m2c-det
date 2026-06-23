# Copyright (c) OpenMMLab. All rights reserved.
import os
import gc
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from mmengine.fileio import get_local_path, join_path
from mmengine.utils import is_abs
from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class OVODDataset(BaseDetDataset):
    """object detection and visual grounding dataset."""

    def __init__(self,
                 *args,
                 cache: bool = True,
                 dataset_mode: list = ('VG', 'VG', 'OD'),
                 use_index: list = (0, 1, 2),
                 data_root: str = '',
                 **kwargs) -> None:
        self.cache = cache
        self.dataset_mode = dataset_mode
        self.use_index = use_index
        super().__init__(*args, data_root=data_root, **kwargs)
        assert self.return_classes is True

    @staticmethod
    def load_dataset_cache_file(path: Path) -> Dict:
        """Load an Ultralytics *.cache dictionary from path."""
        gc.disable()
        cache = np.load(str(path), allow_pickle=True).item()  # load dict
        gc.enable()
        return cache

    @staticmethod
    def get_hash(paths) -> str:
        """Return a single hash value of a list of paths (files or dirs)."""
        size = 0
        for p in paths:
            try:
                size += os.stat(p).st_size
            except OSError:
                continue
        h = __import__("hashlib").sha256(str(size).encode())  # hash sizes
        h.update("".join(paths).encode())  # hash paths
        return h.hexdigest()  # return hash

    @staticmethod
    def save_dataset_cache_file(path: Path, x: Dict):
        """Save an Ultralytics dataset *.cache dictionary x to path."""
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        with open(str(path), "wb") as file:  # context manager here fixes windows async np.save bug
            np.save(file, x)

    def cache_labels_od(self, local_file, data_prefix):
        data = {
            "infos": [],
            'category_freq': defaultdict(int)
        }

        with open(local_file) as f:
            annotations = json.load(f)
            f.close()

        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        # new label id
        texts = [0] * len(annotations["categories"])
        for categories in annotations["categories"]:
            texts[categories["id"] - 1] = categories["name"].split('/')

        for img_id, anns in tqdm(img_to_anns.items(), desc=f"Reading annotations {local_file}"):
            data_info = {}
            img = images[f"{img_id:d}"]
            im_file = Path(data_prefix['img']) / img["file_name"]
            if not im_file.exists():
                print('image not found:', im_file)
                continue

            data_info['img_path'] = im_file
            data_info['height'] = int(img["height"])
            data_info['width'] = int(img["width"])

            instances = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                x1, y1, w, h = ann['bbox']
                if w < 1 or h < 1:
                    continue
                instance = {
                    'bbox': [x1, y1, x1 + w, y1 + h],
                    'bbox_label': ann["category_id"] - 1,
                    'ignore_flag': 0
                }
                assert ann["category_id"] >= 1
                instances.append(instance)

                text = texts[instance['bbox_label']]
                for t in text:
                    t = t.strip()
                    data['category_freq'][t] += 1

            data_info['text'] = texts
            data_info['img_id'] = img_id
            data_info['instances'] = instances
            data['infos'].append(data_info)
        return data

    def cache_labels_vg(self, local_file, data_prefix):
        data = {
            "infos": [],
            'category_freq': defaultdict(int)
        }
        with open(local_file) as f:
            annotations = json.load(f)
            f.close()

        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_id, anns in tqdm(img_to_anns.items(), desc=f"Reading annotations {local_file}"):
            data_info = {}
            img = images[f"{img_id:d}"]
            im_file = Path(data_prefix['img']) / img["file_name"]
            if not im_file.exists():
                continue

            data_info['img_path'] = im_file
            data_info['height'] = int(img["height"])
            data_info['width'] = int(img["width"])

            instances = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                x1, y1, w, h = ann['bbox']
                if w < 1 or h < 1:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                data['category_freq'][cat_name.strip()] += 1

                instance = {
                    'bbox': [x1, y1, x1 + w, y1 + h],
                    'bbox_label': cls,
                    'ignore_flag': 0
                }
                instances.append(instance)

            data_info['text'] = texts
            data_info['instances'] = instances
            data_info['img_id'] = img_id
            data['infos'].append(data_info)
        return data

    def load_data_list(self) -> List[dict]:
        out_data_list = []
        category_freq = defaultdict(int)
        for idx in self.use_index:
            with get_local_path(
                    self.ann_file[idx], backend_args=self.backend_args) as local_path:
                cache_path = Path(local_path).with_suffix(".cache")
                try:
                    cache, _ = self.load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file # identical hash
                except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
                    if self.dataset_mode[idx] == 'OD':
                        cache = self.cache_labels_od(local_path, self.data_prefix[idx])
                    else:
                        cache = self.cache_labels_vg(local_path, self.data_prefix[idx])
                    cache["hash"] = self.get_hash(local_path)
                    self.save_dataset_cache_file(cache_path, cache)
                out_data = cache["infos"]
                out_data_list.extend(out_data)
                category_freq.update(cache["category_freq"])

        threshold = min(max(category_freq.values()), 100)
        self.neg_texts = [k for k, v in category_freq.items() if v >= threshold]
        return out_data_list

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        data_info['neg_text'] = self.neg_texts
        return self.pipeline(data_info)

    def _join_prefix(self):
        for i, ann_file in enumerate(self.ann_file):
            if ann_file and not is_abs(ann_file) and self.data_root[i]:
                self.ann_file[i] = join_path(self.data_root[i], ann_file)
            for data_key, prefix in self.data_prefix[i].items():
                if not isinstance(prefix, str):
                    raise TypeError('prefix should be a string, but got '
                                    f'{type(prefix)}')
                if not is_abs(prefix) and self.data_root[i]:
                    self.data_prefix[i][data_key] = join_path(self.data_root[i], prefix)
                else:
                    self.data_prefix[i][data_key] = prefix

    # def __len__(self) -> int:
    #     """Get the length of filtered dataset and automatically call
    #     ``full_init`` if the  dataset has not been fully init.
    #
    #     Returns:
    #         int: The length of filtered dataset.
    #     """
    #     if self.serialize_data:
    #         return len(self.data_address)
    #     else:
    #         return len(self.data_list)