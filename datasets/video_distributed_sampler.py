# -*- coding: utf-8 -*-
"""
Video-aware distributed sampler (single-node multi-GPU friendly).

Goal:
- Group by video_id (cache friendly)
- Assign whole videos to each rank (minimize cross-rank video mixing)
- Keep sequential order inside each video (frame order)
- Pad to equal steps across ranks to avoid DDP hang

Usage (main.py):
    from video_distributed_sampler import VideoDistributedSampler
    sampler_train = VideoDistributedSampler(dataset_train, batch_size=args.batch_size, shuffle=True, seed=args.seed)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Iterator, Optional

import torch
from torch.utils.data import Sampler

try:
    import torch.distributed as dist
except Exception:
    dist = None


class VideoDistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        assert self.batch_size > 0, "batch_size must be > 0"

        if num_replicas is None:
            if dist is not None and dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist is not None and dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self._vid_to_indices: Dict[int, List[int]] = {}
        self._video_keys: List[int] = []
        self._rebuild_video_groups()

        # computed each epoch in __iter__
        self._num_samples = 0

    def _rebuild_video_groups(self) -> None:
        """
        Build mapping: video_key -> dataset indices
        - video_key uses real video_id for video data.
        - For video_id == -1 (imgnet_det), we use pseudo video_key = -img_id-1, so each image is its own group.
        """
        ds = self.dataset
        assert hasattr(ds, "ids"), "dataset must have .ids (COCO image id list)"
        assert hasattr(ds, "coco"), "dataset must have .coco (pycocotools coco api)"

        vid_to_indices = defaultdict(list)

        for ds_idx, img_id in enumerate(ds.ids):
            info = ds.coco.loadImgs(img_id)[0]
            vid = info.get("video_id", -1)
            if vid == -1:
                vid = -int(img_id) - 1  # pseudo unique group
            vid_to_indices[int(vid)].append(int(ds_idx))

        # Make sure indices inside each video are ordered by image id (frame order)
        for vid, idxs in vid_to_indices.items():
            idxs.sort(key=lambda i: ds.ids[i])

        self._vid_to_indices = dict(vid_to_indices)
        self._video_keys = list(self._vid_to_indices.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        # Step 1: get a video order for this epoch (shuffle with balance)
        rng = random.Random(self.seed + self.epoch)

        vids = list(self._video_keys)
        if self.shuffle:
            # shuffle but still make length balancing stable: random tie-break under length sorting
            vids.sort(key=lambda v: (-len(self._vid_to_indices[v]), rng.random()))
        else:
            # deterministic order: longer first (still helps balance)
            vids.sort(key=lambda v: -len(self._vid_to_indices[v]))

        # Step 2: greedy bin packing by number of frames
        bins: List[List[int]] = [[] for _ in range(self.num_replicas)]
        bin_sizes = [0 for _ in range(self.num_replicas)]

        for vid in vids:
            r = min(range(self.num_replicas), key=lambda i: bin_sizes[i])
            bins[r].append(vid)
            bin_sizes[r] += len(self._vid_to_indices[vid])

        # Step 3: flatten indices for my rank (preserve in-video order)
        my_vids = bins[self.rank]
        indices: List[int] = []
        for vid in my_vids:
            indices.extend(self._vid_to_indices[vid])

        # Step 4: pad so that all ranks have the same number of samples (avoid DDP hanging)
        max_len = max(bin_sizes) if bin_sizes else 0
        if self.drop_last:
            # make divisible by batch_size to keep same number of steps when using drop_last=True BatchSampler
            target = int(math.ceil(max_len / self.batch_size) * self.batch_size) if max_len > 0 else 0
        else:
            target = max_len

        if target == 0:
            self._num_samples = 0
            return iter([])

        if len(indices) == 0:
            # extremely unlikely, but keep safe
            indices = [0]

        if len(indices) < target:
            # repeat cyclically
            rep = (target - len(indices))
            indices.extend(indices * (rep // len(indices)) + indices[: (rep % len(indices))])
        else:
            indices = indices[:target]

        self._num_samples = len(indices)
        return iter(indices)

    def __len__(self) -> int:
        # __len__ is used by BatchSampler to compute number of batches.
        # If __iter__ hasn't been called in this epoch yet, we estimate based on current grouping.
        if self._num_samples > 0:
            return self._num_samples

        # estimate: average / max frame count per rank, then round
        sizes = [len(v) for v in self._vid_to_indices.values()]
        if not sizes:
            return 0
        approx_max = int(math.ceil(sum(sizes) / self.num_replicas))
        if self.drop_last:
            approx_max = int(math.ceil(approx_max / self.batch_size) * self.batch_size)
        return approx_max
