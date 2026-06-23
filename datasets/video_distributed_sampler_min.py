# -*- coding: utf-8 -*-
"""
Video-aware Distributed Sampler (TRUNCATE version, no repeat padding)

特点：
- 按 video_id 分组，将整段视频尽量分配给同一 rank（缓存友好）
- 保持每个视频内部帧顺序（按 img_id 排序）
- 为了 DDP 不 hang：所有 rank 输出长度一致
- 关键：不做 padding 重复样本；统一截断到共同 target 长度（更接近单卡分布）

建议训练使用：
    sampler = VideoDistributedSamplerTruncate(dataset_train, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
每个 epoch 记得：
    sampler.set_epoch(epoch)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Iterator, Optional

from torch.utils.data import Sampler

try:
    import torch.distributed as dist
except Exception:
    dist = None


class VideoDistributedSamplerTruncate(Sampler[int]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        truncate_mode: str = "min",  # "min" or "avg"
    ) -> None:
        """
        truncate_mode:
          - "min": target = min(rank_sizes)，几乎不会 hang，截断最保守
          - "avg": target = floor(avg(rank_sizes))，但仍会取 min(target, min_size) 保证每 rank 够用
        """
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
        self.truncate_mode = str(truncate_mode)
        assert self.truncate_mode in ("min", "avg")

        self.epoch = 0

        self._vid_to_indices: Dict[int, List[int]] = {}
        self._video_keys: List[int] = []
        self._rebuild_video_groups()

        self._num_samples = 0

    def _rebuild_video_groups(self) -> None:
        ds = self.dataset
        assert hasattr(ds, "ids"), "dataset must have .ids (COCO image id list)"
        assert hasattr(ds, "coco"), "dataset must have .coco (pycocotools coco api)"

        vid_to_indices = defaultdict(list)

        for ds_idx, img_id in enumerate(ds.ids):
            info = ds.coco.loadImgs(img_id)[0]
            vid = info.get("video_id", -1)
            if vid == -1:
                # imgnet_det 或单图：每张图一个 pseudo video group
                vid = -int(img_id) - 1
            vid_to_indices[int(vid)].append(int(ds_idx))

        # 视频内部保持帧序：按 img_id（你现在的 ref 逻辑也依赖 img_id 的“时间序”假设）
        for vid, idxs in vid_to_indices.items():
            idxs.sort(key=lambda i: ds.ids[i])

        self._vid_to_indices = dict(vid_to_indices)
        self._video_keys = list(self._vid_to_indices.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)

        # 1) 准备视频列表：为了更平衡，先按长度排序（长的先分配），同长度随机打散
        vids = list(self._video_keys)
        if self.shuffle:
            vids.sort(key=lambda v: (-len(self._vid_to_indices[v]), rng.random()))
        else:
            vids.sort(key=lambda v: -len(self._vid_to_indices[v]))

        # 2) greedy bin packing：把视频分到各 rank（按帧数尽量均衡）
        bins: List[List[int]] = [[] for _ in range(self.num_replicas)]
        bin_sizes = [0 for _ in range(self.num_replicas)]

        for vid in vids:
            r = min(range(self.num_replicas), key=lambda i: bin_sizes[i])
            bins[r].append(vid)
            bin_sizes[r] += len(self._vid_to_indices[vid])

        # 3) 本 rank 的视频：为了长期覆盖均匀，对视频顺序再洗一次（不影响“视频内连续”）
        my_vids = list(bins[self.rank])
        if self.shuffle:
            rng.shuffle(my_vids)

        indices: List[int] = []
        for vid in my_vids:
            indices.extend(self._vid_to_indices[vid])

        if len(indices) == 0:
            self._num_samples = 0
            return iter([])

        # 4) 计算统一 target：所有 rank 截断到同一长度（不 repeat/pad）
        min_len = min(bin_sizes) if bin_sizes else 0
        avg_len = int(sum(bin_sizes) / len(bin_sizes)) if bin_sizes else 0

        if self.truncate_mode == "avg":
            target = min(min_len, avg_len)
        else:
            target = min_len

        if self.drop_last:
            # 保证 batch 对齐：每个 rank 的样本数是 batch_size 的整数倍
            target = (target // self.batch_size) * self.batch_size

        if target <= 0:
            self._num_samples = 0
            return iter([])

        # 5) 截断：不重复样本
        # 为了避免每次总取前 target 导致偏置，再引入一个“窗口起点”
        # 这样不同 epoch 会截取到不同片段（更接近全量覆盖）
        if len(indices) > target:
            max_start = len(indices) - target
            start = rng.randint(0, max_start) if self.shuffle else 0
            indices = indices[start:start + target]
        else:
            # 理论上 indices >= target（因为 target <= min_len <= bin_sizes[rank]）
            indices = indices[:target]

        self._num_samples = len(indices)
        return iter(indices)

    def __len__(self) -> int:
        # 给 BatchSampler 用：每 epoch 的 samples 数
        if self._num_samples > 0:
            return self._num_samples

        # 估计值（在 __iter__ 前被调用时）
        sizes = [len(v) for v in self._vid_to_indices.values()]
        if not sizes:
            return 0
        approx = int(sum(sizes) / self.num_replicas)
        if self.drop_last:
            approx = (approx // self.batch_size) * self.batch_size
        return approx
