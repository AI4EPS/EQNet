# sample a batch of data with different number of stations
# inspired by GroupedBatchSampler from Torchvision https://github.com/pytorch/vision/blob/bb3aae7b2543637191ad9c810f082eae622534b8/references/detection/group_by_aspect_ratio.py

import os
import numpy as np
import torch
import torch.distributed as dist
import math
from torch.utils.data.sampler import Sampler, BatchSampler
from glob import glob
from collections import defaultdict
from itertools import chain, repeat

import datasets

class StationSampler(BatchSampler):
    '''
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    
    :param sampler(Sampler): Base sampler.
    :param group_ids(list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
    :param batch_size(int): Size of mini-batch.
    '''
    def __init__(self, sampler, group_ids, batch_size, drop_last=True):
        if not isinstance(sampler, Sampler):
            raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size
        # self.drop_last = drop_last
        self.count = np.unique(self.group_ids, return_counts=True)[1]
        if len(group_ids[group_ids==0]) > 0:
            self.count = self.count[1:]
        self.count = self.count.sum()

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        expected_num_batches = len(self)
        num_batches = 0
        for idx in self.sampler:
            if num_batches < expected_num_batches:
                group_id = self.group_ids[idx]
                # group_id = 0 means that the sample does not belong to any group
                if group_id != 0:
                    buffer_per_group[group_id].append(idx)
                    samples_per_group[group_id].append(idx)
                    if len(buffer_per_group[group_id]) == self.batch_size:
                        yield buffer_per_group[group_id]
                        num_batches += 1
                        del buffer_per_group[group_id]
                assert len(buffer_per_group[group_id]) < self.batch_size

        # if not self.drop_last:
        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        # print(f"rank{int(os.environ['RANK'])}: num_batches: {num_batches}, len(self): {len(self)}")
        num_remaining = expected_num_batches - num_batches
        # use while instead of if to check num_remaining > 0
        # so that we can pad gaps more than number of groups
        # these gaps may appear when we drop some groups
        while num_remaining > 0:
            # for the remaining batches, take first the buffers with the largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                # group_id = 0 means that the sample does not belong to any group
                # we don't want to return these samples
                if group_id == 0:
                    continue
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        try:
            # world_size = os.environ["WORLD_SIZE"]
            world_size = dist.get_world_size()
        except:
            world_size = 1
        return int((self.count/int(world_size)) // self.batch_size)

def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)    
    
def create_groups(dataset, num_stations_list=[5, 10, 20], is_pad=False):
    '''
    create groups of data with different number of stations
    '''
    group_ids = []
    for data in dataset:
        num_stations = data["station_location"].shape[0]
        num_stations_list = np.array(sorted(num_stations_list))
        if num_stations < 5:
            group_id = 0
        elif is_pad:
            if num_stations >= num_stations_list[-1]:
                group_id = num_stations_list[-1]
            else:
                group_id = num_stations_list[num_stations_list>=num_stations][0]
        else:
            if num_stations < num_stations_list[0]:
                group_id = 0
            else:
                group_id = num_stations_list[num_stations_list<=num_stations][-1]
            
        group_ids.append(group_id)
    group_ids=np.array(group_ids)
    counts = np.unique(group_ids, return_counts=True)[1]
    if len(counts) > len(num_stations_list):
        counts = counts[1:]
    # print group and counts
    print(f"groups: {num_stations_list}, counts: {counts}, sum: {counts.sum()}")
    return group_ids


def cut_reorder_keys(example, num_stations_list=[5, 10, 20], is_pad=False, is_train=True, padding_mode="zeros"):
    num_stations = example["station_location"].shape[0]
    num_stations_list = np.array(sorted(num_stations_list))
    if is_train and num_stations < 5:
        return example
    if is_pad:
        if num_stations >= num_stations_list[-1]:
            group_id = num_stations_list[-1]
            cut = np.random.permutation(num_stations)[:group_id]
            for keys in example.keys():
                example[keys] = example[keys][cut]
        else:
            group_id = num_stations_list[num_stations_list>=num_stations][0]
            # randomly choose a subset of stations to pad the batch
            # so that the number of stations is equal to group_id
            pad_size = group_id - num_stations
            # patch the data
            if pad_size > 0:
                if padding_mode == "random":
                    pad_indices = np.random.choice(num_stations, pad_size, replace=True)
                    for keys in example.keys():
                        example[keys] = torch.cat([example[keys], example[keys][pad_indices]], dim=0)
                elif padding_mode == "zeros":
                    # zero padding
                    for keys in example.keys():
                        example[keys] = torch.cat([example[keys], torch.zeros(pad_size, *example[keys].shape[1:])], dim=0)
    else:
        if num_stations < num_stations_list[0]:
            return reorder_keys(example)
        else:
            group_id = num_stations_list[num_stations_list<=num_stations][-1]
            cut = np.random.permutation(num_stations)[:group_id]
            for keys in example.keys():
                example[keys] = example[keys][cut]

    return reorder_keys(example)
        
        
def reorder_keys(example):
    example["data"] = example["data"].permute(1,2,0).contiguous()
    example["phase_pick"] = example["phase_pick"].permute(1,2,0).contiguous()
    example["event_center"] = example["event_center"].permute(1,0).contiguous()
    example["event_location"] = example["event_location"].permute(1,2,0).contiguous()
    example["event_location_mask"] = example["event_location_mask"].permute(1,0).contiguous()
    example["station_location"] = example["station_location"].contiguous()
    return example