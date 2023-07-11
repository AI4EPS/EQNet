# run it under the tests folder

import time
import numpy as np
from collections import defaultdict, deque

import datasets
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler

import sys
sys.path.append("../")
from eqnet.utils.station_sampler import StationSampler, cut_reorder_keys, create_groups

time1 = time.time()
quakeflow_nc = datasets.load_dataset("../eqnet/data/quakeflow_nc.py", split="test", name="event_test")
quakeflow_nc = quakeflow_nc.with_format("torch")
time2 = time.time()
print("Time to load dataset: ", time2-time1)
is_pad = True
print("is_pad: ", is_pad)
group_ids = create_groups(quakeflow_nc, num_stations_list=[5,10,20], is_pad=is_pad)
quakeflow_nc = quakeflow_nc.map(lambda x: cut_reorder_keys(x, num_stations_list=[5,10,20], is_pad=is_pad))

train_sampler = torch.utils.data.SequentialSampler(quakeflow_nc)
train_batch_sampler = StationSampler(train_sampler, group_ids, 16, drop_last=True)
data_loader_stations = DataLoader(
    quakeflow_nc,
    batch_sampler=train_batch_sampler,
    num_workers=4,
)
time3 = time.time()
print("Time to create dataloader: ", time3-time2)
iter_3 = 0
for batch in data_loader_stations:
    if iter_3 and iter_3%4==0:
        time_worker_1 = time.time()
        print(f"\n\nTime between workers: {time_worker_1-time_worker_2}\n")
    print(f"\nDataloader test{iter_3}\n")
    print(batch.keys())
    for key in batch.keys():
        print(key, batch[key].shape, batch[key].dtype)
    iter_3 += 1
    if iter_3%4==0:
        time_worker_2 = time.time()

print("Number of iterations: ", iter_3)
time4 = time.time()
print("Time to iterate through dataloader: ", time4-time3)