# run it under the tests folder

import os
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

# args
is_pad = True
is_print = False
num_stations_list = [10]


time1 = time.time()
quakeflow_nc = datasets.load_dataset("../eqnet/data/quakeflow_nc.py", split="test", name="event_test")
quakeflow_nc = quakeflow_nc.with_format("torch")
time2 = time.time()
print("Time to load dataset: ", time2-time1)

print("is_pad: ", is_pad)
try:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    num_proc = 4
    gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(gpu)
    dist_backend = "nccl"
    print(f"| distributed init (rank {rank}): env://", flush=True)
    torch.distributed.init_process_group(
        backend=dist_backend, init_method="env://", world_size=world_size, rank=rank
    )
    torch.distributed.barrier()
except:
    world_size = None 
    num_proc = None
group_ids = create_groups(quakeflow_nc, num_stations_list=num_stations_list, is_pad=is_pad)

if world_size is not None:
    if gpu > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    quakeflow_nc = quakeflow_nc.map(lambda x: cut_reorder_keys(x, num_stations_list=num_stations_list, is_pad=is_pad, is_train=True), num_proc=num_proc)
    if gpu == 0:
        print("Mapping finished, loading results from main process")
        torch.distributed.barrier()
else:
    print("Mapping dataset")
    quakeflow_nc = quakeflow_nc.map(lambda x: cut_reorder_keys(x, num_stations_list=num_stations_list, is_pad=is_pad, is_train=True), num_proc=num_proc)

if world_size is not None:
    train_sampler = torch.utils.data.distributed.DistributedSampler(quakeflow_nc)
else:
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
print("Number of iterations: ", len(data_loader_stations))
for batch in data_loader_stations:
    if iter_3 and iter_3%4==0:
        time_worker_1 = time.time()
        print(f"\n\nTime between workers: {time_worker_1-time_worker_2}\n")
    print(f"\nDataloader test{iter_3}\n")
    print(batch.keys())
    for key in batch.keys():
        if is_print:
            print(key, batch[key].shape, batch[key].dtype)
    iter_3 += 1
    if iter_3%4==0:
        time_worker_2 = time.time()

print("Number of iterations: ", iter_3)
time4 = time.time()
print("Time to iterate through dataloader: ", time4-time3)