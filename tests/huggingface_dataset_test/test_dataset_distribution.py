# run it under the tests folder

import os
import time
import numpy as np
import matplotlib.pyplot as plt
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
    num_proc = 12
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
        print(f"Rank {rank}: GPU {gpu} waiting for main process to perform mapping")
        torch.distributed.barrier()
    quakeflow_nc = quakeflow_nc.map(lambda x: cut_reorder_keys(x, num_stations_list=num_stations_list, is_pad=is_pad, is_train=True), num_proc=num_proc, desc="cut_reorder_keys")
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

train_batch_sampler = StationSampler(train_sampler, group_ids, 16)
data_loader_stations = DataLoader(
    quakeflow_nc,
    batch_sampler=train_batch_sampler,
    num_workers=4,
)
time3 = time.time()
print("Time to create dataloader: ", time3-time2)
iter_3 = 0
print("Number of iterations: ", len(data_loader_stations))
relative_x = []
relative_y = []
relative_z = []
for batch in data_loader_stations:
    loc = batch["station_location"]
    relative_coords_sta = loc[:, :, None, :] - loc[:, None, :, :] # B, nx, nx, 3
    relative_coords_sta = relative_coords_sta.reshape(-1, 3) # B*nx*nx, 3
    relative_x.append(relative_coords_sta[:, 0])
    relative_y.append(relative_coords_sta[:, 1])
    relative_z.append(relative_coords_sta[:, 2])

relative_x = torch.cat(relative_x, dim=0)
relative_y = torch.cat(relative_y, dim=0)
relative_z = torch.cat(relative_z, dim=0)
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
axes[0].hist(relative_x, bins=200, range=(-100, 100))
axes[0].set_title("Relative x distribution")
axes[1].hist(relative_y, bins=200, range=(-100, 100))
axes[1].set_title("Relative y distribution")
axes[2].hist(relative_z, bins=40, range=(-20, 20))
axes[2].set_title("Relative z distribution")
fig.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95)


figure_dir = "/home/wanghy/tests/EQNet/tests/huggingface_dataset_test/figures"

if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    fig.savefig(f"{figure_dir}/distribution_{local_rank}.png", dpi=300)
else:
    fig.savefig(f"{figure_dir}/distribution.png", dpi=300)

time4 = time.time()
print("Time to iterate through dataloader: ", time4-time3)