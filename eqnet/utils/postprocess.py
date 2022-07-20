import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

def detect_peaks(scores, vmin=0.3, kernel=101, K=0):

    smax = nn.functional.max_pool2d(scores, (kernel, 1), stride=1, padding=(kernel // 2, 0))
    keep = (smax == scores).float()
    scores = scores * keep

    batch, chn, nt, ns = scores.size()
    scores = torch.transpose(scores, 2, 3)
    if K == 0:
        K = max(round(nt * 10 / 3000.0), 3)
    topk_scores, topk_inds = torch.topk(scores[:, 1:, :, :].view(batch, chn - 1, ns, -1), K)
    topk_inds = topk_inds % nt

    return topk_scores.detach().cpu(), topk_inds.detach().cpu()


def extract_picks(
    topk_index,
    topk_score,
    file_name=None,
    begin_time=None,
    station_name=None,
    vmin=0.3,
    dt=0.01,
    phases=["P", "S"],
    **kwargs,
):
    """Extract picks from prediction results.
    Args:
        topk_scores ([type]): [Nb, Nc, Ns, Ntopk] "batch, channel, station, topk"
        file_names ([type], optional): [Nb]. Defaults to None.
        station_names ([type], optional): [Ns]. Defaults to None.
        t0 ([type], optional): [Nb]. Defaults to None.
        config ([type], optional): [description]. Defaults to None.

    Returns:
        picks [type]: {file_name, station_name, pick_time, pick_prob, pick_type}
    """

    batch, nch, nst, ntopk = topk_score.shape
    assert nch == len(phases)
    picks = []
    if isinstance(dt, float):
        dt = [dt for i in range(batch)]
    else:
        dt = [dt[i].item() for i in range(batch)]
    if ("begin_channel_index" in kwargs) and (kwargs["begin_channel_index"] is not None):
        begin_channel_index = [x.item() for x in kwargs["begin_channel_index"]]
    else:
        begin_channel_index = [0 for i in range(batch)]
    if ("begin_time_index" in kwargs) and (kwargs["begin_time_index"] is not None):
        begin_time_index = [x.item() for x in kwargs["begin_time_index"]]
    else:
        begin_time_index = [0 for i in range(batch)]
    # raise
    for i in range(batch):
        picks_per_file = []
        if file_name is None:
            file_i = f"{i:04d}"
        else:
            file_i = file_name[i] if isinstance(file_name[i], str) else file_name[i].decode()

        if begin_time is None:
            begin_i = "1970-01-01T00:00:00.000"
        else:
            begin_i = begin_time[i] if isinstance(begin_time[i], str) else begin_time[i].decode()
            if len(begin_i) == 0:
                begin_i = "1970-01-01T00:00:00.000"
        begin_i = datetime.fromisoformat(begin_i)

        for j in range(nch):
            for k in range(nst):
                if station_name is None:
                    station_i = f"{k + begin_channel_index[i]:04d}"
                else:
                    station_i = station_name[k] if isinstance(station_name[k], str) else station_name[k].decode()

                for index, score in zip(topk_index[i, j, k], topk_score[i, j, k]):
                    if score > vmin:
                        pick_time = (begin_i + timedelta(seconds=index.item() * dt[i])).isoformat(timespec="milliseconds")
                        picks_per_file.append(
                            {
                                "file_name": file_i,
                                "station_name": station_i,
                                "phase_index": index.item() + begin_time_index[i],
                                "phase_score": f"{score.item():.3f}",
                                "phase_type": phases[j],
                                "phase_time": pick_time,
                                "dt": dt[i],
                            }
                        )

        picks.append(picks_per_file)
    return picks


def merge_picks(raw_folder="picks_phasenet", merged_folder=None):

    in_path = Path(raw_folder)

    rename_folder = False
    if merged_folder is None:
        out_path = Path(raw_folder + "_merged")
        rename_folder = True
    else:
        out_path = Path(merged_folder)

    if not out_path.exists():
        out_path.mkdir()

    files = in_path.glob("*_*_*.csv")

    file_group = defaultdict(list)
    for file in files:
        file_group[file.stem.split("_")[0]].append(file) ## event_id

    num_picks = 0
    for k in tqdm(file_group, desc=f"{out_path}"):
        picks = []
        for i, file in enumerate(sorted(file_group[k])):
            with open(file, "r") as f:
                tmp = f.readlines()
                if i == 0:
                    picks.extend(tmp)
                else:
                    picks.extend(tmp[1:]) ## wihout header
        with open(out_path.joinpath(f"{k}.csv"), "w") as f:
            f.writelines(picks)
        num_picks += len(picks)

    if rename_folder:
        in_path.rename(raw_folder + "_raw")
        out_path.rename(raw_folder)

    print(f"Number of picks: {num_picks}")
    return 0