import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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
    topk_inds,
    topk_scores,
    file_names=None,
    begin_times=None,
    station_names=None,
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

    batch, nch, nst, ntopk = topk_scores.shape
    assert nch == len(phases)
    picks = []
    if isinstance(dt, float):
        dt = [dt for i in range(batch)]
    else:
        dt = [dt[i].item() for i in range(batch)]
    if ("begin_channel_indexs" in kwargs) and (kwargs["begin_channel_indexs"] is not None):
        begin_channel_indexs = [x.item() for x in kwargs["begin_channel_indexs"]]
    else:
        begin_channel_indexs = [0 for i in range(batch)]
    if ("begin_time_indexs" in kwargs) and (kwargs["begin_time_indexs"] is not None):
        begin_time_indexs = [x.item() for x in kwargs["begin_time_indexs"]]
    else:
        begin_time_indexs = [0 for i in range(batch)]
    # raise
    for i in range(batch):
        picks_per_file = []
        if file_names is None:
            file_name = f"{i:04d}"
        else:
            file_name = file_names[i] if isinstance(file_names[i], str) else file_names[i].decode()

        if begin_times is None:
            begin_time = "1970-01-01T00:00:00.000"
        else:
            begin_time = begin_times[i] if isinstance(begin_times[i], str) else begin_times[i].decode()
            if len(begin_time) == 0:
                begin_time = "1970-01-01T00:00:00.000"
        begin_time = datetime.fromisoformat(begin_time)

        for j in range(nch):
            for k in range(nst):
                if station_names is None:
                    station_name = f"{k + begin_channel_indexs[i]:04d}"
                else:
                    station_name = station_names[k] if isinstance(station_names[k], str) else station_names[k].decode()

                for index, score in zip(topk_inds[i, j, k], topk_scores[i, j, k]):
                    if score > vmin:
                        pick_time = (begin_time + timedelta(seconds=index.item() * dt[i])).isoformat(timespec="milliseconds")
                        picks_per_file.append(
                            {
                                "file_name": file_name,
                                "station_name": station_name,
                                "phase_index": index.item() + begin_time_indexs[i],
                                "phase_score": f"{score.item():.3f}",
                                "phase_type": phases[j],
                                "phase_time": pick_time,
                                "dt": dt[i],
                            }
                        )

        picks.append(picks_per_file)
    return picks