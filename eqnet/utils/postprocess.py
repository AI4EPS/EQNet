import os
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def detect_peaks(scores, vmin=0.3, kernel=101, stride=1, K=0, dt=0.01):
    nb, nc, nt, nx = scores.shape
    pad = kernel // 2
    smax = F.max_pool2d(scores, (kernel, 1), stride=(stride, 1), padding=(pad, 0))[:, :, :nt, :]
    scores = scores * (smax == scores)
    # sdiff = torch.zeros_like(scores)
    # sdiff[:, :, 1:, :] = scores[:, :, 1:, :] - scores[:, :, :-1, :]
    # keep = (smax == scores) & (sdiff != 0.0)
    # scores = scores * keep.float()

    batch, chn, nt, ns = scores.size()
    scores = torch.transpose(scores, 2, 3)  # [nb, nc, nt, nx] -> [nb, nc, nx, nt]
    if K == 0:
        K = max(round(nt / (30.0 / dt) * 10.0), 3)  # maximum 10 picks per 30 seconds
    if chn == 1:
        topk_scores, topk_inds = torch.topk(scores, K)
    else:
        topk_scores, topk_inds = torch.topk(scores[:, 1:, :, :].view(batch, chn - 1, ns, -1), K)
    # topk_inds = topk_inds % nt

    topk_scores = topk_scores.detach().cpu()
    topk_inds = topk_inds.detach().cpu()

    ## filtering close picks with the same scores
    topk_inds, idx = torch.sort(topk_inds, dim=-1)
    topk_scores = topk_scores.gather(-1, idx)
    topk_diff = topk_inds[..., 1:] - topk_inds[..., :-1]
    mask = torch.ones_like(topk_inds)
    mask[..., 1:] = (topk_diff > (kernel / stride / 2)).float()
    topk_scores = topk_scores * mask

    return topk_scores, topk_inds


def extract_picks(
    topk_index,
    topk_score,
    file_name=None,
    begin_time=None,
    station_id=None,
    phases=["P", "S"],
    vmin=0.3,
    dt=0.01,
    polarity_score=None,
    waveform=None,
    window_amp=[10, 5],
    polarity_scale=1,
    **kwargs,
):
    """Extract picks from prediction results.
    Args:
        topk_scores ([type]): [Nb, Nc, Ns, Ntopk] "batch, channel, station, topk"
        file_names ([type], optional): [Nb]. Defaults to None.
        station_ids ([type], optional): [Ns]. Defaults to None.
        t0 ([type], optional): [Nb]. Defaults to None.
        config ([type], optional): [description]. Defaults to None.

    Returns:
        picks [type]: {file_name, station_id, pick_time, pick_prob, pick_type}
    """

    batch, nch, nst, ntopk = topk_score.shape
    # assert nch == len(phases)

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

    if waveform is not None:
        waveform_amp = torch.max(torch.abs(waveform), dim=1)[0]
        # waveform_amp = torch.sqrt(torch.mean(waveform ** 2, dim=1))

        if len(window_amp) == 1:
            window_amp = [window_amp[0] for i in range(len(phases))]

    for i in range(batch):
        picks_per_file = []
        if file_name is None:
            file_i = f"{i:04d}"
        else:
            file_i = file_name[i]

        if begin_time is None:
            begin_i = "1970-01-01T00:00:00.000"
        else:
            begin_i = begin_time[i]
            if len(begin_i) == 0:
                begin_i = "1970-01-01T00:00:00.000"
        begin_i = datetime.fromisoformat(begin_i.rstrip("Z"))

        for j in range(nch):
            if waveform is not None:
                window_amp_i = int(window_amp[j] / dt[i])  ## index of window

            for k in range(nst):
                if station_id is None:
                    station_i = f"{k + begin_channel_index[i]:04d}"
                else:
                    station_i = station_id[k][i]

                topk_index_ijk, ii = torch.sort(topk_index[i, j, k])
                topk_score_ijk = topk_score[i, j, k][ii]

                idx = topk_score_ijk > vmin
                topk_index_ijk = topk_index_ijk[idx]
                topk_score_ijk = topk_score_ijk[idx]

                # for ii, (index, score) in enumerate(zip(topk_index[i, j, k], topk_score[i, j, k])):
                for ii, (index, score) in enumerate(zip(topk_index_ijk, topk_score_ijk)):
                    # if score > vmin:
                    pick_index = index.item() + begin_time_index[i]
                    pick_time = (begin_i + timedelta(seconds=index.item() * dt[i])).isoformat(timespec="milliseconds")
                    pick_dict = {
                        # "file_name": file_i,
                        "station_id": station_i,
                        "phase_index": pick_index,
                        "phase_time": pick_time,
                        "phase_score": f"{score.item():.3f}",
                        "phase_type": phases[j],
                        "dt_s": dt[i],
                    }

                    if polarity_score is not None:
                        # pick_dict["phase_polarity"] = (
                        #     f"{(polarity_score[i, 1, index.item()//polarity_scale, k].item() - polarity_score[i, 2, index.item()//polarity_scale, k].item()):.3f}"
                        # )
                        score = polarity_score[i, 1, :, k] - polarity_score[i, 2, :, k]
                        # score = (polarity_score[i, 0, :, k] - 0.5) * 2.0
                        score = score[max(0, index.item() // polarity_scale - 3) : index.item() // polarity_scale + 3]
                        idx = torch.argmax(torch.abs(score))
                        pick_dict["phase_polarity"] = round(score[idx].item(), 3)

                    if waveform is not None:
                        j1 = topk_index_ijk[ii]
                        j2 = (
                            min(j1 + window_amp_i, topk_index_ijk[ii + 1])
                            if ii < len(topk_index_ijk) - 1
                            else j1 + window_amp_i
                        )
                        pick_dict["phase_amplitude"] = f"{torch.max(waveform_amp[i, j1:j2, k]).item():.3e}"

                    picks_per_file.append(pick_dict)

        picks.append(picks_per_file)
    return picks


def extract_events(
    topk_index,
    topk_score,
    file_name=None,
    begin_time=None,
    station_id=None,
    vmin=0.3,
    dt=0.01,
    event_scale=16,
    event_time=None,
    waveform=None,
    window_amp=[2, 2],
    VPVS_RATIO=1.73,
    **kwargs,
):
    """Extract picks from prediction results.
    Args:
        topk_scores ([type]): [Nb, Nc, Ns, Ntopk] "batch, channel, station, topk"
        file_names ([type], optional): [Nb]. Defaults to None.
        station_ids ([type], optional): [Ns]. Defaults to None.
        t0 ([type], optional): [Nb]. Defaults to None.
        config ([type], optional): [description]. Defaults to None.

    Returns:
        picks [type]: {file_name, station_id, pick_time, pick_prob, pick_type}
    """

    batch, nch, nst, ntopk = topk_score.shape
    # assert nch == len(phases)

    events = []
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

    for i in range(batch):
        events_per_file = []
        # if file_name is None:
        #     file_i = f"{i:04d}"
        # else:
        #     file_i = file_name[i]

        if begin_time is None:
            begin_i = "1970-01-01T00:00:00.000"
        else:
            begin_i = begin_time[i]
            if len(begin_i) == 0:
                begin_i = "1970-01-01T00:00:00.000"
        begin_i = datetime.fromisoformat(begin_i.rstrip("Z"))

        p_window = int(window_amp[0] / dt[i])  ## index of window
        s_window = int(window_amp[1] / dt[i])  ## index of window

        for j in range(nch):
            for k in range(nst):
                if station_id is None:
                    station_i = f"{k + begin_channel_index[i]:04d}"
                else:
                    station_i = station_id[k][i]

                topk_index_ijk, ii = torch.sort(topk_index[i, j, k])
                topk_score_ijk = topk_score[i, j, k][ii]

                # for ii, (index, score) in enumerate(zip(topk_index[i, j, k], topk_score[i, j, k])):
                for ii, (index, score) in enumerate(zip(topk_index_ijk, topk_score_ijk)):
                    if score > vmin:
                        center_index = index.item() + begin_time_index[i]
                        center_time = begin_i + timedelta(seconds=index.item() * dt[i] * event_scale)
                        event_dict = {
                            # "file_name": file_i,
                            "station_id": station_i,
                            "center_index": center_index,
                            "center_time": center_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                            "event_score": f"{score.item():.3f}",
                            "dt_s": dt[i] * event_scale,
                        }

                        if event_time is not None:
                            t0 = center_time - timedelta(seconds=event_time[i, 0, index.item(), k].item() * dt[i])
                            event_dict["event_time"] = t0.strftime("%Y-%m-%dT%H:%M:%S.%f")
                            event_dict["travel_time"] = event_time[i, 0, index.item(), k].item() * dt[i]
                            if event_dict["travel_time"] <= 0:
                                continue

                        if waveform is not None:
                            ## calculate PS ratio
                            ps_delta = max(
                                2,
                                event_time[i, 0, index.item(), k].item() * 2.0 * (VPVS_RATIO - 1) / (VPVS_RATIO + 1),
                            )  # 2 is to prevent error of torch.max()
                            itp = max(0, index.item() - int(ps_delta * 0.5))
                            its = max(0, index.item() + int(ps_delta * 0.5))

                            # p_amp = torch.max(torch.abs(waveform[i, :, itp : min(itp + p_window, its), k]))
                            # s_amp = torch.max(torch.abs(waveform[i, :, its : its + s_window, k]))
                            waveform_cut = waveform[i, :, itp : min(itp + p_window, its), k]
                            p_amp = torch.max(waveform_cut) - torch.min(waveform_cut)
                            waveform_cut = waveform[i, :, its : its + s_window, k]
                            s_amp = torch.max(waveform_cut) - torch.min(waveform_cut)
                            event_dict["sp_ratio"] = s_amp.item() / p_amp.item()

                            ## calculate event amplitude
                            # event_amp = torch.max(torch.abs(waveform[i, :, itp : its + (its - itp), k]))
                            waveform_cut = waveform[i, :, itp : its + (its - itp), k]
                            event_amp = (torch.max(torch.abs(waveform_cut)) - torch.min(torch.abs(waveform_cut))) / 2.0
                            event_dict["event_amplitude"] = event_amp.item()

                        events_per_file.append(event_dict)

        events.append(events_per_file)
    return events


def merge_picks(pick_dir):
    pick_dir = Path(pick_dir)

    num_p = 0
    num_s = 0
    picks = []
    for file in tqdm(
        pick_dir.rglob(
            "*.csv",
        ),
        desc=f"Merging {pick_dir.name}",
    ):
        if os.stat(file).st_size == 0:
            continue
        picks_ = pd.read_csv(file)
        num_p += len(picks_[picks_["phase_type"] == "P"])
        num_s += len(picks_[picks_["phase_type"] == "S"])
        picks.append(picks_)
    if len(picks) == 0:
        with open(str(pick_dir) + ".csv", "w") as fp:
            fp.write("")
    else:
        picks = pd.concat(picks)
        picks = picks.sort_values(["phase_time", "station_id", "phase_type"])
        picks = picks.reset_index(drop=True)
        picks.to_csv(str(pick_dir) + ".csv", index=False)

    print(f"Number of picks: {len(picks)}")
    if (num_p > 0) and (num_s > 0):
        print(f"Number of P picks: {num_p}")
        print(f"Number of S picks: {num_s}")
    return 0


def merge_events(event_dir):
    event_dir = Path(event_dir)

    events = []
    for file in tqdm(
        event_dir.rglob(
            "*.csv",
        ),
        desc=f"Merging {event_dir.name}",
    ):
        if os.stat(file).st_size == 0:
            continue
        events_ = pd.read_csv(file)
        events.append(events_)
    if len(events) == 0:
        with open(str(event_dir) + ".csv", "w") as fp:
            fp.write("")
    else:
        events = pd.concat(events)
        events = events.sort_values(["event_time", "station_id"])
        events = events.reset_index(drop=True)
        events.to_csv(str(event_dir) + ".csv", index=False)

    print(f"Number of event x station: {len(events)}")
    return 0


def merge_patch(patch_dir, merged_dir, return_single_file=False):
    patch_dir = Path(patch_dir)
    merged_dir = Path(merged_dir)
    if not merged_dir.exists():
        merged_dir.mkdir()

    files = list(patch_dir.rglob("*_*_*.csv"))
    group = defaultdict(list)
    for file in files:
        key = "_".join((str(file).replace(f"{patch_dir}/", "")).split("_")[:-2])
        group[key].append(file)  ## event_id

    if return_single_file:
        fp_total = open(str(merged_dir).rstrip("/") + ".csv", "w")

    num_picks = 0
    header0 = None
    for k in tqdm(group, desc=f"Merging {merged_dir}"):
        header = None
        if not (merged_dir / f"{k}.csv").parent.exists():
            (merged_dir / f"{k}.csv").parent.mkdir()
        with open(merged_dir / f"{k}.csv", "w") as fp_file:
            for file in sorted(group[k]):
                with open(file, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        if header is None:
                            header = lines[0]
                            fp_file.writelines(header)
                        if return_single_file and (header0 is None):
                            header = lines[0]
                            fp_total.writelines(header)
                        fp_file.writelines(lines[1:])
                        if return_single_file:
                            fp_total.writelines(lines[1:])
                        num_picks += len(lines[1:])

    print(f"Number of detections: {num_picks}")
    return 0
