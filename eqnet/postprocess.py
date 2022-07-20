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


def plot_das(data, pred, picks=None, file_names=None, figure_dir="./figures", epoch=0, **kwargs):

    ## pytorch BCHW => BHWC
    data = np.transpose(data, [0, 2, 3, 1])
    pred = np.transpose(pred, [0, 2, 3, 1])

    if file_names is None:
        file_names = [f"{epoch:03d}_{i:03d}" for i in range(len(data))]
    file_names = [x if isinstance(x, str) else x.decode() for x in file_names]

    if "dx" in kwargs:
        if type(kwargs["dx"]) is list:
            dx = [kwargs["dx"][i].item() for i in range(len(data))]
        else:
            dx = [kwargs["dx"].item() for i in range(len(data))]
    else:
        dx = [10.0 for i in range(len(data))]
    if "dt" in kwargs:
        if type(kwargs["dt"]) is list:
            dt = [kwargs["dt"][i].item() for i in range(len(data))]
        else:
            dt = [kwargs["dt"].item() for i in range(len(data))]
    else:
        dt = [0.01 for i in range(len(data))]

    if ("begin_channel_indexs" in kwargs) and (kwargs["begin_channel_indexs"] is not None):
        begin_channel_indexs = [x.item() for x in kwargs["begin_channel_indexs"]]
    else:
        begin_channel_indexs = [0 for i in range(len(data))]
    if ("begin_time_indexs" in kwargs) and (kwargs["begin_time_indexs"] is not None):
        begin_time_indexs = [x.item() for x in kwargs["begin_time_indexs"]]
    else:
        begin_time_indexs = [0 for i in range(len(data))]

    nt, nx = data.shape[1], data.shape[2]
    # x = np.arange(nx) * dx
    # t = np.arange(nt) * dt

    for i in range(len(data)):

        if (picks is not None) and (len(picks[i]) > 0):
            picks_ = pd.DataFrame(picks[i])  # picks per file

        std = np.std(data[i, :, :, 0]) * 2

        # fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
        fig, axs = plt.subplots(1, 1)
        im = axs.pcolormesh(
            (np.arange(nx) + begin_channel_indexs[i]) * dx[i] / 1e3, #km
            (np.arange(nt) + begin_time_indexs[i]) * dt[i],
            data[i, :, :, 0],
            vmin=-std,
            vmax=std,
            cmap="seismic",
            shading="auto",
            rasterized=True,
        )
        axs.set_xlabel("Distance (km)")
        axs.set_ylabel("Time (s)")
        axs.invert_yaxis()
        axs.xaxis.tick_top()
        axs.xaxis.set_label_position("top")
        # im = axs[0, 0].imshow(
        #     data[i, :, :, 0],
        #     vmin=-std,
        #     vmax=std,
        #     cmap="seismic",
        #     interpolation='none',
        #     aspect="auto",
        # )
        # plt.colorbar(im0, ax=axs[0])
        # axs[0].set_title("DAS data")

        # im2 = axs[1].pcolormesh(x, t, pred[i, :, :, 2],  cmap="seismic", vmin=-1, vmax=1, alpha=0.5, shading='auto', rasterized=True)
        # im1 = axs[1].pcolormesh(x, t, -pred[i, :, :, 1],  cmap="seismic", vmin=-1, vmax=1, alpha=0.5, shading='auto', rasterized=True)

        # axs[1].invert_yaxis()
        # # plt.colorbar(im1, ax=axs[1])
        # axs[1].set_title("Prediction")

        # im = axs[0, 1].imshow(
        #     pred[i, :, :, 1],
        #     vmin=0,
        #     # vmax=0.5,
        #     cmap="hot",
        #     interpolation='none',
        #     aspect="auto",
        # )
        # plt.colorbar(im, ax=axs[0, 1])
        # axs[0, 1].set_title("P-phase")

        # im = axs[1, 0].imshow(
        #     pred[i, :, :, 2],
        #     vmin=0,
        #     # vmax=0.5,
        #     cmap="hot",
        #     interpolation='none',
        #     aspect="auto",
        # )
        # plt.colorbar(im, ax=axs[1, 0])
        # axs[1, 0].set_title("S-phase")

        # # axs[1].pcolormesh(1-pred[i, :, :, 0], vmin=0, vmax=1, cmap="hot", rasterized=True)
        # # axs[1].invert_yaxis()
        # im = axs[1, 1].imshow(
        #     1 - pred[i, :, :, 0],
        #     vmin=0,
        #     # vmax=0.5,
        #     cmap="hot",
        #     interpolation='none',
        #     aspect="auto",
        # )
        # plt.colorbar(im, ax=axs[1, 1])
        # axs[1, 1].set_title("(P+S)")

        if (picks is not None) and (len(picks[i]) > 0):
            p_picks = picks_[picks_["phase_type"] == "P"]
            s_picks = picks_[picks_["phase_type"] == "S"]
            axs.plot(
                p_picks["station_name"].astype("int") * dx[i] / 1e3, #km
                p_picks["phase_index"] * dt[i],
                ".C0",
                # linewidth=5,
                linewidth=0.0,
                markersize=0.5,
                alpha=1.0,
                label="P-phase",
            )
            axs.plot(
                s_picks["station_name"].astype("int") * dx[i] / 1e3, #km
                s_picks["phase_index"] * dt[i],
                # "-C3",
                ".C2",
                # linewidth=5,
                linewidth=0.0,
                markersize=0.5,
                alpha=1.0,
                label="S-phase",
            )
            axs.legend(markerscale=10.0)
            # axs[1].plot(p_picks["station_name"], p_picks["phase_index"], "r,", linewidth=0)
            # axs[1].plot(s_picks["station_name"], s_picks["phase_index"], "b,", linewidth=0)

        try:
            fig.savefig(
                os.path.join(figure_dir, file_names[i] + ".png"),
                bbox_inches="tight",
                dpi=300,
            )
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(figure_dir, file_names[i])), exist_ok=True)
            fig.savefig(
                os.path.join(figure_dir, file_names[i] + ".png"),
                bbox_inches="tight",
                dpi=300,
            )

        plt.close(fig)

    return 0