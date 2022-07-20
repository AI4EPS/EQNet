import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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