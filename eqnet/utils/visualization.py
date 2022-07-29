import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def visualize_das_train(meta, preds, epoch, figure_dir="figures"):

    meta_data = meta["data"].cpu()
    raw_data = meta_data.clone().permute(0, 2, 3, 1).numpy()
    # data = normalize_local(meta_data.clone()).permute(0, 2, 3, 1).numpy()
    targets = meta["targets"].permute(0, 2, 3, 1).numpy()

    y = preds.permute(0, 2, 3, 1).numpy()

    for i in range(len(raw_data)):

        raw_vmax = np.std(raw_data[i]) * 2
        raw_vmin = -raw_vmax

        vmax = np.std(raw_data[i]) * 2
        vmin = -vmax

        fig, ax = plt.subplots(2,2, figsize=(12, 12), sharex=False, sharey=False)
        ax[0, 0].imshow((raw_data[i]-np.mean(raw_data[i])), vmin=raw_vmin, vmax=raw_vmax, interpolation='none', cmap="seismic", aspect='auto')
        # ax[1, 0].imshow((data[i]-np.mean(data[i])), vmin=vmin, vmax=vmax, interpolation='none', cmap="seismic", aspect='auto')
        ax[0, 1].imshow(y[i], vmin=0, vmax=1, interpolation='none', aspect='auto')
        ax[1, 1].imshow(targets[i],  vmin=0, vmax=1, interpolation='none', aspect='auto')
        # ax[0, 1].imshow(y[i], interpolation='none', aspect='auto')
        # ax[1, 1].imshow(targets[i], interpolation='none', aspect='auto')
        
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{local_rank}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}.png", dpi=300)

        plt.close(fig)

def visualize_eqnet_train(meta, phase, event, epoch, figure_dir="figures"):

    for i in range(meta["waveform"].shape[0]):
        plt.close("all")
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        for j in range(phase.shape[-1]):
            axes[0].plot((meta["waveform"][i, -1, :, j])/torch.std(meta["waveform"][i, -1, :, j])/8 + j)
            
            axes[1].plot(phase[i, 1, :, j] + j, "r")
            axes[1].plot(phase[i, 2, :, j] + j, "b")
            axes[1].plot(meta["phase_pick"][i, 1, :, j] + j, "--C3")
            axes[1].plot(meta["phase_pick"][i, 2, :, j] + j, "--C0")

            axes[2].plot(event[i, :, j] + j, "b")
            axes[2].plot(meta["center_heatmap"][i, :, j] + j, "--C0")

        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{local_rank}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}.png", dpi=300)


def plot_das(data, pred, picks=None, file_name=None, figure_dir="./figures", epoch=0, **kwargs):

    ## pytorch BCHW => BHWC
    data = np.transpose(data, [0, 2, 3, 1])
    pred = np.transpose(pred, [0, 2, 3, 1])

    if file_name is None:
        file_name = [f"{epoch:03d}_{i:03d}" for i in range(len(data))]
    file_name = [x if isinstance(x, str) else x.decode() for x in file_name]

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

    if ("begin_channel_index" in kwargs) and (kwargs["begin_channel_index"] is not None):
        begin_channel_index = [x.item() for x in kwargs["begin_channel_index"]]
    else:
        begin_channel_index = [0 for i in range(len(data))]
    if ("begin_time_index" in kwargs) and (kwargs["begin_time_index"] is not None):
        begin_time_index = [x.item() for x in kwargs["begin_time_index"]]
    else:
        begin_time_index = [0 for i in range(len(data))]

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
            (np.arange(nx) + begin_channel_index[i]) * dx[i] / 1e3, #km
            (np.arange(nt) + begin_time_index[i]) * dt[i],
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
                os.path.join(figure_dir, file_name[i] + ".png"),
                bbox_inches="tight",
                dpi=300,
            )
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(figure_dir, file_name[i])), exist_ok=True)
            fig.savefig(
                os.path.join(figure_dir, file_name[i] + ".png"),
                bbox_inches="tight",
                dpi=300,
            )

        plt.close(fig)

    return 0