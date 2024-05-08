import os
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def normalize(x):
    """x: [batch, channel, time, station]"""
    x = x - torch.mean(x, dim=2, keepdim=True)
    std = torch.std(x, dim=2, keepdim=True)
    std[std == 0] == 1
    x = x / std / 6
    return x


def plot_autoencoder_das_train(meta, preds, epoch, figure_dir="figures"):
    meta_data = meta["data"]
    raw_data = meta_data.clone().permute(0, 2, 3, 1).numpy()
    # data = normalize_local(meta_data.clone()).permute(0, 2, 3, 1).numpy()
    targets = meta["targets"].permute(0, 2, 3, 1).numpy()

    y = preds.permute(0, 2, 3, 1).numpy()

    for i in range(len(raw_data)):
        raw_vmax = np.std(raw_data[i]) * 2
        raw_vmin = -raw_vmax

        vmax = np.std(raw_data[i]) * 2
        vmin = -vmax

        fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False)
        im = ax[0, 0].imshow(
            raw_data[i], vmin=raw_vmin, vmax=raw_vmax, interpolation="none", cmap="seismic", aspect="auto"
        )
        fig.colorbar(im, ax=ax[0, 0])
        im = ax[0, 1].imshow(
            targets[i], vmin=raw_vmin, vmax=raw_vmax, interpolation="none", cmap="seismic", aspect="auto"
        )
        fig.colorbar(im, ax=ax[0, 1])
        im = ax[1, 0].imshow(y[i], vmin=raw_vmin, vmax=raw_vmax, interpolation="none", cmap="seismic", aspect="auto")
        fig.colorbar(im, ax=ax[1, 0])
        im = ax[1, 1].imshow(y[i], interpolation="none", cmap="seismic", aspect="auto")
        fig.colorbar(im, ax=ax[1, 1])

        # ax[0, 1].imshow(y[i], vmin=0, vmax=1, interpolation='none', aspect='auto')
        # ax[1, 1].imshow(targets[i],  vmin=0, vmax=1, interpolation='none', aspect='auto')
        # ax[0, 1].imshow(y[i], interpolation='none', aspect='auto')
        # ax[1, 1].imshow(targets[i], interpolation='none', aspect='auto')

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}.png", dpi=300)

        plt.close(fig)


def plot_das_train(meta, preds, epoch, figure_dir="figures", dt=0.01, dx=10, prefix=""):
    meta_data = meta["data"].cpu()
    raw_data = meta_data.clone().permute(0, 2, 3, 1).numpy()
    # data = normalize_local(meta_data.clone()).permute(0, 2, 3, 1).numpy()
    targets = meta["phase_pick"].permute(0, 2, 3, 1).numpy()
    y = preds.permute(0, 2, 3, 1).numpy()

    if targets.shape[-1] < 3:
        targets_ = np.zeros((targets.shape[0], targets.shape[1], targets.shape[2], 3))
        targets_[:, :, :, : targets.shape[-1]] = targets
        targets = targets_
    if y.shape[-1] < 3:
        y_ = np.zeros((y.shape[0], y.shape[1], y.shape[2], 3))
        y_[:, :, :, : y.shape[-1]] = y
        y = y_
    if targets.shape[-1] == 4:
        targets = targets[:, :, :, 1:]
    if y.shape[-1] == 4:
        y = y[:, :, :, 1:]

    for i in range(len(raw_data)):
        raw_vmax = np.std(raw_data[i]) * 2
        raw_vmin = -raw_vmax

        vmax = np.std(raw_data[i]) * 2
        vmin = -vmax

        fig, ax = plt.subplots(1, 3, figsize=(3 * 3, 3), sharex=False, sharey=False)
        ax[0].imshow(
            (raw_data[i] - np.mean(raw_data[i])),
            vmin=raw_vmin,
            vmax=raw_vmax,
            extent=(0, raw_data[i].shape[1] * dx / 1e3, raw_data[i].shape[0] * dt, 0),
            interpolation="none",
            cmap="seismic",
            aspect="auto",
        )
        ax[0].set_xlabel("Distance (km)")
        ax[0].set_ylabel("Time (s)")
        ax[0].set_title("DAS Data")
        # ax[1, 0].imshow((data[i]-np.mean(data[i])), vmin=vmin, vmax=vmax, interpolation='none', cmap="seismic", aspect='auto')

        # targets[i][:, :, 1] = 1 - targets[i][:, :, 1]
        # targets[i][:, :, 2] = 1 - targets[i][:, :, 2]
        # targets[i][:, (targets[i][:, :, 0] == 0).all(axis=0), :] = 0
        ax[1].imshow(
            # targets[i][:, :, [1, 0, 2]],
            targets[i],
            vmin=0,
            vmax=1,
            extent=(0, targets[i].shape[1] * dx / 1e3, targets[i].shape[0] * dt, 0),
            interpolation="none",
            aspect="auto",
        )
        ax[1].set_xlabel("Distance (km)")
        ax[1].set_title("Noisy Label")
        ax[1].set_ylabel("Time (s)")
        # ax[0, 1].imshow(y[i], interpolation='none', aspect='auto')

        # y[i][:, :, 0] = 0
        # y[i][:, :, 1] = 1 - y[i][:, :, 1]
        # y[i][:, :, 2] = 1 - y[i][:, :, 2]
        # y[i][:, (y[i][:, :, 0] == 0).all(axis=0), :] = 0
        ax[2].imshow(
            # y[i][:, :, [1, 0, 2]],
            y[i],
            vmin=0,
            vmax=1,
            extent=(0, y[i].shape[1] * dx / 1e3, y[i].shape[0] * dt, 0),
            interpolation="none",
            aspect="auto",
        )
        ax[2].set_xlabel("Distance (km)")
        ax[2].set_ylabel("Time (s)")
        ax[2].set_title("Prediction")

        fig.tight_layout()

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}_{prefix}.png", dpi=300, bbox_inches="tight")
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{prefix}.png", dpi=300, bbox_inches="tight")

        plt.close(fig)


def plot_phasenet_train(meta, phase, epoch=0, figure_dir="figures", prefix=""):
    for i in range(meta["data"].shape[0]):
        plt.close("all")

        chn_name = ["E", "N", "Z"]

        if "raw_data" in meta:
            shift = 3
            fig, axes = plt.subplots(7, 1, figsize=(10, 10))
            for j in range(3):
                axes[j].plot(meta["raw_data"][i, j, :, 0], linewidth=0.5, color="k", label=f"{chn_name[j]}")
                axes[j].set_xticklabels([])
                axes[j].grid("on")
        else:
            fig, axes = plt.subplots(4, 1, figsize=(10, 10))
            shift = 0

        for j in range(3):
            axes[j + shift].plot(meta["data"][i, j, :, 0], linewidth=0.5, color="k", label=f"{chn_name[j]}")
            axes[j + shift].set_xticklabels([])
            axes[j + shift].grid("on")

        k = 3 + shift
        axes[k].plot(phase[i, 1, :, 0], "b")
        axes[k].plot(phase[i, 2, :, 0], "r")
        axes[k].plot(meta["phase_pick"][i, 1, :, 0], "--C0")
        axes[k].plot(meta["phase_pick"][i, 2, :, 0], "--C3")
        axes[k].plot(meta["phase_mask"][i, 0, :, 0], ":", color="gray")
        axes[k].set_ylim(-0.05, 1.05)
        axes[k].set_xticklabels([])
        axes[k].grid("on")

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}_{prefix}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{prefix}.png", dpi=300)

        if i >= 20:
            break


def plot_phasenet_plus_train(
    meta, phase, polarity=None, event_center=None, event_time=None, epoch=0, figure_dir="figures", prefix=""
):
    nb, nc, nt, ns = meta["data"].shape
    dt = 0.01
    nt_event = event_center.shape[2]
    dt_event = dt * nt / nt_event
    nt_polarity = polarity.shape[2]
    dt_polarity = dt * nt / nt_polarity

    for i in range(meta["data"].shape[0]):
        plt.close("all")

        chn_name = ["E", "N", "Z"]

        if "raw_data" in meta:
            shift = 3
            fig, axes = plt.subplots(9, 1, figsize=(10, 10))
            t = torch.arange(nt) * dt
            for j in range(3):
                axes[j].plot(t, meta["raw_data"][i, j, :, 0], linewidth=0.5, color="k", label=f"{chn_name[j]}")
                axes[j].set_xlim(t[0], t[-1])
                axes[j].set_xticklabels([])
                axes[j].grid("on")
        else:
            fig, axes = plt.subplots(6, 1, figsize=(10, 10))
            shift = 0

        for j in range(3):
            t = torch.arange(nt) * dt
            axes[j + shift].plot(t, meta["data"][i, j, :, 0], linewidth=0.5, color="k", label=f"{chn_name[j]}")
            axes[j + shift].set_xlim(t[0], t[-1])
            axes[j + shift].set_xticklabels([])
            axes[j + shift].grid("on")

        k = 3 + shift
        t = torch.arange(nt) * dt
        axes[k].plot(t, phase[i, 1, :, 0], "b")
        axes[k].plot(t, phase[i, 2, :, 0], "r")
        axes[k].plot(t, meta["phase_pick"][i, 1, :, 0], "--C0")
        axes[k].plot(t, meta["phase_pick"][i, 2, :, 0], "--C3")
        axes[k].plot(t, meta["phase_mask"][i, 0, :, 0], ":", color="gray")
        axes[k].set_xlim(t[0], t[-1])
        axes[k].set_ylim(-0.05, 1.05)
        axes[k].set_xticklabels([])
        axes[k].grid("on")

        t = torch.arange(nt_polarity) * dt_polarity

        # 1 channel for polarity
        # axes[k + 1].plot(t, polarity[i, 0, :, 0], "b")
        # axes[k + 1].plot(t, meta["polarity"][i, 0, :, 0], "--C0")
        # axes[k + 1].plot(t, meta["polarity_mask"][i, 0, :, 0], ":", color="gray")

        # 3 channels for polarity
        axes[k + 1].plot(t, (1.0 + polarity[i, 1, :, 0]) / 2.0, "g", alpha=0.3, linewidth=0.5)
        axes[k + 1].plot(t, (1.0 - polarity[i, 2, :, 0]) / 2.0, "g", alpha=0.3, linewidth=0.5)
        axes[k + 1].plot(t, ((polarity[i, 1, :, 0] - polarity[i, 2, :, 0]) + 1.0) / 2.0, "b", alpha=1.0, linewidth=1.0)
        axes[k + 1].plot(t, (1.0 + meta["polarity"][i, 1, :, 0]) / 2.0, "--C0", alpha=0.5, linewidth=1.0)
        axes[k + 1].plot(t, (1.0 - meta["polarity"][i, 2, :, 0]) / 2.0, "--C0", alpha=0.5, linewidth=1.0)
        axes[k + 1].plot(t, meta["polarity_mask"][i, 0, :, 0], ":", color="gray")

        axes[k + 1].set_xlim(t[0], t[-1])
        axes[k + 1].set_ylim(-0.05, 1.05)
        axes[k + 1].set_xticklabels([])
        axes[k + 1].grid("on")

        t = torch.arange(nt_event) * dt_event
        axes[k + 2].plot(t, event_center[i, 0, :, 0], "b")
        axes[k + 2].plot(t, meta["event_center"][i, 0, :, 0], "--C0")
        axes[k + 2].plot(t, meta["event_mask"][i, 0, :, 0], ":", color="gray")
        axes[k + 2].set_xlim(t[0], t[-1])
        axes[k + 2].set_ylim(-0.05, 1.05)
        axes[k + 2].grid("on")

        axes2 = axes[k + 2].twinx()
        axes2.plot(t, event_time[i, 0, :, 0], "--C1")
        axes2.plot(t, meta["event_time"][i, 0, :, 0], ":C3")

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}_{prefix}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{prefix}.png", dpi=300)

        if i >= 20:
            break


def plot_phasenet(
    meta,
    phase,
    picks=None,
    phases=None,
    dt=0.01,
    nt=6000 * 10,
    epoch=0,
    file_name=None,
    figure_dir="figures",
    **kwargs,
):
    nb0, nc0, nt0, ns0 = phase.shape
    chn_name = ["E", "N", "Z"]
    # normalize = lambda x: (x - torch.mean(x, dim=2, keepdim=True)) / torch.std(x, dim=2, keepdim=True) / 10
    # waveform_raw = normalize(meta["waveform_raw"])
    # waveform = normalize(meta["waveform"])
    # waveform = meta["waveform"] / 3.0

    waveform = meta["data"]
    # waveform = normalize(waveform)
    # vmax = torch.std(waveform) * 3
    # vmin = -vmax
    if "begin_time" in meta:
        begin_time = meta["begin_time"]
    else:
        begin_time = [0] * nb0

    for i in range(nb0):
        dt = dt[i].item()

        for ii in range(0, nt0, nt):
            plt.close("all")
            fig, axes = plt.subplots(2, 1, figsize=(20, 10))  # , gridspec_kw={"height_ratios": [5, 5, 1, 1, 1]})

            # j = 2
            # # for j in range(3):
            # for k in range(ns0):
            #     axes[0].plot(t, phase[i, 1, ii:ii+nt, k] + k, "-C0")
            #     axes[0].plot(t, phase[i, 2, ii:ii+nt, k] + k, "-C1")
            #     mask = ((phase[i, 1, ii:ii+nt, k] > 0.1) | (phase[i, 2, ii:ii+nt, k] > 0.1))
            #     axes[0].plot(t[mask], polarity[i, 0, ii:ii+nt, k][mask] + k, "-C2")
            #     axes[0].plot(t_event, event[i, 0, ii//event_dt_ratio:(ii+nt)//event_dt_ratio, k] + k, "-C3")
            #     axes[0].plot(t, waveform_raw[i, j, ii:ii+nt, k] + k, linewidth=0.5, color="k", label=f"{chn_name[j]}")
            # # axes[0].set_xticklabels([])
            # axes[0].grid("on")

            # for j in range(3):
            for k in range(ns0):
                begin_time_i = datetime.fromisoformat(begin_time[i])
                t = [
                    begin_time_i + timedelta(seconds=(ii + it) * dt) for it in range(len(phase[i, 1, ii : ii + nt, k]))
                ]

                if ns0 == 1:
                    for j in range(3):
                        waveform_ijk = waveform[i, j, ii : ii + nt, k]
                        waveform_ijk -= torch.mean(waveform_ijk)
                        waveform_ijk /= torch.std(waveform_ijk) * 6
                        axes[0].plot(t, waveform_ijk + j, linewidth=0.2, color="k", label=f"{chn_name[j]}")
                else:
                    waveform_ijk = waveform[i, 2, ii : ii + nt, k]
                    waveform_ijk -= torch.mean(waveform_ijk)
                    waveform_ijk /= torch.std(waveform_ijk) * 6
                axes[0].plot(t, waveform_ijk + k, linewidth=0.2, color="k", label=f"{chn_name[2]}")

                axes[1].plot(t, phase[i, 1, ii : ii + nt, k] + k, "-C0", linewidth=1.0)
                axes[1].plot(t, phase[i, 2, ii : ii + nt, k] + k, "-C1", linewidth=1.0)

            axes[0].grid("on")
            axes[1].grid("on")

            fig.tight_layout()

            fig.savefig(
                os.path.join(figure_dir, file_name[i].replace("/", "_") + f"_{ii:06d}.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)


def plot_phasenet_plus(
    meta,
    phase,
    polarity=None,
    event_center=None,
    event_time=None,
    phase_picks=None,
    event_detects=None,
    dt=0.01,
    nt=6000 * 10,
    file_name=None,
    figure_dir="figures",
    **kwargs,
):
    nb, nc, nt, ns = meta["data"].shape
    if isinstance(dt, torch.Tensor):
        dt = dt.item()
    nt_event = event_center.shape[2]
    dt_event = dt * nt / nt_event
    nt_polarity = polarity.shape[2]
    dt_polarity = dt * nt / nt_polarity

    if "begin_time" in meta:
        begin_time = meta["begin_time"]
    else:
        begin_time = [0] * nb

    for i in range(meta["data"].shape[0]):
        plt.close("all")

        chn_name = ["E", "N", "Z"]

        if "raw_data" in meta:
            shift = 3
            fig, axes = plt.subplots(9, 1, figsize=(10, 10))
            t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
            for j in range(3):
                axes[j].plot(t, meta["raw_data"][i, j, :, 0], lw=0.5, color="k", label=f"{chn_name[j]}")
                axes[j].set_xlim(t[0], t[-1])
                axes[j].set_xticklabels([])
                axes[j].grid("on")
                axes[j + shift].legend(loc="upper right")
        else:
            fig, axes = plt.subplots(6, 1, figsize=(10, 10))
            shift = 0

        for j in range(3):
            t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
            axes[j + shift].plot(t, meta["data"][i, j, :, 0], lw=0.5, color="k", label=f"{chn_name[j]}")
            axes[j + shift].set_xlim(t[0], t[-1])
            axes[j + shift].set_xticklabels([])
            axes[j + shift].grid("on")
            axes[j + shift].legend(loc="upper right")

        k = 3 + shift
        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
        axes[k].plot(t, phase[i, 2, :, 0], "r", lw=1.0)
        axes[k].plot(t, phase[i, 1, :, 0], "b", lw=1.0)
        color = {"P": "b", "S": "r"}
        for ii, pick in enumerate(phase_picks[i]):
            tt = pd.to_datetime(pick["phase_time"])
            axes[k].plot([tt, tt], [-0.05, 1.05], f"--{color[pick['phase_type']]}", linewidth=0.8)

        axes[k].plot([], [], "-b", label="P-phase")
        axes[k].plot([], [], "-r", label="S-phase")
        axes[k].set_xlim(t[0], t[-1])
        axes[k].set_ylim(-0.05, 1.05)
        axes[k].set_xticklabels([])
        axes[k].grid("on")
        axes[k].legend(loc="upper right")

        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt_polarity, freq=pd.Timedelta(seconds=dt_polarity))
        # axes[k + 1].plot(t, (polarity[i, 0, :, 0] - 0.5) * 2.0, "b", label="polarity")
        # axes[k + 1].plot(t, polarity[i, 1, :, 0] - polarity[i, 2, :, 0], "b", label="Polarity")
        for ii, pick in enumerate(phase_picks[i]):
            tt = pd.to_datetime(pick["phase_time"])
            amp = pick["phase_polarity"]
            if abs(amp) > 0.15:
                axes[k + 1].annotate(
                    "",
                    xy=(tt, -0.03 * np.sign(amp)),
                    xytext=(tt, amp),
                    arrowprops=dict(arrowstyle="<-", color=f"{color[pick['phase_type']]}", lw=1.5),
                )
        axes[k + 1].plot([], [], "-b", label="P-polarity")
        axes[k + 1].plot([], [], "-r", label="S-polarity")
        axes[k + 1].plot([t[0], t[-1]], [0.0, 0.0], "-", color="blue", lw=1.0)
        axes[k + 1].set_xlim(t[0], t[-1])
        axes[k + 1].set_ylim(-1.05, 1.05)
        axes[k + 1].set_xticklabels([])
        axes[k + 1].grid("on")
        axes[k + 1].legend(loc="upper right")

        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt_event, freq=pd.Timedelta(seconds=dt_event))
        axes[k + 2].plot(t, event_center[i, 0, :, 0], "b", label="Event")
        axes[k + 2].set_xlim(t[0], t[-1])
        axes[k + 2].set_ylim(-0.05, 1.05)
        axes[k + 2].grid("on")
        axes[k + 2].legend(loc="upper right")
        axes[k + 2].set_xlabel("Time (s)")

        # axes2 = axes[k + 2].twinx()
        # axes2.plot(t, event_time[i, 0, :, 0], "--C1")
        # axes2.set_ylabel("Time (s)")

        for ii, event in enumerate(event_detects[i]):
            ot = pd.to_datetime(event["event_time"])
            axes[k + 2].plot([ot, ot], [-0.05, 1.05], "--r", linewidth=2.0)
            at = pd.to_datetime(event["center_time"])
            axes[k + 2].plot([at, at], [-0.05, 1.05], "--b", linewidth=1.0)
            axes[k + 2].annotate(
                "",
                xy=(max(t[0], at), 0.3),
                xytext=(max(t[0], ot), 0.3),
                arrowprops=dict(arrowstyle="<-", color="C1", lw=2),
            )
        axes[k + 2].plot([], [], "--C3", label="Origin time")

        axes[k + 2].set_xlim(t[0], t[-1])
        axes[k + 2].set_ylim(-0.05, 1.05)
        axes[k + 2].grid("on")
        axes[k + 2].legend(loc="upper right")
        axes[k + 2].set_xlabel("Time (s)")

        fig.tight_layout()

        fig.savefig(
            os.path.join(figure_dir, file_name[i].replace("/", "_") + ".png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def plot_eqnet_train(meta, phase, event, epoch, figure_dir="figures"):
    for i in range(meta["data"].shape[0]):
        plt.close("all")
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        for j in range(phase.shape[-1]):
            axes[0].plot((meta["data"][i, -1, :, j]) / torch.std(meta["data"][i, -1, :, j]) / 8 + j)

            axes[1].plot(phase[i, 1, :, j] + j, "r")
            axes[1].plot(phase[i, 2, :, j] + j, "b")
            axes[1].plot(meta["phase_pick"][i, 1, :, j] + j, "--C3")
            axes[1].plot(meta["phase_pick"][i, 2, :, j] + j, "--C0")

            axes[2].plot(event[i, :, j] + j, "b")
            axes[2].plot(meta["event_center"][i, :, j] + j, "--C0")

        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{local_rank}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}.png", dpi=300)


def plot_das(data, pred, picks=None, phases=["P", "S"], file_name=None, figure_dir="./figures", epoch=0, **kwargs):
    ## pytorch BCHW => BHWC
    data = normalize(data)
    data = np.transpose(data, [0, 2, 3, 1])
    pred = np.transpose(pred, [0, 2, 3, 1])
    if pred.shape[-1] < 3:
        pred_ = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 3))
        pred_[:, :, :, : pred.shape[-1]] = pred
        pred = pred_
    if pred.shape[-1] == 4:
        pred = pred[:, :, :, 1:]

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

        # std = np.std(data[i, :, :, 0])
        std = torch.std(data[i, :, :, 0]).item()

        # fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
        # fig, axs = plt.subplots(1, 1)
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        # im = axs[0].pcolormesh(
        #     (np.arange(nx) + begin_channel_index[i]) * dx[i] / 1e3,  # km
        #     (np.arange(nt) + begin_time_index[i]) * dt[i],
        #     data[i, :, :, 0],
        #     vmin=-std,
        #     vmax=std,
        #     cmap="seismic",
        #     shading="auto",
        #     rasterized=True,
        # )
        im = axs[0].imshow(
            data[i, :, :, 0],
            extent=[
                begin_channel_index[i] * dx[i] / 1e3,
                (begin_channel_index[i] + nx) * dx[i] / 1e3,
                (begin_time_index[i] + nt) * dt[i],
                begin_time_index[i] * dt[i],
            ],
            vmin=-std,
            vmax=std,
            cmap="seismic",
            aspect="auto",
            interpolation="none",
            origin="upper",
        )
        axs[0].set_xlabel("Distance (km)")
        axs[0].set_ylabel("Time (s)")
        # axs[0].invert_yaxis()
        axs[0].xaxis.tick_top()
        axs[0].xaxis.set_label_position("top")

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
            for phase in phases:
                tmp_picks = picks_[picks_["phase_type"] == phase]
                axs[0].plot(
                    tmp_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
                    tmp_picks["phase_index"] * dt[i],
                    # ".C0",
                    # ".C2",
                    # linewidth=5,
                    # color=
                    ".",
                    linewidth=0.0,
                    markersize=0.2,
                    alpha=0.7,
                    label=f"{phase}-phase",
                )

            # p_picks = picks_[picks_["phase_type"] == "P"]
            # s_picks = picks_[picks_["phase_type"] == "S"]
            # ps_picks = picks_[picks_["phase_type"] == "PS"]
            # sp_picks = picks_[picks_["phase_type"] == "SP"]
            # axs[0].plot(
            #     p_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
            #     p_picks["phase_index"] * dt[i],
            #     # ".C0",
            #     ".C2",
            #     # linewidth=5,
            #     linewidth=0.0,
            #     markersize=0.5,
            #     alpha=1.0,
            #     label="P-phase",
            # )
            # axs[0].plot(
            #     s_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
            #     s_picks["phase_index"] * dt[i],
            #     # "-C3",
            #     # ".C2",
            #     ".C0",
            #     # linewidth=5,
            #     linewidth=0.0,
            #     markersize=0.5,
            #     alpha=1.0,
            #     label="S-phase",
            # )

            # axs[0].plot(
            #     sp_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
            #     sp_picks["phase_index"] * dt[i],
            #     # ".C0",
            #     ".C1",
            #     # linewidth=5,
            #     linewidth=0.0,
            #     markersize=0.05,
            #     alpha=0.5,
            #     label="SP-phase",
            # )
            # axs[0].plot(
            #     ps_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
            #     ps_picks["phase_index"] * dt[i],
            #     # "-C3",
            #     # ".C2",
            #     ".C3",
            #     # linewidth=5,
            #     linewidth=0.0,
            #     markersize=0.05,
            #     alpha=0.5,
            #     label="PS-phase",
            # )

            axs[0].legend(markerscale=20.0)
            # axs[1].plot(p_picks["station_id"], p_picks["phase_index"], "r,", linewidth=0)
            # axs[1].plot(s_picks["station_id"], s_picks["phase_index"], "b,", linewidth=0)

        # im = axs[1].pcolormesh(
        #     (np.arange(nx) + begin_channel_index[i]) * dx[i] / 1e3,  # km
        #     (np.arange(nt) + begin_time_index[i]) * dt[i],
        #     pred[i, :, :, 0],
        #     vmin=0,
        #     vmax=1,
        #     cmap="hot",
        #     shading="auto",
        #     rasterized=True,
        # )
        im = axs[1].imshow(
            pred[i, :, :, :],
            extent=[
                begin_channel_index[i] * dx[i] / 1e3,
                (begin_channel_index[i] + nx) * dx[i] / 1e3,
                (begin_time_index[i] + nt) * dt[i],
                begin_time_index[i] * dt[i],
            ],
            aspect="auto",
            interpolation="none",
            origin="upper",
        )
        # axs[1].set_xlabel("Distance (km)")
        axs[1].set_ylabel("Time (s)")
        # axs[1].invert_yaxis()
        axs[1].xaxis.tick_top()
        # axs[1].xaxis.set_label_position("top")

        fig.tight_layout()
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
