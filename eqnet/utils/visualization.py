import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random


def normalize(x):
    """x: [batch, channel, time, station]"""
    x = x - torch.mean(x, dim=2, keepdim=True)
    std = torch.std(x, dim=2, keepdim=True)
    std[std == 0] == 1
    x = x / std / 6
    return x


def visualize_autoencoder_das_train(meta, preds, epoch, figure_dir="figures"):
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


def visualize_das_train(meta, preds, epoch, figure_dir="figures", dt=0.01, dx=10, prefix=""):
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
            fig.savefig(f"{figure_dir}/{prefix}{epoch:02d}_{rank:02d}_{i:02d}.png", dpi=300, bbox_inches="tight")
        else:
            fig.savefig(f"{figure_dir}/{prefix}{epoch:02d}_{i:02d}.png", dpi=300, bbox_inches="tight")

        plt.close(fig)


def visualize_phasenet_train(meta, phase, event, polarity=None, epoch=0, figure_dir="figures"):
    for i in range(meta["waveform"].shape[0]):
        plt.close("all")
        fig, axes = plt.subplots(9, 1, figsize=(10, 10))
        chn_name = ["E", "N", "Z"]
        # chn_id = list(range(meta["waveform_raw"].shape[1]))
        # random.shuffle(chn_id)
        # for j in chn_id:
        #     if torch.max(torch.abs(meta["waveform_raw"][i, j, :, 0])) > 0.1:
        #         axes[0].plot(meta["waveform_raw"][i, j, :, 0], linewidth=0.5, color=f"C{j}", label=f"{chn_name[j]}")
        #         axes[0].legend(loc="upper right")
        #         axes[1].plot(meta["waveform"][i, j, :, 0], linewidth=0.5, color=f"C{j}", label=f"{chn_name[j]}")
        #         axes[1].legend(loc="upper right")
        #         break

        for j in range(3):
            axes[j].plot(meta["waveform_raw"][i, j, :, 0], linewidth=0.5, color="k", label=f"{chn_name[j]}")
            axes[j].set_xticklabels([])
            axes[j].grid("on")
        for j in range(3):
            axes[j + 3].plot(meta["waveform"][i, j, :, 0], linewidth=0.5, color="k", label=f"{chn_name[j]}")
            axes[j + 3].set_xticklabels([])
            axes[j + 3].grid("on")

        k = 6
        axes[k].plot(phase[i, 1, :, 0], "b")
        axes[k].plot(phase[i, 2, :, 0], "r")
        axes[k].plot(meta["phase_pick"][i, 1, :, 0], "--C0")
        axes[k].plot(meta["phase_pick"][i, 2, :, 0], "--C3")
        axes[k].plot(meta["phase_mask"][i, 0, :, 0], ":", color="gray")
        axes[k].set_ylim(-0.05, 1.05)
        axes[k].set_xticklabels([])
        axes[k].grid("on")

        axes[k + 1].plot(polarity[i, 0, :, 0], "b")
        axes[k + 1].plot(meta["polarity"][i, 0, :, 0], "--C0")
        axes[k + 1].plot(meta["polarity_mask"][i, 0, :, 0], ":", color="gray")
        axes[k + 1].set_ylim(-0.05, 1.05)
        axes[k + 1].set_xticklabels([])
        axes[k + 1].grid("on")

        axes[k + 2].plot(event[i, 0, :, 0], "b")
        axes[k + 2].plot(meta["event_center"][i, 0, :, 0], "--C0")
        axes[k + 2].plot(meta["event_mask"][i, 0, :, 0], ":", color="gray")
        axes[k + 2].set_ylim(-0.05, 1.05)
        # axes[k+2].set_xticklabels([])
        axes[k + 2].grid("on")

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}.png", dpi=300)

        if i >= 20:
            break


def plot_phasenet(
    meta,
    phase,
    event=None,
    polarity=None,
    picks=None,
    phases=None,
    dt=0.01,
    event_dt_ratio=16,
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

    waveform = meta["waveform"]
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

                mask = (phase[i, 1, ii : ii + nt, k] < 0.1) & (phase[i, 2, ii : ii + nt, k] < 0.1)
                axes[1].plot(t, np.ma.masked_where(mask, polarity[i, 0, ii : ii + nt, k] + k), "--C2", linewidth=1.0)

                # t_event = torch.arange(len(event[i, 0, ii//event_dt_ratio:(ii+nt)//event_dt_ratio, k])) * dt[i] * event_dt_ratio
                # axes[1].plot(t_event, event[i, 0, ii//event_dt_ratio:(ii+nt)//event_dt_ratio, k] + k, "-C3", linewidth=1.)

            axes[0].grid("on")
            axes[1].grid("on")

            # k = 2
            # axes[k].plot(phase[i, 1, ii:ii+nt, 0], "b")
            # axes[k].plot(phase[i, 2, ii:ii+nt, 0], "r")
            # axes[k].set_ylim(-0.05, 1.05)
            # axes[k].set_xticklabels([])
            # axes[k].grid("on")

            # axes[k+1].plot(polarity[i, 0, ii:ii+nt, 0], "b")
            # axes[k+1].set_ylim(-1.05, 1.05)
            # axes[k+1].set_xticklabels([])
            # axes[k+1].grid("on")

            # axes[k+2].plot(event[i, 0, ii//16:(ii+nt)//16, 0], "b")
            # axes[k+2].set_ylim(-0.05, 1.05)
            # axes[k+2].set_xticklabels([])
            # axes[k+2].grid("on")

            fig.tight_layout()

            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            fig.savefig(
                os.path.join(figure_dir, file_name[i].replace("/", "_") + f"_{ii:06d}.png"),
                bbox_inches="tight",
                dpi=300,
            )

            plt.close(fig)


def visualize_eqnet_train(meta, phase, event, epoch, figure_dir="figures", prefix="", offset=None, hypocenter=None):
    flag=True if (offset is not None) and (hypocenter is not None) else False
    for i in range(meta["data"].shape[0]):
        plt.close("all")
        if flag:
            #fig, axes = plt.subplots(5, 1, figsize=(10, 20), dpi=100)
            fig = plt.figure(figsize=(10, 18), dpi=150)
            axes=[]
            axes.append(plt.subplot2grid((6, 4), (0, 0), colspan=4))
            axes.append(plt.subplot2grid((6, 4), (1, 0), colspan=4))
            axes.append(plt.subplot2grid((6, 4), (2, 0), colspan=4))
            axes.append(plt.subplot2grid((6, 4), (3, 0), colspan=3, rowspan=2))
            axes.append(plt.subplot2grid((6, 4), (5, 0), colspan=3))
            axes.append(plt.subplot2grid((6, 4), (3, 3), colspan=1, rowspan=2))
        else:
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        for j in range(phase.shape[-1]):
            axes[0].plot((meta["data"][i, -1, :, j]) / torch.std(meta["data"][i, -1, :, j]) / 8 + j)

            axes[1].plot(phase[i, 1, :, j] + j, "r")
            axes[1].plot(phase[i, 2, :, j] + j, "b")
            axes[1].plot(meta["phase_pick"][i, 1, :, j] + j, "--C3")
            axes[1].plot(meta["phase_pick"][i, 2, :, j] + j, "--C0")

            axes[2].plot(event[i, :, j] + j, "b")
            axes[2].plot(meta["event_center"][i, :, j] + j, "--C0")
                
        # draw t, xy and xz scatter, show the station location, and the event location
        if flag:
            feature_scale=16
            sampling_rate=100.0
            event_center = (meta["event_center"][i,:,:].eq(1).float()*torch.arange(meta["event_center"][i].shape[-2])[:,None].float()).sum(axis=-2)
            true_station_number = event_center.shape[0]
            mask = meta["event_location_mask"][i]
            mask_divide = mask.sum(axis=-2)
            for k in range(mask_divide.shape[0]):
                if mask_divide[k] == 0:
                    mask_divide[k] = 1
                    true_station_number -= 1
            #assert abs((mask*torch.arange(mask.shape[-2])[:,None].float()).sum(axis=-2)/mask.sum(axis=-2) - event_center).max() <= 1
            dt = (hypocenter[i, 0,:,:]*mask)*sampling_rate
            dt = dt.sum(axis=-2)/mask_divide
            offset_pred = (offset[i, 0,:,:]*mask)
            offset_pred = offset_pred.sum(axis=-2)/mask_divide
            dt = dt + offset_pred*feature_scale
            
            dt_ground_truth = (meta["event_location"][i, 0,:,:]*mask)*sampling_rate
            dt_ground_truth = dt_ground_truth.sum(axis=-2)/mask_divide
            offset_ground_truth = (meta["event_location"][i, 4,:,:]*mask)
            offset_ground_truth = offset_ground_truth.sum(axis=-2)/mask_divide
            dt_ground_truth = dt_ground_truth + offset_ground_truth*feature_scale
            
            #axes[2].scatter((feature_scale*(event_center+offset_ground_truth))/feature_scale, np.arange(event_center.shape[-1]), c=np.arange(event_center.shape[-1]), s=50, marker=".")
            axes[2].scatter((feature_scale*(event_center+offset_pred)-dt)/feature_scale, np.arange(event_center.shape[-1]), c=np.arange(event_center.shape[-1]), s=50, marker="^")
            axes[2].scatter((feature_scale*(event_center+offset_ground_truth)-dt_ground_truth)/feature_scale, np.arange(event_center.shape[-1]), c=np.arange(event_center.shape[-1]), s=50, marker="*")
            axes[2].scatter(((feature_scale*(event_center+offset_pred)-dt).sum()/true_station_number)/feature_scale, event_center.shape[-1], c="C1", s=100, marker="^")
            axes[2].scatter(((feature_scale*(event_center+offset_ground_truth)-dt_ground_truth).sum()/true_station_number)/feature_scale, event_center.shape[-1], c="C0", s=100, marker="*")
            
            width_pred = (offset[i, 1,:,:]*mask)
            width_pred = width_pred.sum(axis=-2)/mask_divide
            width_ground_truth = (meta["event_location"][i, 5,:,:]*mask)
            width_ground_truth = width_ground_truth.sum(axis=-2)/mask_divide
            
            for k in range(len(event_center)):
                # use arrow to show the prediction
                axes[2].annotate("", xy=(event_center[k], k), xytext=(event_center[k]-width_pred[k], k), arrowprops=dict(color='red', arrowstyle='<-', mutation_scale=0.7))
                axes[2].annotate("", xy=(event_center[k], k), xytext=(event_center[k]+width_pred[k], k), arrowprops=dict(color='blue', arrowstyle='<-', mutation_scale=0.7))
                axes[2].annotate("", xy=(event_center[k], k), xytext=(event_center[k]+offset_pred[k], k), arrowprops=dict(color='green', arrowstyle='<-', mutation_scale=0.7))
                
                # use vertical line to show the ground truth
                axes[2].vlines(event_center[k]-width_ground_truth[k], k-0.2, k+0.2, color="C3", linestyle="--")
                axes[2].vlines(event_center[k]+width_ground_truth[k], k-0.2, k+0.2, color="C0", linestyle="--")
                axes[2].vlines(event_center[k]+offset_ground_truth[k], k-0.2, k+0.2, color="C2", linestyle="--")
            
            distance = (hypocenter[i, 1:,:,:]*mask[None, :, :])
            distance = distance.sum(axis=-2)/mask_divide[None,:]
            # if prediction doesn't have the depth, use 0
            if distance.shape[0]==2:
                distance = torch.cat([distance, torch.zeros_like(distance[0:1])], dim=0)
            distance_ground_truth = (meta["event_location"][i, 1:4,:,:]*mask[None, :, :])
            distance_ground_truth = distance_ground_truth.sum(axis=-2)/mask_divide[None, :]
            station_location = meta["station_location"][i]
            
            axes[3].scatter(station_location[:, 0], station_location[:, 1], c=np.arange(station_location.shape[0]), s=50, marker="^")
            axes[3].scatter(distance[0]+station_location[:, 0], distance[1]+station_location[:, 1], c=np.arange(station_location.shape[0]), s=50, marker=".", alpha=0.5)
            axes[3].scatter(distance_ground_truth[0]+station_location[:, 0], distance_ground_truth[1]+station_location[:, 1], c=np.arange(station_location.shape[0]), s=50, marker="*", alpha=0.5)
            axes[3].scatter((distance[0]+station_location[:, 0]).sum()/true_station_number, (distance[1]+station_location[:, 1]).sum()/true_station_number, c="C1", s=150, marker=".", alpha=0.7)
            axes[3].scatter((distance_ground_truth[0]+station_location[:, 0]).sum()/true_station_number, (distance_ground_truth[1]+station_location[:, 1]).sum()/true_station_number, c="C0", s=150, marker="*", alpha=0.7)
            
            axes[4].scatter(station_location[:, 0], station_location[:, 2], c=np.arange(station_location.shape[0]), s=50, marker="^")
            axes[4].scatter(distance[0]+station_location[:, 0], distance[2]+station_location[:, 2], c=np.arange(station_location.shape[0]), s=50, marker=".", alpha=0.5)
            axes[4].scatter(distance_ground_truth[0]+station_location[:, 0], distance_ground_truth[2]+station_location[:, 2], c=np.arange(station_location.shape[0]), s=50, marker="*", alpha=0.5)
            axes[4].scatter((distance[0]+station_location[:, 0]).sum()/true_station_number, (distance[2]+station_location[:, 2]).sum()/true_station_number, c="C1", s=150, marker=".", alpha=0.7)
            axes[4].scatter((distance_ground_truth[0]+station_location[:, 0]).sum()/true_station_number, (distance_ground_truth[2]+station_location[:, 2]).sum()/true_station_number, c="C0", s=150, marker="*", alpha=0.7)
            
            axes[5].scatter(station_location[:, 2], station_location[:, 1], c=np.arange(station_location.shape[0]), s=50, marker="^")
            axes[5].scatter(distance[2]+station_location[:, 2], distance[1]+station_location[:, 1], c=np.arange(station_location.shape[0]), s=50, marker=".", alpha=0.5)
            axes[5].scatter(distance_ground_truth[2]+station_location[:, 2], distance_ground_truth[1]+station_location[:, 1], c=np.arange(station_location.shape[0]), s=50, marker="*", alpha=0.5)
            axes[5].scatter((distance[2]+station_location[:, 2]).sum()/true_station_number, (distance[1]+station_location[:, 1]).sum()/true_station_number, c="C1", s=150, marker=".", alpha=0.7)
            axes[5].scatter((distance_ground_truth[2]+station_location[:, 2]).sum()/true_station_number, (distance_ground_truth[1]+station_location[:, 1]).sum()/true_station_number, c="C0", s=150, marker="*", alpha=0.7)

        axes[0].set_title("data")
        axes[1].set_title("phase_pick")
        axes[2].set_title("event_center")
        if flag:
            axes[3].set_title("hypocenter xy")
            axes[4].set_title("hypocenter xz")
            axes[5].set_title("hypocenter yz")
            #axes[3].set_aspect("equal")
            #axes[4].set_aspect("equal")
            #axes[5].set_aspect("equal")
            axes[4].sharex(axes[3])
            axes[5].sharey(axes[3])
            #axes[8].set_title("hypocenter dz")
        fig.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95)
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            fig.savefig(f"{figure_dir}/{prefix}{epoch:02d}_{i:02d}_{local_rank}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{prefix}{epoch:02d}_{i:02d}.png", dpi=300)


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

def plot_eqnet(meta, pred_phase, pred_event, phase_picks=None, event_picks=None, phases=["P", "S"], file_name=None, begin_time_index=None, dt=torch.tensor(0.01), figure_dir="./figures", feature_scale = 16, **kwargs):
    nt=60000
    nb0, nc0, nt0, ns0 = pred_phase.shape
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
            fig, axes = plt.subplots(4, 1, figsize=(20, 10))

            for k in range(ns0):
                begin_time_i = datetime.fromisoformat(begin_time[i])
                t = [
                    begin_time_i + timedelta(seconds=(ii + it) * dt) for it in range(len(pred_phase[i, 1, ii : ii + nt, k]))
                ]
                
                t_event = [
                    begin_time_i + timedelta(seconds=(ii + it) * dt*feature_scale) for it in range(len(pred_event[i, ii : ii + nt, k]))
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

                axes[1].plot(t, pred_phase[i, 1, ii : ii + nt, k] + k, "-C0", linewidth=1.0)
                axes[1].plot(t, pred_phase[i, 2, ii : ii + nt, k] + k, "-C1", linewidth=1.0)

                axes[1].plot(t_event, pred_event[i, ii : ii + nt, k] + k, "-C3", linewidth=1.0)

            axes[2].scatter(meta["station_location"][i, :, 0], meta["station_location"][i, :, 1], c=np.arange(ns0), s=50, marker="^")
            axes[3].scatter(meta["station_location"][i, :, 0], meta["station_location"][i, :, 2], c=np.arange(ns0), s=50, marker="^")
            for pick_dict in event_picks[i]:
                if ii > pick_dict["event_center_index"]*feature_scale or ii + nt < pick_dict["event_center_index"]*feature_scale:
                    continue
                sta_order = pick_dict["station_index"]
                axes[1].scatter(datetime.fromisoformat(pick_dict["event_original_time"]), sta_order[0], marker="^", s=50, c=sta_order[0])
                p_time = (datetime.fromisoformat(pick_dict["phase_time"])
                            + timedelta(seconds=((pick_dict["p_index"]-pick_dict["phase_index"])*dt*feature_scale)))
                s_time = (datetime.fromisoformat(pick_dict["phase_time"])
                            + timedelta(seconds=((pick_dict["s_index"]-pick_dict["phase_index"])*dt*feature_scale)))
                axes[1].vlines(p_time, sta_order[0]-0.5, sta_order[0]+0.5, color="r", linewidth=1.0)
                axes[1].vlines(s_time, sta_order[0]-0.5, sta_order[0]+0.5, color="b", linewidth=1.0)
                
                axes[2].scatter(pick_dict["event_location_x"], pick_dict["event_location_y"], marker=".", s=50, c=sta_order[0])
                axes[3].scatter(pick_dict["event_location_x"], pick_dict["event_location_z"], marker=".", s=50, c=sta_order[0])
            
            axes[0].grid("on")
            axes[1].grid("on")
            axes[2].grid("on")
            axes[3].grid("on")
            axes[0].set_title(f"{file_name[i]} data")
            axes[1].set_title("phase & event")
            axes[2].set_title("event location")

            fig.tight_layout()
            fig.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95)
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            fig.savefig(
                os.path.join(figure_dir, file_name[i].replace("/", "_") + f"_{ii:06d}.png"),
                bbox_inches="tight",
                dpi=300,
            )

            plt.close(fig)