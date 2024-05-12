# %%
import os
from functools import partial
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


# %%
def read_labels(file):
    label_list = []
    with h5py.File(file, "r") as fp:
        event_ids = list(fp.keys())
        for event_id in tqdm(event_ids):
            event_attrs = dict(fp[event_id].attrs)
            station_ids = list(fp[event_id].keys())
            # for station_id in station_ids:
            #     label.append(fp[event_id][station_id]["label"][()])
            for station_id in station_ids:
                station_attrs = dict(fp[event_id][station_id].attrs)
                num_picks = len(station_attrs["phase_time"])
                labels = {
                    "event_index": [event_id] * num_picks,
                    "station_id": [station_id] * num_picks,
                    "phase_time": list(station_attrs["phase_time"]),
                    "phase_type": list(station_attrs["phase_type"]),
                    "phase_polarity": list(station_attrs["phase_polarity"]),
                }
                label_list.append(pd.DataFrame(labels))
        labels = pd.concat(label_list, ignore_index=True)
        labels["phase_time"] = pd.to_datetime(labels["phase_time"])

    return labels


# %%
if __name__ == "__main__":

    # %% Load origin PhaseNet model
    reload = False
    region = "NC"
    phasenet_csv = f"phasenet_picks_{region}.csv"
    if reload or not os.path.exists(phasenet_csv):
        if region == "NC":
            phasenet_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/models/phasenet_picks.csv"
            )
        elif region == "SC":
            phasenet_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_sc/resolve/main/models/phasenet_picks.csv"
            )
        # phasenet_picks = pd.read_csv("phasenet_origin/picks_sc.csv")
        # phasenet_picks[["event_index", "station_id"]] = phasenet_picks["file_name"].str.split("/", expand=True)
        # phasenet_picks.drop(columns=["file_name", "begin_time"], inplace=True)
        # phasenet_picks.rename(columns={"event_index": "event_id"}, inplace=True)
        # phasenet_picks["phase_time"] = pd.to_datetime(phasenet_picks["phase_time"])
        # phasenet_picks["phase_time"] = phasenet_picks["phase_time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        # phasenet_picks.to_csv(
        #     "phasenet_picks.csv",
        #     columns=["event_id", "station_id", "phase_time", "phase_score", "phase_type"],
        #     index=False,
        # )
    else:
        phasenet_picks = pd.read_csv(phasenet_csv)
    phasenet_picks["event_index"] = phasenet_picks["event_id"]
    phasenet_picks["phase_time"] = pd.to_datetime(phasenet_picks["phase_time"])

    # %% Load PhaseNet+ model
    phasenet_plus_csv = f"phasenet_plus_picks_{region}.csv"
    if reload or not os.path.exists(phasenet_plus_csv):
        if region == "NC":
            phasenet_plus_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/models/phasenet_plus_picks.csv"
            )
        elif region == "SC":
            phasenet_plus_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_sc/resolve/main/models/phasenet_plus_picks.csv"
            )
        # pick_path = "../../results_ps_test_sc/picks_phasenet_plus"
        # event_ids = glob(f"{pick_path}/*")
        # picks_list = []
        # for event_id in tqdm(event_ids):
        #     station_ids = glob(f"{event_id}/*.csv")
        #     for station_id in station_ids:
        #         if os.stat(station_id).st_size == 0:
        #             # print(f"Empty file: {station_id}")
        #             continue
        #         phasenet_plus_picks = pd.read_csv(station_id)
        #         phasenet_plus_picks["event_index"] = event_id.split("/")[-1]
        #         picks_list.append(phasenet_plus_picks)
        # phasenet_plus_picks = pd.concat(picks_list, ignore_index=True)
        # phasenet_plus_picks["phase_time"] = pd.to_datetime(phasenet_plus_picks["phase_time"])
        # phasenet_plus_picks.drop(columns=["dt_s"], inplace=True)
        # phasenet_plus_picks = filter_duplicates(phasenet_plus_picks)
        # phasenet_plus_picks.drop(columns=["phase_index"], inplace=True)
        # phasenet_plus_picks.rename(columns={"event_index": "event_id"}, inplace=True)
        # phasenet_plus_picks.to_csv(
        #     phasenet_plus_csv,
        #     columns=["event_id", "station_id", "phase_time", "phase_score", "phase_type", "phase_polarity"],
        #     index=False,
        # )
    else:
        phasenet_plus_picks = pd.read_csv(phasenet_plus_csv, parse_dates=["phase_time"])

    phasenet_plus_picks["event_index"] = phasenet_plus_picks["event_id"]
    phasenet_plus_picks["phase_time"] = pd.to_datetime(phasenet_plus_picks["phase_time"])

    # %% Load labels
    label_csv = f"labels_{region}.csv"
    if not os.path.exists(label_csv):
        if region == "NC":
            labels = read_labels(file="/nfs/quakeflow_dataset/NC/quakeflow_nc/waveform_test.h5")
        elif region == "SC":
            labels = read_labels(file="/nfs/quakeflow_dataset/SC/quakeflow_sc/waveform_test.h5")
        labels.to_csv(label_csv, index=False)
    else:
        labels = pd.read_csv(label_csv, parse_dates=["phase_time"])

    # %%
    score_threshold = 0.5
    polarity_threshold = 0.5
    phase_type = "P"
    time_tolerance = 0.1
    picks = phasenet_plus_picks.copy()
    picks_ = picks[(picks["phase_score"] > score_threshold) & (picks["phase_type"] == phase_type)].copy()
    labels_ = labels[(labels["phase_type"] == phase_type)].copy()
    picks_.drop(columns=["phase_type"], inplace=True)
    labels_.drop(columns=["phase_type"], inplace=True)
    if "phase_polarity" in picks_.columns:
        picks_.loc[:, "phase_polarity"] = picks_["phase_polarity"].map(
            lambda x: "D" if x < -polarity_threshold else "N" if x < polarity_threshold else "U"
        )

    merged = pd.merge(labels_, picks_, how="left", on=["station_id", "event_index"], suffixes=("_true", "_pred"))

    # %%
    def check_phase_polarity(row, tolerance=time_tolerance):
        return (
            # abs((row[f"phase_time_true"] - row[f"phase_time_pred"]).total_seconds()) <= tolerance
            # and row[f"phase_polarity_true"] == row[f"phase_polarity_pred"]
            abs((row[f"phase_time_true"] - row[f"phase_time_pred"]).total_seconds()) <= tolerance
            and row[f"phase_polarity_true"] != row[f"phase_polarity_pred"]
            and (row[f"phase_polarity_pred"] != "N")
        )

    merged["check"] = merged.apply(check_phase_polarity, axis=1)

    # %%
    # no space between ax
    fig, ax = plt.subplots(1, 1, figsize=(6, 2), sharex=True, gridspec_kw={"hspace": 0.0})
    dt = 0.01
    figure_path = "./figures_errors"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    with h5py.File("/nfs/quakeflow_dataset/NC/quakeflow_nc/waveform_test.h5", "r") as fp:
        for i, row in tqdm(merged.iterrows(), total=merged.shape[0]):
            # if (not row["correct"]) and (row["phase_polarity_pred"] != row["phase_polarity_true"]):
            if row["check"]:
                data = fp[f"{row['event_id']}/{row['station_id']}"]
                begin_time = pd.to_datetime(fp[f"{row['event_id']}"].attrs["begin_time"])
                end_time = pd.to_datetime(fp[f"{row['event_id']}"].attrs["end_time"])
                t = pd.date_range(begin_time, end_time, freq=f"{dt}S", inclusive="left")

                idx1 = int(((row["phase_time_true"] - begin_time).total_seconds() - 0.5) / dt)
                idx2 = int(((row["phase_time_true"] - begin_time).total_seconds() + 1.5) / dt)

                # ax[0].clear()
                # ax[1].clear()
                # ax[2].clear()
                # ax[0].set_title(f"{row['event_id']}/{row['station_id']}")
                # ax[0].plot(t[idx1:idx2], data[0, idx1:idx2], color="k", label="E")
                # ax[0].legend(loc="upper right")
                # ax[0].grid(axis="x")
                # ax[1].plot(t[idx1:idx2], data[1, idx1:idx2], color="k", label="N")
                # ax[1].legend(loc="upper right")
                # ax[1].grid(axis="x")

                ax.clear()
                ax.plot(t[idx1:idx2], data[-1, idx1:idx2], color="k", label="Z")
                ax.axvline(
                    row["phase_time_true"],
                    color="r",
                    linestyle="--",
                    label=f"Label: {row['phase_polarity_true']}",
                    alpha=0.5,
                )
                ax.axvline(
                    row["phase_time_pred"],
                    color="g",
                    linestyle="--",
                    label=f"Pred: {row['phase_polarity_pred']} ({row['phase_score']:.1f})",
                    alpha=0.5,
                )
                ax.legend(loc="upper right")
                ax.grid(axis="x")
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M:%S"))

                fig.savefig(f"{figure_path}/{row['event_id']}_{row['station_id']}.png", dpi=300, bbox_inches="tight")
                # raise


# %%
