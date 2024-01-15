# %%
import os
from pathlib import Path
import fsspec
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from tqdm.auto import tqdm
import random

np.random.seed(42)
random.seed(42)

# %%
protocol = "gs://"
bucket = "quakeflow_das"

# %%
figure_path = Path("debug_figures")
figure_path.mkdir(exist_ok=True)
plot_figure = False
# label_path = Path("results/training_v0")
# label_path = Path("results/training_v1")
label_path = Path("results/training_v2")
if label_path.exists():
    print(f"Warning: {label_path} exist!")
    raise FileExistsError
else:
    label_path.mkdir(parents=True)
fs = fsspec.filesystem(protocol.replace("://", ""))
folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south"]
# picker = "phasenet"
# picker = "phasenet_das"
picker = "phasenet_das_v1"


# %%
def get_eventid(filename):
    return filename.split("/")[-1].split("_")[-1].replace(".csv", "").replace(".h5", "")


def filter_noise(data_list, gamma_list):
    data = [get_eventid(file) for file in data_list]
    event = [get_eventid(file) for file in gamma_list]
    noise = set(data) - set(event)
    return list(noise)


# %%
for folder in folders:
    if not (label_path / folder / "labels").exists():
        (label_path / folder / "labels").mkdir(parents=True)

    data_list = list(fs.glob(f"{bucket}/{folder}/data/*.h5"))
    gamma_events = list(fs.glob(f"{bucket}/{folder}/gamma/{picker}/picks/*.csv"))
    print(f"{folder}: data {len(data_list)}, label {len(gamma_events)}")

    ## data list
    with open(label_path / f"{folder}" / "data.txt", "w") as f:
        tmp = [protocol + file for file in data_list]
        f.write("\n".join(tmp))

    ## noise list
    noise_list = filter_noise(data_list, gamma_events)
    with open(label_path / f"{folder}" / "noise.txt", "w") as f:
        noise_list = [f"{protocol}{bucket}/{folder}/data/{file}.h5" for file in noise_list]
        f.write("\n".join(noise_list))

    ## label list
    for file in tqdm(gamma_events):
        picks = pd.read_csv(protocol + file, parse_dates=["phase_time"])
        idx = picks["event_index"] != -1
        idx_p = idx & (picks["phase_type"] == "P")
        idx_s = idx & (picks["phase_type"] == "S")

        ## plot before filtering
        if plot_figure:
            fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
            ax[0, 0].scatter(picks[~idx]["station_id"], picks[~idx]["phase_index"], s=1, c="gray", alpha=0.1)
            ax[0, 0].scatter(picks[idx_p]["station_id"], picks[idx_p]["phase_index"], s=1, c="r")
            ax[0, 0].scatter(picks[idx_s]["station_id"], picks[idx_s]["phase_index"], s=1, c="b")

        ## filtering station_id with both P and S picks
        station_id_set = [
            set(picks[idx_p & (picks["event_index"] == event_index)]["station_id"].unique())
            & set(picks[idx_s & (picks["event_index"] == event_index)]["station_id"].unique())
            for event_index in picks[idx]["event_index"].unique()
        ]
        common_id = set.intersection(*station_id_set)
        if len(common_id) < 100:
            picks["event_index"] = -1
        picks.loc[~picks["station_id"].isin(common_id), "event_index"] = -1
        idx = picks["event_index"] != -1
        idx_p = idx & (picks["phase_type"] == "P")
        idx_s = idx & (picks["phase_type"] == "S")

        ## plot after filtering
        if plot_figure:
            ax[0, 1].scatter(picks[~idx]["station_id"], picks[~idx]["phase_index"], s=1, c="gray", alpha=0.1)
            ax[0, 1].scatter(picks[idx_p]["station_id"], picks[idx_p]["phase_index"], s=1, c="r")
            ax[0, 1].scatter(picks[idx_s]["station_id"], picks[idx_s]["phase_index"], s=1, c="b")
            plt.savefig(figure_path / f"{folder}_{os.path.basename(file)}.png", dpi=300)
            # plt.show()
            plt.close(fig)

        picks = picks[idx]
        if len(picks) > 0:
            # print(f'{label_path / f"{folder}" / "labels" / (get_eventid(file) + ".csv")}')
            picks.to_csv(label_path / f"{folder}" / "labels" / (get_eventid(file) + ".csv"), index=False)

# %%
labels = []
data = []
noise = []
for folder in folders:
    labels += list((label_path / folder / "labels").glob("*.csv"))
    with open(label_path / f"{folder}" / "data.txt", "r") as f:
        data += f.read().splitlines()
    with open(label_path / f"{folder}" / "noise.txt", "r") as f:
        noise += f.read().splitlines()

labels_train = sorted(random.sample(labels, int(len(labels) * 0.8)))
labels_test = sorted(list(set(labels) - set(labels_train)))
noise_train = sorted(random.sample(noise, int(len(noise) * 0.8)))
noise_test = sorted(list(set(noise) - set(noise_train)))
with open(label_path / "labels_train.txt", "w") as f:
    f.write("\n".join([str(file) for file in labels_train]))
with open(label_path / "labels_test.txt", "w") as f:
    f.write("\n".join([str(file) for file in labels_test]))
with open(label_path / "noise_train.txt", "w") as f:
    f.write("\n".join([str(file) for file in noise_train]))
with open(label_path / "noise_test.txt", "w") as f:
    f.write("\n".join([str(file) for file in noise_test]))
with open(label_path / "data.txt", "w") as f:
    f.write("\n".join(data))

# %%
