# %%
import os
from pathlib import Path
import fsspec
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from tqdm.auto import tqdm

# %%
protocol = "gs"
bucket = "gs://quakeflow_das"

# %%
figure_path = Path("debug_figures")
figure_path.mkdir(exist_ok=True)
plot_figure = False
pick_path = Path("results/training_v0")
pick_path.mkdir(exist_ok=True)


# %%
max_dt = 100
min_picks = 50
regressor = linear_model.RANSACRegressor(min_samples=10, residual_threshold=100, random_state=42, max_trials=1000)
# regressor = linear_model.TheilSenRegressor(n_subsamples=2, random_state=42)

fs = fsspec.filesystem(protocol)
folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south"]
for folder in folders:
    data_list = fs.glob(f"{bucket}/{folder}/data/*.h5")
    gamma_events = fs.glob(f"{bucket}/{folder}/gamma/picks/*.csv")
    print(f"{folder}: {len(data_list)} {len(gamma_events)}")

    # for file in fs.glob(f"{bucket}/{folder}/gamma/picks/*.csv"):
    for file in tqdm(gamma_events, total=len(gamma_events)):
        # print(file)
        picks = pd.read_csv(protocol + "://" + file, parse_dates=["phase_time"])
        idx = picks["event_index"] != -1
        idx_p = idx & (picks["phase_type"] == "P")
        idx_s = idx & (picks["phase_type"] == "S")

        if plot_figure:
            fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
            ax[0, 0].scatter(picks[~idx]["station_id"], picks[~idx]["phase_index"], s=1, c="gray", alpha=0.1)
            ax[0, 0].scatter(picks[idx_p]["station_id"], picks[idx_p]["phase_index"], s=1, c="r")
            ax[0, 0].scatter(picks[idx_s]["station_id"], picks[idx_s]["phase_index"], s=1, c="b")

        # ## correcting missing S picks for distant events
        # X, y = picks[~idx]["station_id"].values[:, np.newaxis], picks[~idx]["phase_index"].values[:, np.newaxis]
        # regressor.fit(X, y)
        # y_ = regressor.predict(picks[~idx]["station_id"].values[:, np.newaxis])
        # idx_y = (np.abs(y - y_) < max_dt).squeeze()
        # num_match = len(y[idx_y])
        # if (len(picks[idx_s]) < min_picks) and (len(picks[~idx][idx_y]) > 100):
        #     picks.loc[picks[~idx][idx_y].index, "event_index"] = 1
        #     picks.loc[picks[~idx][idx_y].index, "phase_type"] = "S"
        #     ax[0, 1].scatter(picks[~idx][idx_y]["station_id"], y[idx_y], s=10, c="green", alpha=0.1)
        #     idx = picks["event_index"] != -1
        #     idx_p = idx & (picks["phase_type"] == "P")
        #     idx_s = idx & (picks["phase_type"] == "S")

        # ### filtering too many picks that can be modeled by a linear regression, maybe missing association
        # if num_match > 500:
        #     print(f"{file}: {num_match} picks matched by linear regression, maybe missing association")
        #     picks["event_index"] = -1
        #     idx = picks["event_index"] != -1
        #     idx_p = idx & (picks["phase_type"] == "P")
        #     idx_s = idx & (picks["phase_type"] == "S")

        ## filtering station_id with both P and S picks
        common_station_id = set(picks[idx_p]["station_id"].unique()) & set(picks[idx_s]["station_id"].unique())
        if len(common_station_id) < 100:
            picks["event_index"] = -1
        else:
            # print(f"{len(common_station_id) = }")
            pass
        picks.loc[~picks["station_id"].isin(common_station_id), "event_index"] = -1
        idx = picks["event_index"] != -1
        idx_p = idx & (picks["phase_type"] == "P")
        idx_s = idx & (picks["phase_type"] == "S")

        ## plot after filtering
        idx = picks["event_index"] != -1
        idx_p = idx & (picks["phase_type"] == "P")
        idx_s = idx & (picks["phase_type"] == "S")

        if plot_figure:
            ax[0, 1].scatter(picks[~idx]["station_id"], picks[~idx]["phase_index"], s=1, c="gray", alpha=0.1)
            ax[0, 1].scatter(picks[idx_p]["station_id"], picks[idx_p]["phase_index"], s=1, c="r")
            ax[0, 1].scatter(picks[idx_s]["station_id"], picks[idx_s]["phase_index"], s=1, c="b")
            plt.savefig(figure_path / f"{folder}_{os.path.basename(file)}.png", dpi=300)
            # plt.show()
            plt.close(fig)

        picks = picks[idx]
        if len(picks) > 0:
            picks.to_csv(pick_path / os.path.basename(file), index=False)

# %%
