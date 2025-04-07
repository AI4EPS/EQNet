# %%
import multiprocessing
import sys
import warnings
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gamma.utils import association
from pyproj import Proj

warnings.filterwarnings("ignore")

# %%
root_dir = "./"

# %%
data_path = Path("./")
picks_path = Path("./")
output_path = Path("./")
figures_path = Path("./figures")

if not output_path.exists():
    output_path.mkdir(parents=True)
if not figures_path.exists():
    figures_path.mkdir(parents=True)

# %% Match data format for GaMMA
stations = pd.read_csv(data_path / "das_info.csv", index_col="index")
y0 = stations["latitude"].mean()
x0 = stations["longitude"].mean()
proj = Proj(f"+proj=aeqd +lon_0={x0} +lat_0={y0} +units=km")
stations[["x(km)", "y(km)"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)
stations["id"] = stations.index
stations["id"] = stations["id"].astype(str)

# %% Setting for GaMMA
degree2km = 111.32
config = {"center": (x0, y0), "xlim_degree": [x0 - 2, x0 + 2], "ylim_degree": [y0 - 2, y0 + 2], "degree2km": degree2km}
config["dims"] = ["x(km)", "y(km)", "z(km)"]
config["use_dbscan"] = True
config["dbscan_eps"] = 10
config["dbscan_min_samples"] = 3
config["use_amplitude"] = False
config["x(km)"] = (np.array(config["xlim_degree"]) - np.array(config["center"][0])) * config["degree2km"]
config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
config["z(km)"] = (0, 30)
# config["vel"] = {"p": 6.0, "s": 6.0 / 1.73}
config["vel"] = {"p": 5.5, "s": 5.5 / 1.73}  ## Mammoth
config["method"] = "BGMM"
# config["method"] = "GMM"
if config["method"] == "BGMM":
    # config["oversample_factor"] = 5
    config["oversample_factor"] = 5
if config["method"] == "GMM":
    config["oversample_factor"] = 1
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # x
    (0, None),  # t
)
config["initial_mode"] = "one_point"

# Filtering
config["min_picks_per_eq"] = 200  # len(stations)//10
config["min_p_picks_per_eq"] = 100  # len(stations)//20
config["min_s_picks_per_eq"] = 100  # len(stations)//20
config["max_sigma11"] = 2.0

for k, v in config.items():
    print(f"{k}: {v}")


# %%
def associate(picks, stations, config):

    picks = picks.join(stations, on="channel_index", how="inner")

    ## match data format for GaMMA
    picks["id"] = picks["channel_index"].astype(str)
    # picks["id"] = picks["id"].astype(str)
    picks["type"] = picks["phase_type"]
    picks["prob"] = picks["phase_score"]
    picks["timestamp"] = picks["phase_time"].apply(lambda x: datetime.fromisoformat(x))

    event_idx0 = 0  ## current earthquake index
    catalogs, assignments = association(picks, stations, config, event_idx0, config["method"])

    assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
    picks = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({"event_index": int})

    return catalogs, picks


# %%
def run(i, picks, stations, event_list, picks_list):

    print(f"Start split {i}")
    # picks = picks[picks["phase_index"] > 10]
    # picks = picks[picks["phase_prob"] > 0.5]

    # event_list = []
    # picks_list = []

    # print(f"{i} step1 : picks: {len(picks)}")

    events, picks = associate(picks, stations, config)

    # print(f"{i} step2 : events: {len(events)}")
    # print(f"{i} step2 : picks: {len(picks)}")

    if events is not None:
        for e in events:
            e["event_id"] = f"{i:04d}_{e['event_index']:04d}" if e["event_index"] != -1 else "-1"
            event_list.append(e)

    # picks = picks[picks["event_idx"] != -1]
    if len(picks) > 0:
        picks["event_id"] = picks["event_index"].apply(lambda x: f"{i:04d}_{x:04d}" if x != -1 else "-1")
        picks.sort_values(by=["channel_index", "phase_time"], inplace=True)
        picks_list.append(
            picks[["channel_index", "phase_time", "phase_score", "phase_type", "gamma_score", "event_id"]]
        )

    print(f"Finished split {i}")
    # events = pd.DataFrame(list(event_list))
    # picks = pd.concat(picks_list)

    # events.to_csv(
    #     f"gamma_catalog_{i:02d}_wintermute.csv",
    #     index=False,
    # )

    # picks.to_csv(
    #     f"gamma_picks_{i:02d}_wintermute.csv",
    #     index=False,
    # )

    # print(f"{i} step3: events: {len(events)}")
    # print(f"{i} step3: picks: {len(picks)}")


# %%
if __name__ == "__main__":
    manager = Manager()
    event_list = manager.list()
    picks_list = manager.list()

    picks = pd.read_csv(picks_path / "mammoth_picks.csv")

    # num_cores = multiprocessing.cpu_count()
    # split_index = np.array_split(np.arange(len(picks)), num_cores)

    picks.sort_values(by=["phase_time"], inplace=True)
    picks.reset_index(drop=True, inplace=True)

    # picks.iloc[split_index[3]].to_csv("debug.csv")

    # picks = pd.read_csv("debug.csv")
    # print("read debug.csv")

    # raise
    # print(f"{len(picks) = }")
    # picks = picks.iloc[:10000000]
    # picks = picks.iloc[:467650]
    # raise

    # run(0, picks, event_list, picks_list)
    # print(event_list)
    # print(picks_list)
    # raise

    # num_cores = multiprocessing.cpu_count()
    # num_splits = 64 * 2
    # num_cores = len(picks) // (10000000//64)
    # num_splits = 96
    num_splits = len(picks) // 300_000
    print(f"{num_splits = }")
    split_index = np.array_split(np.arange(len(picks)), num_splits)

    # run(3, picks, event_list, picks_list)
    jobs = []
    for i in range(num_splits):
        # for i in range(3, 4):
        # for i in range(1, 2):

        p = multiprocessing.Process(
            target=run, args=(i, picks.iloc[split_index[i]].copy(), stations, event_list, picks_list)
        )
        jobs.append(p)
        p.start()

        if len(jobs) >= multiprocessing.cpu_count() // 2:
            for p in jobs:
                p.join()
            jobs = []

    for p in jobs:
        p.join()

    # # print(event_list)
    events = pd.DataFrame(list(event_list))
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z(km)"]
    mapping = {events.loc[i, "event_id"]: i + 1 for i in events.index}
    mapping["-1"] = -1
    events["event_id"] = events.index + 1

    picks = pd.concat(picks_list)
    picks["event_id"] = picks["event_id"].map(mapping)
    picks["event_id"] = picks["event_id"].astype(int)

    events.to_csv(
        "gamma_catalog2.csv",
        index=False,
        columns=["event_id", "time", "longitude", "latitude", "depth_km", "sigma_time", "gamma_score"],
        float_format="%.6f",
    )

    picks.to_csv(
        "gamma_picks2.csv",
        index=False,
    )
