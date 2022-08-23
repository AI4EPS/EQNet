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
picks_path = Path(root_dir + f"picks_phasenet_das/")
output_path = Path(f"picks_phasenet_filtered/")
figures_path = Path(f"figures_phasenet_filtered/")

if not output_path.exists():
    output_path.mkdir(parents=True)
if not figures_path.exists():
    figures_path.mkdir(parents=True)

# %% Match data format for GaMMA
stations = pd.read_csv(root_dir + "das_info.csv", index_col="index")
y0 = stations["latitude"].mean()
x0 = stations["longitude"].mean()
proj = Proj(f"+proj=sterea +lon_0={x0} +lat_0={y0} +units=km")
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
config["z(km)"] = (0, 20)
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
config["min_picks_per_eq"] = 20  # len(stations)//10
config["min_p_picks_per_eq"] = 10  # len(stations)//20
config["min_s_picks_per_eq"] = 10  # len(stations)//20
config["max_sigma11"] = 3.0

for k, v in config.items():
    print(f"{k}: {v}")


# %%
def associate(picks, stations, config):

    picks = picks.join(stations, on="channel_index", how="inner")

    ## match data format for GaMMA
    picks["id"] = picks["channel_index"].astype(str)
    picks["id"] = picks["id"].astype(str)
    picks["type"] = picks["phase_type"]
    picks["prob"] = picks["phase_score"]
    picks["timestamp"] = picks["phase_time"].apply(lambda x: datetime.fromisoformat(x))

    event_idx0 = 0  ## current earthquake index
    catalogs, assignments = association(picks, stations, config, event_idx0, config["method"])

    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    picks = picks.join(assignments.set_index("pick_idx")).fillna(-1).astype({"event_idx": int})

    return catalogs, picks


# %%
def run(files, event_list):
    for file in files:
        picks = pd.read_csv(file)
        # picks = picks[picks["phase_index"] > 10]
        # picks = picks[picks["phase_prob"] > 0.5]

        events, picks = associate(picks, stations, config)

        if (events is None) or (picks is None):
            continue
        for e in events:
            e["event_id"] = file.stem
            event_list.append(e)

        picks_ = picks.copy()
        picks["event_index"] = picks["event_idx"]
        picks = picks[picks["event_idx"] != -1]

        if len(picks) == 0:
            continue
        ## filter: keep both P/S picks
        picks["event_index"] = picks["event_idx"]
        picks["station_id"] = picks["channel_index"]

        picks.sort_values(by=["channel_index", "phase_time"], inplace=True)
        picks.to_csv(
            output_path.joinpath(file.name),
            index=False,
            # columns=["channel_index", "phase_index", "phase_time", "phase_score", "phase_type", "event_index"],
            columns=["channel_index", "phase_time", "phase_score", "phase_type", "event_index"],
            float_format="%.3f",
        )


# %%
if __name__ == "__main__":
    manager = Manager()
    event_list = manager.list()
    files = sorted(list(picks_path.rglob("*.csv")))

    files = [Path("mammoth.csv")]
    run(files, event_list)
    print(event_list)

    # jobs = []
    # num_cores = multiprocessing.cpu_count()
    # # num_cores = 1
    # for i in range(num_cores):
    #     p = multiprocessing.Process(target=run, args=(files[i::num_cores], event_list))
    #     jobs.append(p)
    #     p.start()
    # for p in jobs:
    #     p.join()

    events = pd.DataFrame(list(event_list))
    # events["event_index"] = events["event_idx"]

    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z(km)"]

    # events["longitude"] = events["x(km)"].apply(
    #     lambda x: x / config["degree2km"] + config["center"][0]
    # )
    # events["latitude"] = events["y(km)"].apply(
    #     lambda x: x / config["degree2km"] + config["center"][1]
    # )
    # events["depth_km"] = events["z(km)"]

    events.to_csv(
        "catalog_gamma.csv",
        index=False,
        columns=["event_id", "time", "longitude", "latitude", "depth_km", "event_index"],
        float_format="%.6f",
    )
