# %%
import multiprocessing
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
root_dir = Path("./")
result_path = Path(f"gamma/")
figures_path = Path(f"gamma/figures")
if not result_path.exists():
    result_path.mkdir(parents=True)
if not figures_path.exists():
    figures_path.mkdir(parents=True)
stations = pd.read_csv(root_dir / "DAS-LAX_coor_tap_test.csv").dropna()
stations["elevation_m"] = stations["elevation"]
stations["id"] = stations["channel"]
experiment = ""
picks_path = Path(f"picks_by_day")

# %%
plot_figure = True

# %% Match data format for GaMMA
y0 = stations["latitude"].mean()
x0 = stations["longitude"].mean()
proj = Proj(f"+proj=aeqd +lon_0={x0} +lat_0={y0} +units=km")
stations[["x(km)", "y(km)"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)
stations["id"] = stations["id"].astype(str)

# %% Setting for GaMMA
degree2km = 111.32
config = {"center": (x0, y0), "xlim_degree": [x0 - 2, x0 + 2], "ylim_degree": [y0 - 2, y0 + 2], "degree2km": degree2km}
config["dims"] = ["x(km)", "y(km)", "z(km)"]
config["use_dbscan"] = True
config["dbscan_eps"] = 10.0
config["dbscan_min_samples"] = 300
config["use_amplitude"] = False
config["x(km)"] = (
    (np.array(config["xlim_degree"]) - np.array(config["center"][0]))
    * config["degree2km"]
    * np.cos(np.deg2rad(config["center"][1]))
)
config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
config["z(km)"] = (0, 20)
config["vel"] = {"p": 6.0, "s": 6.0 / 1.73}
# config["vel"] = {"p": 5.5, "s": 5.5 / 1.73} ## Mammoth
config["covariance_prior"] = [1000, 1000]
config["method"] = "BGMM"
if config["method"] == "BGMM":
    config["oversample_factor"] = 4
if config["method"] == "GMM":
    config["oversample_factor"] = 1
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # x
    (None, None),  # t
)
config["initial_points"] = [2, 2, 1]  # x, y, z dimension

# Filtering
config["min_picks_per_eq"] = len(stations) // 5
# config["min_p_picks_per_eq"] = 500 #len(stations)//20
# config["min_s_picks_per_eq"] = 500 #len(stations)//20
config["max_sigma11"] = 2.0

# cpu
config["ncpu"] = 1

for k, v in config.items():
    print(f"{k}: {v}")


# %%
def associate(picks, stations, config):

    ## match format from PhaseNet to PhaseNet-DAS
    picks = picks.join(stations, on="channel_index", how="inner")

    ## match data format for GaMMA
    picks["id"] = picks["channel_index"].astype(str)
    picks["type"] = picks["phase_type"]
    picks["prob"] = picks["phase_score"]
    picks["timestamp"] = picks["phase_time"].apply(lambda x: datetime.fromisoformat(x))

    event_idx0 = 0
    catalogs, assignments = association(picks, stations, config, event_idx0, config["method"])

    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    picks = picks.join(assignments.set_index("pick_idx")).fillna(-1).astype({"event_idx": int})

    return catalogs, picks


def run(files, event_list):
    for file in files:
        try:
            picks = pd.read_csv(file)
        except:
            continue

        events, picks = associate(picks, stations, config)
        if (events is None) or (picks is None):
            continue

        for e in events:
            e["event_id"] = file.stem
            event_list.append(e)

        if plot_figure:
            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(20, 10))
            axs[0, 0].scatter(
                picks["channel_index"][picks["phase_type"] == "S"],
                picks["phase_index"][picks["phase_type"] == "S"],
                edgecolors=picks["event_idx"][picks["phase_type"] == "S"].apply(lambda x: f"C{x}" if x != -1 else "k"),
                s=10,
                marker="o",
                facecolors="none",
                linewidths=0.1,
            )
            axs[0, 0].scatter(
                picks["channel_index"][picks["phase_type"] == "P"],
                picks["phase_index"][picks["phase_type"] == "P"],
                c=picks["event_idx"][picks["phase_type"] == "P"].apply(lambda x: f"C{x}" if x != -1 else "k"),
                s=1,
                marker=".",
            )
            axs[0, 0].invert_yaxis()
            plt.savefig(figures_path.joinpath(file.stem + ".jpg"), bbox_inches="tight")
            plt.close(fig)

        ## match pick format of PhaseNet-DAS
        picks = picks[picks["event_idx"] != -1]
        if len(picks) == 0:
            continue
        picks["event_index"] = picks["event_idx"]
        picks["station_id"] = picks["channel_index"]

        if len(picks) == 0:
            continue

        picks.sort_values(by=["channel_index", "phase_index"], inplace=True)
        picks.to_csv(
            result_path.joinpath(file.name),
            index=False,
            columns=["channel_index", "phase_index", "phase_time", "phase_score", "phase_type", "event_index"],
            float_format="%.3f",
        )


# %%
if __name__ == "__main__":

    # # %%
    # files = sorted(list(picks_path.rglob('*.csv')))
    # event_list = []
    # run(files, event_list)

    # %%
    manager = Manager()
    event_list = manager.list()
    files = sorted(list(picks_path.rglob("*.csv")))

    jobs = []
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    with multiprocessing.Pool(num_cores) as pool:
        pool.starmap(run, [(files[i::num_cores], event_list) for i in range(num_cores)])

    if len(event_list) > 0:
        events = pd.DataFrame(list(event_list))
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z(km)"]
        events.to_csv(f"catalog_gamma{experiment}.csv", index=False, float_format="%.6f")
    else:
        print("No events found!")

# %%


# %%
# print(output_path.joinpath("events.csv"))

# %%
# plt.figure()
# plt.plot(picks["channel_index"][picks["phase_type"] == "P"], picks["phase_index"][picks["phase_type"] == "P"], '.')
# plt.plot(picks["channel_index"][picks["phase_type"] == "S"], picks["phase_index"][picks["phase_type"] == "S"], '.')
# plt.gca().invert_yaxis()
# plt.show()
# plt.figure()
# plt.plot(picks["x_km"][picks["phase_type"] == "P"], picks["phase_index"][picks["phase_type"] == "P"], '.')
# plt.plot(picks["x_km"][picks["phase_type"] == "S"], picks["phase_index"][picks["phase_type"] == "S"], '.')
# plt.gca().invert_yaxis()
# plt.show()
# plt.figure()
# plt.plot(picks["y_km"][picks["phase_type"] == "P"], picks["phase_index"][picks["phase_type"] == "P"], '.')
# plt.plot(picks["y_km"][picks["phase_type"] == "S"], picks["phase_index"][picks["phase_type"] == "S"], '.')
# plt.gca().invert_yaxis()
# plt.show()

# %%


# %%
