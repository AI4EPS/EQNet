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
# root_dir = Path("/net/kuafu/mnt/tank/data/EventData/Ridgecrest_South/")
# root_dir = Path("/net/kuafu/mnt/tank/data/EventData/Ridgecrest/")

# %%
picks_path = root_dir / f"picks_phasenet_das/"
output_path = Path(f"picks_phasenet_filtered_debug2/")
figures_path = Path(f"figures_phasenet_filtered_debug2/")
if not output_path.exists():
    output_path.mkdir(parents=True)
if not figures_path.exists():
    figures_path.mkdir(parents=True)

# %%
plot_figure = True

# %% Match data format for GaMMA
stations = pd.read_csv(root_dir / "das_info.csv", index_col="index")
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
config = {"center": (x0, y0), "xlim_degree": [x0 - 5, x0 + 5], "ylim_degree": [y0 - 5, y0 + 5], "degree2km": degree2km}
config["dims"] = ["x(km)", "y(km)", "z(km)"]
config["use_dbscan"] = False
# config["dbscan_eps"] = 10.0
# config["dbscan_min_samples"] = 500
config["use_amplitude"] = False
config["x(km)"] = (
    (np.array(config["xlim_degree"]) - np.array(config["center"][0]))
    * config["degree2km"]
    * np.cos(np.deg2rad(config["center"][1]))
)
config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
config["z(km)"] = (0, 40)
config["vel"] = {"p": 6.0, "s": 6.0 / 1.73}
# config["vel"] = {"p": 5.5, "s": 5.5 / 1.73} ## Mammoth
config["covariance_prior"] = [1000, 1000]
config["method"] = "BGMM"
# config["method"] = "GMM"
if config["method"] == "BGMM":
    config["oversample_factor"] = 5
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
config["min_picks_per_eq"] = 2000  # len(stations)//10
# config["min_p_picks_per_eq"] = 500 #len(stations)//20
# config["min_s_picks_per_eq"] = 500 #len(stations)//20
config["max_sigma11"] = 2.0

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


# %%
def filter_close_false_picks(picks, events, tol=500):
    event_index = list(picks["event_index"].unique())
    num_events = len(event_index)
    mean_time = np.zeros([num_events, 2])
    events = pd.DataFrame(events)
    events["event_index"] = events["event_idx"]
    events.set_index("event_index", inplace=True)
    events_ = []
    for i, k in enumerate(event_index):
        mean_time[i, 0] = picks[(picks["event_index"] == k) & (picks["phase_type"] == "P")]["phase_index"].mean()
        mean_time[i, 1] = picks[(picks["event_index"] == k) & (picks["phase_type"] == "S")]["phase_index"].mean()
        events_.append(events.loc[k])
    events_ = pd.concat(events_, axis=1).T
    # index_selected = list(range(num_events))
    index_selected = event_index.copy()
    for i in range(num_events):
        for j in range(i + 1, num_events):
            if (np.abs(mean_time[i, 0] - mean_time[j, 0]) < tol) or (np.abs(mean_time[i, 1] - mean_time[j, 1]) < tol):
                if (
                    events_.loc[event_index[i]]["prob_gamma"] / events_.loc[event_index[i]]["sigma_time"]
                    > events_.loc[event_index[j]]["prob_gamma"] / events_.loc[event_index[j]]["sigma_time"]
                ):
                    if event_index[j] in index_selected:
                        index_selected.remove(event_index[j])
                else:
                    if event_index[i] in index_selected:
                        index_selected.remove(event_index[i])
    return picks[picks["event_index"].isin(index_selected)]


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

        ## filter: keep both P and S picks
        # for i, phase_type in enumerate(["P", "S"]):
        #     if i == 0:
        #         filt_picks = picks[(picks["event_index"] != -1) & (picks["phase_type"] == phase_type)][["station_id", "event_index"]]
        #     else:
        #         filt_picks = filt_picks.merge(picks[(picks["event_index"] != -1) & (picks["phase_type"] == phase_type)][["station_id", "event_index"]], how="inner", on=["station_id", "event_index"])
        # picks = picks.merge(filt_picks, on=["station_id", "event_index"], how='right')
        # if len(picks) == 0:
        #     continue

        ## filter very close events
        # picks = filter_close_false_picks(picks, events)
        if len(picks) == 0:
            continue

        picks.sort_values(by=["channel_index", "phase_index"], inplace=True)
        picks.to_csv(
            output_path.joinpath(file.name),
            index=False,
            columns=["channel_index", "phase_index", "phase_time", "phase_score", "phase_type", "event_index"],
            float_format="%.3f",
        )


# %%
if __name__ == "__main__":
    manager = Manager()
    event_list = manager.list()
    files = sorted(list(picks_path.rglob("*.csv")))
    # event_id = "ci39488679"
    # event_id = "ci39463055"
    # event_id = "ci39550567"
    # event_id = "nc73642105"
    # event_id = "nc73581316"
    # event_id = "nc73514575"
    # event_id = "nc73566210"
    # event_id = "nc73563905"
    # event_id = "nc73560470"
    # event_id = "nc73511770"
    # event_id = "nc73566210"
    # event_id = "ci39759432"
    # files = sorted(list(picks_path.rglob(f'{event_id}.csv')))

    # run(files, event_list)
    # raise

    jobs = []
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    for i in range(num_cores):
        p = multiprocessing.Process(target=run, args=(files[i::num_cores], event_list))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    if len(event_list) > 0:
        events = pd.DataFrame(list(event_list))

        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z(km)"]

        events.to_csv(
            "catalog_gamma_debug2.csv",
            index=False,
            columns=["event_id", "time", "longitude", "latitude", "depth_km", "event_index"],
            float_format="%.6f",
        )
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
