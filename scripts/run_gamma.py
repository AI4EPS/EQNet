# %%
import multiprocessing
import warnings
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gamma.utils import association
from pyproj import Proj

warnings.filterwarnings("ignore")

# %%
bucket = "gs://quakeflow_das"
folder = "mammoth_north"
station_csv = "das_info.csv"
catalog_csv = "catalog.csv"
# das_info = pd.read_csv(f"{bucket}/{folder}/{station_csv}")

# %%
protocol = "gs"
fs = fsspec.filesystem(protocol)

# %%
root_dir = Path("./")
picks_path = Path(f"results/phasenet/{folder}/picks_phasenet")
stations = pd.read_csv(f"{bucket}/{folder}/{station_csv}")
catalog = pd.read_csv(f"{bucket}/{folder}/{catalog_csv}", parse_dates=["event_time"])
catalog = catalog.set_index("event_id")
stations["id"] = stations["index"]

# %%
result_path = Path(f"results/gamma/{folder}")
if not result_path.exists():
    result_path.mkdir(parents=True)
if not (result_path / "picks").exists():
    (result_path / "picks").mkdir(parents=True)
if not (result_path / "figures").exists():
    (result_path / "figures").mkdir(parents=True)

# %%
plot_figure = True

# %% Match data format for GaMMA
y0 = stations["latitude"].mean()
x0 = stations["longitude"].mean()
proj = Proj(f"+proj=sterea +lon_0={x0} +lat_0={y0} +units=km")
stations[["x(km)", "y(km)"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)
stations["id"] = stations["id"].astype(str)

catalog[["x(km)", "y(km)"]] = catalog.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
catalog["z(km)"] = catalog["depth_km"]

plt.figure(figsize=(8, 8))
plt.scatter(stations["x(km)"], stations["y(km)"], s=1, c="r")
plt.scatter(catalog["x(km)"], catalog["y(km)"], s=1, c="k")
plt.savefig(result_path / "station_catalog.png", bbox_inches="tight", dpi=300)

# %% Setting for GaMMA
degree2km = 111.32
config = {"center": (x0, y0), "xlim_degree": [x0 - 5, x0 + 5], "ylim_degree": [y0 - 5, y0 + 5], "degree2km": degree2km}
config["dims"] = ["x(km)", "y(km)", "z(km)"]
config["use_dbscan"] = True
config["dbscan_eps"] = 30.0
config["dbscan_min_samples"] = len(stations) // 5
config["use_amplitude"] = False
# config["x(km)"] = (np.array(config["xlim_degree"])-np.array(config["center"][0]))*config["degree2km"]*np.cos(np.deg2rad(config["center"][1]))
config["x(km)"] = (np.array(config["xlim_degree"]) - np.array(config["center"][0])) * config["degree2km"]
config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
config["z(km)"] = (0, 30)
# config["vel"] = {"p": 6.0, "s": 6.0 / 1.73}
config["vel"] = {"p": 5.5, "s": 5.5 / 1.73}  ## Mammoth
# config["covariance_prior"] = [1000, 1000]
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
config["initial_points"] = [1, 1, 1]  # x, y, z dimension

# Filtering
config["min_picks_per_eq"] = len(stations) // 10
# config["min_s_picks_per_eq"] = 10
# config["min_p_picks_per_eq"] = 10
config["max_sigma11"] = 1.0

# cpu
config["ncpu"] = 1

for k, v in config.items():
    print(f"{k}: {v}")


# %%
def associate(picks, stations, config):
    picks = picks.join(stations, on="station_id", how="inner")

    ## match data format for GaMMA
    picks["id"] = picks["station_id"].astype(str)
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
        ## test init location
        # event_id = file.name.split("_")[-1].replace(".h5.csv", "")
        # config["x_init"] = [catalog.loc[event_id, "x(km)"]]
        # config["y_init"] = [catalog.loc[event_id, "y(km)"]]
        # config["z_init"] = [catalog.loc[event_id, "z(km)"]]
        # dx = 80
        # dy = 80
        # config["bfgs_bounds"] = (
        #     (catalog.loc[event_id, "x(km)"] - dx/2, catalog.loc[event_id, "x(km)"] + dx/2),  # x
        #     (catalog.loc[event_id, "y(km)"] - dy/2, catalog.loc[event_id, "y(km)"] + dy/2),  # y
        #     (0, config["z(km)"][1] + 1),  # z
        #     (None, None),  # t
        # )

        try:
            picks = pd.read_csv(file)
            picks = picks[picks["phase_score"] > 0.5]
        except Exception as e:
            print(e)
            continue

        events, picks = associate(picks, stations, config)
        if (events is None) or (picks is None):
            continue

        for e in events:
            # e["event_id"] = file.name.split("_")[-1].replace(".h5.csv", "")
            e["event_id"] = file.stem
            event_list.append(e)

        if len(picks[picks["event_idx"] != -1]) == 0:
            continue

        if plot_figure:
            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
            idx = picks["event_idx"] == -1
            idx_p = (picks["phase_type"] == "P") & idx
            idx_s = (picks["phase_type"] == "S") & idx
            axs[0, 0].scatter(
                picks["station_id"][idx_p], picks["phase_index"][idx_p], c="k", s=1, marker=".", alpha=0.5
            )
            axs[0, 0].scatter(
                picks["station_id"][idx_s],
                picks["phase_index"][idx_s],
                edgecolors="k",
                facecolors="none",
                s=10,
                marker="o",
                linewidths=0.2,
            )
            idx = picks["event_idx"] != -1
            idx_p = (picks["phase_type"] == "P") & idx
            idx_s = (picks["phase_type"] == "S") & idx
            axs[0, 0].scatter(
                picks["station_id"][idx_p],
                picks["phase_index"][idx_p],
                c=picks["event_idx"][idx_p].apply(lambda x: f"C{x}"),
                s=1,
                marker=".",
                alpha=0.5,
            )
            axs[0, 0].scatter(
                picks["station_id"][idx_s],
                picks["phase_index"][idx_s],
                edgecolors=picks["event_idx"][idx_s].apply(lambda x: f"C{x}"),
                facecolors="none",
                s=10,
                marker="o",
                linewidths=0.2,
            )
            axs[0, 0].invert_yaxis()
            plt.savefig(
                # result_path / "figures" / (file.name.split("_")[-1].replace(".h5.csv", "") + ".jpg"),
                result_path / "figures" / (file.stem + ".jpg"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)


        picks["event_index"] = picks["event_idx"]
        picks.sort_values(by=["station_id", "phase_index"], inplace=True)
        picks.to_csv(
            # result_path / "picks" / (file.name.split("_")[-1].replace(".h5.csv", "") + ".csv"),
            result_path / "picks" / file.name,
            index=False,
            columns=["station_id", "phase_index", "phase_time", "phase_score", "phase_type", "event_index"],
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

    files = files[:10]

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
        events.drop(columns=["x(km)", "y(km)", "z(km)"], inplace=True)
        events.to_csv(result_path / f"events_gamma.csv", index=False, float_format="%.6f")
    else:
        print("No events found!")

