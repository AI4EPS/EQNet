# %%
import multiprocessing
import os
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
def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run GaMMA")
    parser.add_argument("--bucket", type=str, default="gs://das_mammoth")
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--picker", type=str, default="phasenet_das_v1")
    parser.add_argument("--station_csv", type=str, default="das_info.csv")
    parser.add_argument("--plot_figure", type=bool, default=True)
    parser.add_argument("--protocol", type=str, default="gs://")
    parser.add_argument("--overwrite", action="store_true", default=False)

    return parser


def set_config(args):
    # %%
    # stations = pd.read_csv(f"{args.station_csv}")
    stations = pd.read_csv(f"{args.protocol}{args.bucket}{args.folder}/das_info.csv")
    stations["id"] = stations["index"]

    # %% Match data format for GaMMA
    y0 = stations["latitude"].mean()
    x0 = stations["longitude"].mean()
    proj = Proj(f"+proj=sterea +lon_0={x0} +lat_0={y0} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    if "elevation_m" in stations.columns:
        stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)
    else:
        stations["z(km)"] = 0
    stations["id"] = stations["id"].astype(str)

    plt.figure(figsize=(8, 8))
    plt.scatter(stations["x(km)"], stations["y(km)"], s=1, c="r")
    plt.savefig(result_path / "das_info.png", bbox_inches="tight", dpi=300)

    # %% Setting for GaMMA
    degree2km = 111.32
    config = {
        "center": (x0, y0),
        "xlim_degree": [x0 - 5, x0 + 5],
        "ylim_degree": [y0 - 5, y0 + 5],
        "degree2km": degree2km,
    }
    config["dims"] = ["x(km)", "y(km)", "z(km)"]
    config["use_dbscan"] = True
    if args.picker == "phasenet":
        config["dbscan_eps"] = 10.0
    else:
        config["dbscan_eps"] = 30.0
    config["dbscan_min_samples"] = 1000  # len(stations) // 5
    config["use_amplitude"] = False
    # config["x(km)"] = (np.array(config["xlim_degree"])-np.array(config["center"][0]))*config["degree2km"]*np.cos(np.deg2rad(config["center"][1]))
    config["x(km)"] = (np.array(config["xlim_degree"]) - np.array(config["center"][0])) * config["degree2km"]
    config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
    config["z(km)"] = (0, 30)
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.73}
    # config["vel"] = {"p": 5.5, "s": 5.5 / 1.73}  ## Mammoth
    config["covariance_prior"] = [1000, 1000]  # Used for PhaseNet-DAS
    config["method"] = "BGMM"
    if config["method"] == "BGMM":
        if args.picker == "phasenet":
            config["oversample_factor"] = 4  # Used for PhaseNet
        else:
            config["oversample_factor"] = 2  # Used for PhaseNet-DAS
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
    config["min_picks_per_eq"] = 500  # len(stations) // 10
    # config["min_s_picks_per_eq"] = 10
    # config["min_p_picks_per_eq"] = 10
    if args.picker == "phasenet":
        config["max_sigma11"] = 1.0  # Used for PhaseNet
    else:
        config["max_sigma11"] = 0.5  # Used for PhaseNet-DAS

    # Eikonal
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 6.3, 6.7, 7.8]
    # vp_vs_ratio = 1.73
    # vs = [v / vp_vs_ratio for v in vp]
    # h = 0.5
    # vel = {"z": zz, "p": vp, "s": vs}
    # config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

    # cpu
    config["ncpu"] = 1

    for k, v in config.items():
        print(f"{k}: {v}")

    return config, stations, proj


# %%
def associate(picks, stations, config):
    picks = picks.join(stations, on="station_id", how="inner")

    ## match data format for GaMMA
    picks["id"] = picks["station_id"].astype(str)
    picks["type"] = picks["phase_type"]
    picks["prob"] = picks["phase_score"]
    picks["timestamp"] = picks["phase_time"].apply(lambda x: datetime.fromisoformat(x))

    event_idx0 = 0
    events, assignments = association(picks, stations, config, event_idx0, config["method"])

    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    picks = picks.join(assignments.set_index("pick_idx")).fillna(-1).astype({"event_idx": int})

    return events, picks


# def run(files, event_list, config, stations, result_path):
def run(files, config, stations, result_path, args, proj):
    for file in files:
        ## test init location
        # event_id = filename.split("_")[-1].replace(".h5.csv", "")
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

        filename = file.split("/")[-1]

        if os.path.exists(result_path / "picks" / filename) and (not args.overwrite):
            continue

        # if os.path.getsize(file) == 0:
        #     continue

        with fsspec.open(args.protocol + file, "rb") as f:
            if f.size == 0:
                print(f"Empty file: {file}")
                continue
            picks = pd.read_csv(f)

        picks = pd.read_csv(args.protocol + file)

        if args.picker == "phasenet":
            picks = picks[picks["phase_score"] > 0.5]  # used for PhaseNet
            # picks = picks[picks["phase_score"] > 0.8]
        else:
            picks = picks[picks["phase_score"] > 0.8]  # used for PhaseNet-DAS
        if len(picks) < config["min_picks_per_eq"]:
            continue

        if ("station_id" not in picks.columns) and ("channel_index" in picks.columns):
            picks["station_id"] = picks["channel_index"]

        events, picks = associate(picks, stations, config)
        if (events is None) or (picks is None):
            continue

        # for e in events:
        #     # e["event_id"] = filename.split("_")[-1].replace(".h5.csv", "")
        #     e["event_id"] = file.stem
        #     event_list.append(e)

        if len(picks[picks["event_idx"] != -1]) == 0:
            continue
        else:
            events = pd.DataFrame(events)
            events[["longitude", "latitude"]] = events.apply(
                lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
            )
            events["depth_km"] = events["z(km)"]
            events["filename"] = filename
            events.drop(columns=["x(km)", "y(km)", "z(km)"], inplace=True)
            events.to_csv(result_path / "events" / filename, index=False, float_format="%.7f")

        if args.plot_figure:
            plot_picks(result_path, filename.replace(".csv", ""), picks)

        picks["event_index"] = picks["event_idx"]
        picks.sort_values(by=["station_id", "phase_index"], inplace=True)
        picks.to_csv(
            result_path / "picks" / filename,
            index=False,
            columns=[
                "station_id",
                "phase_index",
                "phase_time",
                "phase_score",
                "phase_type",
                "event_index",
            ],
            float_format="%.3f",
        )


def plot_picks(result_path, filename, picks):
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
    idx = picks["event_idx"] == -1
    idx_p = (picks["phase_type"] == "P") & idx
    idx_s = (picks["phase_type"] == "S") & idx
    axs[0, 0].scatter(picks["station_id"][idx_p], picks["phase_index"][idx_p], c="k", s=1, marker=".", alpha=0.5)
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
        result_path / "figures" / (filename + ".jpg"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


# %%
if __name__ == "__main__":
    # # %%
    # pick_path = Path(f"results/{args.picker}/{args.folder}/picks_{args.picker}")
    # files = sorted(list(picks_path.rglob('*.csv')))
    # event_list = []
    # run(files, event_list)

    # %%
    args = get_args_parser().parse_args()
    # print(args)

    # %%
    # pick_path = Path(
    #     f"results/{args.picker}/{args.folder}/picks_{args.picker.replace('_v0', '').replace('_v1', '').replace('_v2', '')}"
    # )
    pick_path = Path(f"picks/{args.picker}/")
    files = sorted(list(pick_path.rglob("*.csv")), key=lambda x: -os.path.getsize(x))

    fs = fsspec.filesystem(args.protocol.replace("://", ""))
    files = fs.glob(f"{args.bucket}{args.folder}/{args.picker}/picks/*.csv")
    files = sorted(files, key=lambda x: -fs.size(x))

    # files = files[:100]

    # %%
    result_path = Path(f"results/gamma_10s/{args.picker}/{args.folder}")
    if not result_path.exists():
        result_path.mkdir(parents=True)
    if not (result_path / "picks").exists():
        (result_path / "picks").mkdir(parents=True)
    if not (result_path / "events").exists():
        (result_path / "events").mkdir(parents=True)
    if not (result_path / "figures").exists():
        (result_path / "figures").mkdir(parents=True)

    config, stations, proj = set_config(args)

    # %%
    manager = Manager()
    jobs = []
    # num_cores = max(1, multiprocessing.cpu_count() // 2)
    num_cores = max(1, multiprocessing.cpu_count() // 2 - 1)
    with multiprocessing.get_context("spawn").Pool(num_cores) as pool:
        # pool.starmap(run, [(files[i::num_cores], event_list, config, stations, result_path) for i in range(num_cores)])
        pool.starmap(run, [(files[i::num_cores], config, stations, result_path, args, proj) for i in range(num_cores)])

    # print(f"Number of events: {len(event_list)}")
    # if len(event_list) > 0:
    #     events = pd.DataFrame(list(event_list))
    #     events[["longitude", "latitude"]] = events.apply(
    #         lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
    #     )
    #     events["depth_km"] = events["z(km)"]
    #     events.drop(columns=["x(km)", "y(km)", "z(km)"], inplace=True)
    #     # events = pd.concat(event_list, ignore_index=True)
    #     events.to_csv(result_path / f"gamma_events.csv", index=False, float_format="%.6f")
    # else:
    #     print("No events found!")

    events = []
    for f in (result_path / "events").rglob("*.csv"):
        e = pd.read_csv(f)
        events.append(e)
    if len(events) > 0:
        events = pd.concat(events, ignore_index=True)
        events.to_csv(result_path / f"gamma_events.csv", index=False, float_format="%.7f")

    # cmd = f"gsutil -m cp -r {result_path}/picks {args.bucket}/{args.folder}/gamma/{args.picker}/picks"
    # cmd += f" && gsutil cp -r {result_path}/gamma_events.csv {args.bucket}/{args.folder}/gamma/{args.picker}/gamma_events.csv"
    # cmd += f" && gsutil cp {result_path}/station_catalog.png {args.bucket}/{args.folder}/gamma/{args.picker}/station_catalog.png"
    # print(cmd)
    # os.system(cmd)

# %%
