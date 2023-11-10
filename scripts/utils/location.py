# %%
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import multiprocessing as mp
from gamma.seismic_ops import initialize_eikonal, traveltime
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)


def filter_duplicates(df):
    # Sort the DataFrame
    df.sort_values(["station_id", "phase_index"], ascending=[True, True], inplace=True)

    # Define a custom function to merge rows
    def merge_rows(group):
        merged_rows = []
        previous_index = None

        for i, row in group.iterrows():
            if (previous_index is not None) and (abs(row["phase_index"] - previous_index) <= 10):
                merged_rows[-1] = row
            else:
                merged_rows.append(row)

            previous_index = row["phase_index"]

        return pd.DataFrame(merged_rows)

    # Group by station_id and apply the custom function
    df = df.groupby("station_id", group_keys=False).apply(merge_rows).reset_index(drop=True)
    return df


# %%
def calc_diff(
    event,
    stations,
    bucket,
    folder,
    picker,
    config,
    event_list,
    mag_list,
    dist_list,
    pdiff_mean_list,
    sdiff_mean_list,
    pdiff_std_list,
    sdiff_std_list,
    lock,
):
    # for i, (_, event) in enumerate(tqdm(catalog.iterrows(), total=len(catalog))):
    try:
        # picks = pd.read_csv(
        #     f"{bucket}/{folder}/{picker}/picks/{event.event_id}.csv", parse_dates=["phase_time"]
        # )
        picks = pd.read_csv(f"{bucket}/{folder}/gamma/{picker}/picks/{event.event_id}.csv", parse_dates=["phase_time"])
    except:
        return

    # picks = picks[picks["phase_score"] > 0.8]
    # if len(picks) < 500:
    #     continue
    # picks = filter_duplicates(picks)
    if "channel_index" in picks.columns:
        picks["station_id"] = picks["channel_index"]
    picks = picks[picks["event_index"] != -1]

    for event_index in picks["event_index"].unique():
        picks_ = picks[picks["event_index"] == event_index]

        event_loc = event[["x(km)", "y(km)", "z(km)"]].values[np.newaxis, :].astype(np.float32)
        station_loc = stations[["x(km)", "y(km)", "z(km)"]].values.astype(np.float32)
        phase_type = ["p"] * len(station_loc) + ["s"] * len(station_loc)
        tt = traveltime(
            event_loc, np.concatenate([station_loc, station_loc], axis=0), np.array(phase_type), config["eikonal"]
        )
        p_tt = tt[: len(station_loc)].flatten()
        s_tt = tt[len(station_loc) :].flatten()
        event_time = event["event_time"]
        p_time = event_time + pd.to_timedelta(p_tt, unit="s")
        s_time = event_time + pd.to_timedelta(s_tt, unit="s")
        dist_km = np.mean(np.sqrt(np.sum((station_loc - event_loc) ** 2, axis=1)))

        picks_ = picks_[picks_["station_id"].isin(stations["id"])]
        mapping = {x.id: i for i, x in stations.iterrows()}
        p_picks = picks_[picks_["phase_type"] == "P"]
        p_diff = (p_time[[mapping[x] for x in p_picks["station_id"]]] - p_picks["phase_time"]).dt.total_seconds()
        p_diff_mean = np.mean(p_diff)
        p_diff_std = np.std(p_diff)
        s_picks = picks_[(picks_["phase_type"] == "S")]
        s_diff = (s_time[[mapping[x] for x in s_picks["station_id"]]] - s_picks["phase_time"]).dt.total_seconds()
        s_diff_mean = np.mean(s_diff)
        s_diff_std = np.std(s_diff)
        # print(f"{dist_km = }, {p_diff_mean = }, {p_diff_std = }, {s_diff_mean = }, {s_diff_std = }")

        with lock:
            event_list.append(event.name)
            mag_list.append(event.magnitude)
            dist_list.append(dist_km)
            # pdiff_list.append(s_diff)
            # sdiff_list.append(s_diff)
            pdiff_mean_list.append(p_diff_mean)
            sdiff_mean_list.append(s_diff_mean)
            pdiff_std_list.append(p_diff_std)
            sdiff_std_list.append(s_diff_std)


if __name__ == "__main__":
    # %%
    ## Eikonal for 1D velocity model
    config = {}
    config["x(km)"] = [-500, 500]
    config["y(km)"] = [-500, 500]
    config["z(km)"] = [0, 30]

    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 6.3, 6.7, 7.8]
    # zz = [0.0, 5.5, 16.0, 35.0]
    # vp = [5.5, 6.3, 6.7, 7.8]
    # zz = [0.0, 1.0, 3.0, 4.0, 5.0, 17.0, 25.0]
    # vp = [3.2, 4.5, 4.8, 5.51, 6.21, 6.89, 7.83]

    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.5
    vel = {"z": zz, "p": vp, "s": vs}
    config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

    config["eikonal"] = initialize_eikonal(config["eikonal"])

    # %%
    root_path = Path("results/gamma")
    pickers = ["phasenet", "phasenet_das", "phasenet_das_v1"]
    pickers = pickers[::-1]
    # pickers = ["phasenet"]

    # %%
    folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south"]
    # folders = ["ridgecrest_south"]

    # %%
    bucket = "gs://quakeflow_das"
    station_csv = "das_info.csv"
    catalog_csv = "catalog_data.csv"

    # %%
    for folder in folders:
        stations = pd.read_csv(f"{bucket}/{folder}/{station_csv}")
        # stations = stations[~stations.isna().any(axis=1)]
        # stations.to_csv(f"{station_csv}", index=False)
        # print(f"{bucket}/{folder}/{station_csv}")
        # raise
        stations["id"] = stations["index"]

        catalog = pd.read_csv(f"{bucket}/{folder}/{catalog_csv}", parse_dates=["event_time"])
        catalog = catalog.set_index("event_id")
        catalog["event_id"] = catalog.index

        # %% Match data format for GaMMA
        y0 = stations["latitude"].mean()
        x0 = stations["longitude"].mean()
        proj = pyproj.Proj(f"+proj=sterea +lon_0={x0} +lat_0={y0} +units=km")
        stations[["x(km)", "y(km)"]] = stations.apply(
            lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
        )
        if "elevation_m" in stations.columns:
            stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)
        else:
            stations["z(km)"] = 0
        # stations["id"] = stations["id"].astype(str)

        catalog[["x(km)", "y(km)"]] = catalog.apply(
            lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
        )
        catalog["z(km)"] = catalog["depth_km"]

        # plt.figure(figsize=(8, 8))
        # plt.scatter(catalog["x(km)"], catalog["y(km)"], s=0.5, c="k", alpha=0.5)
        # plt.scatter(stations["x(km)"], stations["y(km)"], s=1, c="r")
        # # plt.savefig(result_path / "station_catalog.png", bbox_inches="tight", dpi=300)
        # plt.show()

        # %%
        max_diff = 10
        for picker in pickers:
            print(f"{folder = }, {picker = }")

            manager = mp.Manager()
            event_list = manager.list()
            mag_list = manager.list()
            dist_list = manager.list()
            # pdiff_list = manager.list()
            # sdiff_list = manager.list()
            pdiff_mean_list = manager.list()
            sdiff_mean_list = manager.list()
            pdiff_std_list = manager.list()
            sdiff_std_list = manager.list()
            lock = manager.Lock()

            pbar = tqdm(total=len(catalog))
            ncpu = mp.cpu_count()
            with mp.get_context("spawn").Pool(ncpu) as pool:
                for i, (_, event) in enumerate(catalog.iterrows()):
                    pool.apply_async(
                        calc_diff,
                        args=(
                            event,
                            stations,
                            bucket,
                            folder,
                            picker,
                            config,
                            event_list,
                            mag_list,
                            dist_list,
                            pdiff_mean_list,
                            sdiff_mean_list,
                            pdiff_std_list,
                            sdiff_std_list,
                            lock,
                        ),
                        callback=lambda _: pbar.update(),
                    )
                    # if i > 1000:
                    #     break
                pool.close()
                pool.join()

            # for _, event in catalog.iterrows():
            #     calc_diff(
            #         event,
            #         stations,
            #         bucket,
            #         folder,
            #         picker,
            #         config,
            #         event_list,
            #         mag_list,
            #         dist_list,
            #         pdiff_mean_list,
            #         sdiff_mean_list,
            #         pdiff_std_list,
            #         sdiff_std_list,
            #         lock,
            #     )

            dist_list = list(dist_list)
            event_list = list(event_list)
            mag_list = list(mag_list)
            # pdiff_list = list(pdiff_list)
            # sdiff_list = list(sdiff_list)
            pdiff_mean_list = list(pdiff_mean_list)
            sdiff_mean_list = list(sdiff_mean_list)
            pdiff_std_list = list(pdiff_std_list)
            sdiff_std_list = list(sdiff_std_list)

            dist_list = np.array(dist_list)
            pdiff_mean_list = np.array(pdiff_mean_list)
            sdiff_mean_list = np.array(sdiff_mean_list)
            pdiff_std_list = np.array(pdiff_std_list)
            sdiff_std_list = np.array(sdiff_std_list)
            index = (pdiff_std_list < 10) & (sdiff_std_list < 10)

            result_path = Path(f"results/stats/{folder}/{picker}")
            if not result_path.exists():
                result_path.mkdir(parents=True)

            plt.figure()
            plt.scatter(dist_list[index], pdiff_mean_list[index], s=3)
            plt.scatter(dist_list[index], sdiff_mean_list[index], s=3)
            # plt.savefig(f"{picker}_residual.png", bbox_inches="tight", dpi=300)
            plt.savefig(result_path / f"time_residual.png", bbox_inches="tight", dpi=300)

            # with open(f"{picker}_residual.csv", "w") as f:
            with open(result_path / f"time_residual.csv", "w") as f:
                f.write("event_id,dist_km,magnitude,pdiff_mean,pdiff_std,sdiff_mean,sdiff_std\n")
                for i in range(len(event_list)):
                    f.write(
                        f"{event_list[i]},{dist_list[i]},{mag_list[i]},{pdiff_mean_list[i]},{pdiff_std_list[i]},{sdiff_mean_list[i]},{sdiff_std_list[i]}\n"
                    )
