# %%
import multiprocessing as mp
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from gamma.seismic_ops import initialize_eikonal, traveltime
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

warnings.filterwarnings("error", category=RuntimeWarning)


# %%
def run_proc(idx, residual, ii, ijk, tt, station_loc, phase_type, config, lock):
    tt_ = traveltime(
        np.array([ijk]).astype(np.float32),
        station_loc,
        phase_type,
        config,
    )
    res = np.std(tt - tt_)

    with lock:
        idx.append(ii)
        residual.append(res)


# %%
if __name__ == "__main__":
    # %%
    xlim_degree = [-120.0, -117.5]
    ylim_degree = [37.1, 38.7]

    # %%
    stations = pd.read_csv(f"das_info.csv")
    stations["id"] = stations["index"]
    y0 = stations["latitude"].mean()
    x0 = stations["longitude"].mean()
    proj = pyproj.Proj(f"+proj=sterea +lon_0={x0} +lat_0={y0} +units=km")
    xlim_km = proj(xlim_degree[0], y0)[0], proj(xlim_degree[1], y0)[0]
    ylim_km = proj(x0, ylim_degree[0])[1], proj(x0, ylim_degree[1])[1]
    stations[["x(km)", "y(km)"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    if "elevation_m" in stations.columns:
        stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)
    else:
        stations["z(km)"] = 0
    stations["id"] = stations["id"].astype(str)

    catalog = pd.read_csv(f"gs://quakeflow_das/mammoth_north/catalog_data.csv")
    catalog = catalog[
        (catalog["longitude"] > xlim_degree[0])
        & (catalog["longitude"] < xlim_degree[1])
        & (catalog["latitude"] > ylim_degree[0])
        & (catalog["latitude"] < ylim_degree[1])
    ]
    catalog[["x(km)", "y(km)"]] = catalog.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    catalog["z(km)"] = catalog["depth_km"]

    # %%
    X = catalog[["x(km)", "y(km)", "z(km)"]].values
    db = DBSCAN(eps=5, min_samples=100).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    centers = []
    for i in range(n_clusters_):
        centers.append(np.mean(X[labels == i, :], axis=0))
    centers = np.array(centers)

    # %%
    plt.figure(figsize=(8, 8))
    plt.scatter(stations["x(km)"], stations["y(km)"], s=1, c="b")
    plt.scatter(catalog["x(km)"], catalog["y(km)"], s=0.5, c="k", alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c="r")
    plt.show()

    # %%
    ## Eikonal for 1D velocity model
    config = {}
    config["x(km)"] = xlim_km
    config["y(km)"] = ylim_km
    config["z(km)"] = [0, 30]

    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 6.3, 6.7, 7.8]

    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.5
    vel = {"z": zz, "p": vp, "s": vs}
    config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

    config["eikonal"] = initialize_eikonal(config["eikonal"])

    # %%
    for center in centers:
        event_loc = center[np.newaxis, :].astype(np.float32)
        station_loc = stations[["x(km)", "y(km)", "z(km)"]].values.astype(np.float32)
        phase_type = ["p"] * len(station_loc) + ["s"] * len(station_loc)
        tt = traveltime(
            event_loc, np.concatenate([station_loc, station_loc], axis=0), np.array(phase_type), config["eikonal"]
        )

        xgrid = np.arange(np.floor(config["x(km)"][0]), np.ceil(config["x(km)"][1]), 2)
        ygrid = np.arange(np.floor(config["y(km)"][0]), np.ceil(config["y(km)"][1]), 2)
        zgrid = np.arange(config["z(km)"][0], config["z(km)"][1], 2)
        xgrid, ygrid, zgrid = np.meshgrid(xgrid, ygrid, zgrid, indexing="ij")

        # residuals = []
        # for i, j, k in tqdm(zip(xgrid.flatten(), ygrid.flatten(), zgrid.flatten()), total=len(xgrid.flatten())):
        #     tt_ = traveltime(
        #         np.array([[i, j, k]]).astype(np.float32),
        #         np.concatenate([station_loc, station_loc], axis=0),
        #         np.array(phase_type),
        #         config["eikonal"],
        #     )
        #     res = np.std(tt - tt_)
        #     residuals.append(res)

        # for ii, (i, j, k) in enumerate(zip(xgrid.flatten(), ygrid.flatten(), zgrid.flatten())):
        #     process(idx, residual, ii, [i, j, k], tt, station_loc, phase_type, config_eikonal, lock)
        #     pbar.update()

        manager = mp.Manager()
        idx = manager.list()
        residual = manager.list()
        lock = manager.Lock()
        ncpu = mp.cpu_count()
        print(f"{ncpu = }")
        pbar = tqdm(total=len(xgrid.flatten()))
        station_loc = np.concatenate([station_loc, station_loc], axis=0)
        phase_type = np.array(phase_type)
        config_eikonal = config["eikonal"]

        with mp.get_context("spawn").Pool(ncpu) as pool:
            for ii, (i, j, k) in enumerate(zip(xgrid.flatten(), ygrid.flatten(), zgrid.flatten())):
                pool.apply_async(
                    run_proc,
                    args=(idx, residual, ii, [i, j, k], tt, station_loc, phase_type, config_eikonal, lock),
                    callback=lambda _: pbar.update(),
                )
            pool.close()
            pool.join()

        residual = np.array(list(residual))
        idx = np.array(list(idx))
        residual = residual[np.argsort(idx)]
        residual = residual.reshape(xgrid.shape)

        # %%
        fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 8))
        im = ax[0, 0].pcolormesh(xgrid[:, :, 0], ygrid[:, :, 0], residual[:, :, 0], cmap="binary", shading="nearest")
        fig.colorbar(im)
        ax[0, 0].scatter(center[0], center[1], s=100, c="r")

        cut_y = np.argmin(np.abs(ygrid[0, :, 0] - center[1]))
        im = ax[1, 0].pcolormesh(
            xgrid[:, cut_y, :], zgrid[:, cut_y, :], residual[:, cut_y, :], cmap="binary", shading="nearest"
        )
        fig.colorbar(im)
        ax[1, 0].scatter(center[0], center[2], s=100, c="r")

        cut_x = np.argmin(np.abs(xgrid[:, 0, 0] - center[0]))
        im = ax[2, 0].pcolormesh(
            ygrid[cut_x, :, :], zgrid[cut_x, :, :], residual[cut_x, :, :], cmap="binary", shading="nearest"
        )
        fig.colorbar(im)
        ax[2, 0].scatter(center[1], center[2], s=100, c="r")

        plt.savefig(f"debug_{center[0]:.1f}.png")
        plt.show()

        #     plt.figure()
        #     plt.plot(tt)
        #     plt.plot(tt_)
        #     plt.show()
        #     raise
        # raise

    # %%
