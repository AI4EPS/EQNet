# %%
import multiprocessing as mp
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
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
    # xlim_degree = [-120.0, -117.5]
    # ylim_degree = [37.1, 38.7]

    protocol = "gs://"
    bucket = "quakeflow_das"
    # fs = fsspec.filesystem(protocol.replace("://", ""))

    min_longitude = -120
    max_longitude = -117.5
    min_latitude = 37.1
    max_latitude = 38.7
    min_depth = 0
    max_depth = 25

    xlim_degree = [min_longitude, max_longitude]
    ylim_degree = [min_latitude, max_latitude]

    # %%
    # stations = pd.read_csv(f"das_info.csv")
    stations = pd.read_csv(f"{protocol}{bucket}/mammoth/das_info.csv")
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

    catalog = pd.read_csv(f"{protocol}{bucket}/mammoth_north/catalog_data.csv")
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
    centers[:, 2] += 3.0

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
    config["z(km)"] = [min_depth, max_depth]

    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 6.3, 6.7, 7.8]

    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.5
    vel = {"z": zz, "p": vp, "s": vs}
    config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

    config["eikonal"] = initialize_eikonal(config["eikonal"])

    # %%
    for ic, center in enumerate(centers):
        event_loc = center[np.newaxis, :].astype(np.float32)
        station_loc = stations[["x(km)", "y(km)", "z(km)"]].values.astype(np.float32)
        phase_type = ["p"] * len(station_loc) + ["s"] * len(station_loc)
        tt = traveltime(
            event_loc, np.concatenate([station_loc, station_loc], axis=0), np.array(phase_type), config["eikonal"]
        )

        ngrid = 50
        xgrid_ = np.linspace(np.floor(config["x(km)"][0]), np.ceil(config["x(km)"][1]), ngrid)
        ygrid_ = np.linspace(np.floor(config["y(km)"][0]), np.ceil(config["y(km)"][1]), ngrid)
        xgrid_degree_ = proj(xgrid_, np.zeros_like(xgrid_), inverse=True)[0]
        ygrid_degree_ = proj(np.zeros_like(ygrid_), ygrid_, inverse=True)[1]
        center_degree = proj(center[0], center[1], inverse=True)
        zgrid_ = np.linspace(config["z(km)"][0], config["z(km)"][1], 10)
        xgrid, ygrid, zgrid = np.meshgrid(xgrid_, ygrid_, zgrid_, indexing="ij")
        xgrid_degree, ygrid_degree, zgrid_degree = np.meshgrid(xgrid_degree_, ygrid_degree_, zgrid_, indexing="ij")

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

        # with mp.get_context("spawn").Pool(ncpu) as pool:
        #     for ii, (i, j, k) in enumerate(zip(xgrid.flatten(), ygrid.flatten(), zgrid.flatten())):
        #         pool.apply_async(
        #             run_proc,
        #             args=(idx, residual, ii, [i, j, k], tt, station_loc, phase_type, config_eikonal, lock),
        #             callback=lambda _: pbar.update(),
        #         )
        #     pool.close()
        #     pool.join()

        # residual = np.array(list(residual))
        # idx = np.array(list(idx))
        # residual = residual[np.argsort(idx)]
        # residual = residual.reshape(xgrid.shape)

        # # Save residual
        # np.savez(
        #     f"gridsearch_{ic}.npz",
        #     residual=residual,
        #     xgrid=xgrid,
        #     ygrid=ygrid,
        #     zgrid=zgrid,
        #     center=center,
        #     xgrid_degree=xgrid_degree,
        #     ygrid_degree=ygrid_degree,
        #     zgrid_degree=zgrid_degree,
        #     center_degree=center_degree,
        # )

        #     # %%
        #     fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 8))
        #     # im = ax[0, 0].pcolormesh(xgrid[:, :, 0], ygrid[:, :, 0], residual[:, :, 0], cmap="binary", shading="nearest")
        #     im = ax[0, 0].pcolormesh(
        #         xgrid_degree[:, :, 0],
        #         ygrid_degree[:, :, 0],
        #         residual[:, :, 0],
        #         cmap="binary",
        #         shading="nearest",
        #     )
        #     fig.colorbar(im)
        #     # ax[0, 0].scatter(center[0], center[1], s=100, c="r")
        #     ax[0, 0].scatter(center_degree[0], center_degree[1], s=100, c="r")

        #     cut_y = np.argmin(np.abs(ygrid[0, :, 0] - center[1]))
        #     im = ax[1, 0].pcolormesh(
        #         # xgrid[:, cut_y, :], zgrid[:, cut_y, :], residual[:, cut_y, :], cmap="binary", shading="nearest"
        #         xgrid_degree[:, cut_y, :],
        #         zgrid[:, cut_y, :],
        #         residual[:, cut_y, :],
        #         cmap="binary",
        #         shading="nearest",
        #     )
        #     fig.colorbar(im)
        #     # ax[1, 0].scatter(center[0], center[2], s=100, c="r")
        #     ax[1, 0].scatter(center_degree[0], center[2], s=100, c="r")

        #     cut_x = np.argmin(np.abs(xgrid[:, 0, 0] - center[0]))
        #     im = ax[2, 0].pcolormesh(
        #         # ygrid[cut_x, :, :], zgrid[cut_x, :, :], residual[cut_x, :, :], cmap="binary", shading="nearest"
        #         ygrid_degree[cut_x, :, :],
        #         zgrid[cut_x, :, :],
        #         residual[cut_x, :, :],
        #         cmap="binary",
        #         shading="nearest",
        #     )
        #     fig.colorbar(im)
        #     # ax[2, 0].scatter(center[1], center[2], s=100, c="r")
        #     ax[2, 0].scatter(center_degree[1], center[2], s=100, c="r")

        #     plt.savefig(f"gridsearch_{ic}.png")
        #     plt.show()

        #     # raise
        #     #     plt.figure()
        #     #     plt.plot(tt)
        #     #     plt.plot(tt_)
        #     #     plt.show()
        #     #     raise
        #     # raise

        # for ic, center in enumerate(centers):
        meta = np.load(f"gridsearch_{ic}.npz")
        residual = meta["residual"]
        # convert residual into probability
        residual = np.exp(-residual)
        residual = residual / residual.sum()

        xgrid = meta["xgrid"]
        ygrid = meta["ygrid"]
        zgrid = meta["zgrid"]
        center = meta["center"]

        protocol = "gs://"
        bucket = "quakeflow_das"
        das_mammoth_north = pd.read_csv(f"{protocol}{bucket}/mammoth_north/das_info.csv")
        das_mammoth_south = pd.read_csv(f"{protocol}{bucket}/mammoth_south/das_info.csv")

        figure_path = Path("paper_figures")
        if not figure_path.exists():
            figure_path.mkdir()

        min_longitude = -120
        max_longitude = -117.5
        min_latitude = 37.1
        max_latitude = 38.7
        min_depth = 0
        max_depth = 25
        cmap = "viridis"
        xlim = [min_longitude, max_longitude]
        ylim = [min_latitude, max_latitude]
        zlim = [min_depth, max_depth]

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(
                10,
                10
                * (max_latitude - min_latitude)
                / ((max_longitude - min_longitude) * np.cos(np.deg2rad(min_latitude))),
            ),
            gridspec_kw={"width_ratios": [4, 1], "height_ratios": [4, 1], "wspace": 0.05, "hspace": 0.05},
            # sharex=True,
            # sharey=True,
        )

        axes[0, 0].pcolormesh(
            xgrid_degree[:, :, 0],
            ygrid_degree[:, :, 0],
            residual[:, :, 0],
            cmap="binary",
            shading="nearest",
            rasterized=True,
        )
        ## add contour of probability
        axes[0, 0].contour(
            xgrid_degree[:, :, 0],
            ygrid_degree[:, :, 0],
            residual[:, :, 0],
            levels=[0.5],
            colors="r",
            rasterized=True,
        )
        axes[0, 0].scatter(center_degree[0], center_degree[1], s=100, c="r")
        axes[0, 0].scatter(
            das_mammoth_north["longitude"],
            das_mammoth_north["latitude"],
            s=1,
            c="C0",
            marker=".",
            alpha=0.5,
            rasterized=True,
        )
        axes[0, 0].scatter(
            das_mammoth_south["longitude"],
            das_mammoth_south["latitude"],
            s=1,
            c="C0",
            marker=".",
            alpha=0.5,
            rasterized=True,
            # label="DAS cable",
        )
        ## add legend
        axes[0, 0].scatter([], [], s=3, c="r", label="Event")
        axes[0, 0].scatter([], [], s=3, c="C0", label="DAS cable")
        axes[0, 0].legend(markerscale=5, loc="center left")

        axes[0, 0].set_aspect(1.0 / np.cos(np.deg2rad(min_latitude)))
        axes[0, 0].set_xlim(xlim)
        axes[0, 0].set_ylim(ylim)
        axes[0, 0].set_ylabel("Latitude")
        axes[0, 0].xaxis.set_label_position("top")
        axes[0, 0].xaxis.tick_top()
        axes[0, 0].set_xlabel("Longitude")

        cut_x = np.argmin(np.abs(xgrid[:, 0, 0] - center[0]))
        axes[0, 1].pcolormesh(
            # ygrid[cut_x, :, :], zgrid[cut_x, :, :], residual[cut_x, :, :], cmap="binary", shading="nearest"
            zgrid[cut_x, :, :],
            ygrid_degree[cut_x, :, :],
            residual[cut_x, :, :],
            cmap="binary",
            shading="nearest",
            rasterized=True,
        )
        axes[0, 1].scatter(center[2], center_degree[1], s=100, c="r")
        axes[0, 1].autoscale(tight=True)
        axes[0, 1].set_ylim(ylim)
        axes[0, 1].set_xlim([0, 25])
        # axes[0, 1].xaxis.set_label_position("top")
        # axes[0, 1].xaxis.tick_top()
        axes[0, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        axes[0, 1].set_xlabel("Depth (km)")

        axes[0, 1].yaxis.set_label_position("right")
        axes[0, 1].yaxis.tick_right()
        # axes[0, 1].set_ylabel("Latitude")

        cut_y = np.argmin(np.abs(ygrid[0, :, 0] - center[1]))
        axes[1, 0].pcolormesh(
            # xgrid[:, cut_y, :], zgrid[:, cut_y, :], residual[:, cut_y, :], cmap="binary", shading="nearest"
            xgrid_degree[:, cut_y, :],
            zgrid[:, cut_y, :],
            residual[:, cut_y, :],
            cmap="binary",
            shading="nearest",
            rasterized=True,
        )
        axes[1, 0].scatter(center_degree[0], center[2], s=100, c="r")
        axes[1, 0].autoscale(tight=True)
        axes[1, 0].set_xlim(xlim)
        axes[1, 0].set_ylim([0, 25])
        axes[1, 0].invert_yaxis()
        axes[1, 0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        axes[1, 0].set_ylabel("Depth (km)")
        # axes[1, 0].set_xlabel("Longitude")

        axes[1, 1].axis("off")
        # axes[1, 1].set_xlim([0,20])
        # axes[1, 1].set_ylim([0,20])
        # axes[1, 1].invert_yaxis()

        # fig.tight_layout()
        fig.savefig(figure_path / f"gridsearch_location_{ic}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(figure_path / f"gridsearch_location_{ic}.png", dpi=300, bbox_inches="tight")
        plt.show()

# %%
