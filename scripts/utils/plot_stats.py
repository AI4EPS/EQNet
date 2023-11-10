# %%
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import fsspec
import matplotlib.patches as mpatches
from sklearn.linear_model import RANSACRegressor
import pyproj
from tqdm.auto import tqdm

plt.rcParams["axes.xmargin"] = 0

#########
# root_path = Path("results")
# paths = ["training_v0", "training_v1", "training_v2"]
# root_path = Path("results/gamma")
# paths = ["phasenet_das_v1"]
# %%
# %%
# snr_df = pd.read_csv(result_path / f"snr_list.txt", header=None, names=["h5", "snr"])
# snr_df["event_id"] = snr_df["h5"].apply(lambda x: x.split("/")[-1].split(".")[0])
# plt.figure()
# plt.hist(snr_df["snr"], bins=100)
# plt.yscale("log")
# plt.xlabel("SNR (dB)")
# plt.ylabel("Frequency")
# plt.show()

# %%
protocol = "gs://"
bucket = "quakeflow_das"
fs = fsspec.filesystem(protocol.replace("://", ""))

pickers = ["phasenet", "phasenet_das", "phasenet_das_v1"]
pickers = pickers[::-1]
picker_name = {
    "phasenet": "PhaseNet",
    "phasenet_das": "PhaseNet-DAS v1",
    "phasenet_das_v1": "PhaseNet-DAS v2",
}

folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south"]

figure_path = Path("paper_figures")
if not figure_path.exists():
    figure_path.mkdir(parents=True)

# %%

# # %%
# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# for folder in folders:
#     for picker in pickers:
#         stats = pd.read_csv(f"stats_{picker}_{folder}.txt")

#         stats = stats[stats["num_associated"] >= 500]

#         bins = range(min(stats["num_picks"]), max(stats["num_picks"]) + 1, 100)
#         ax[0, 0].hist(stats["num_picks"], bins=50, alpha=0.5)
#         ax[0, 1].hist(stats["num_associated"], bins=50, alpha=0.5)
#         ax[1, 0].hist(stats["num_associated_P"], bins=50, alpha=0.5)
#         ax[1, 1].hist(stats["num_associated_S"], bins=50, alpha=0.5)

# # set y log scale
# ax[0, 0].set_yscale("log")
# ax[0, 1].set_yscale("log")
# ax[1, 0].set_yscale("log")
# ax[1, 1].set_yscale("log")
# plt.show()
# # raise

# %%
for folder in folders:
    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(10, 2.5))
    for j, picker in enumerate(pickers):
        events = pd.read_csv(f"{protocol}{bucket}/{folder}/gamma/{picker}/gamma_events.csv")
        if j == 0:
            bins = np.linspace(min(events["num_picks"]), max(events["num_picks"]), 25)
            bins_p = np.linspace(min(events["num_p_picks"]), max(events["num_p_picks"]), 25)
            bins_s = np.linspace(min(events["num_p_picks"]), max(events["num_p_picks"]), 25)
        ax[0, 0].hist(
            events["num_picks"],
            #   bins=25,
            bins=bins,
            alpha=0.5,
            edgecolor="white",
        )
        ax[0, 1].hist(
            events["num_p_picks"],
            # bins=25,
            bins=bins_p,
            alpha=0.5,
            edgecolor="white",
        )
        if picker == "phasenet":
            label = f"{picker_name[picker]}\n{events['num_picks'].sum()/1e6:.1f}M: {events['num_p_picks'].sum()/1e6:.1f}M(P)+{events['num_s_picks'].sum()/1e6:.1f}M(S)"
        else:
            label = f"{picker_name[picker]}\n{events['num_picks'].sum()/1e6:.0f}M: {events['num_p_picks'].sum()/1e6:.0f}M(P)+{events['num_s_picks'].sum()/1e6:.0f}M(S)"
        ax[0, 2].hist(
            events["num_s_picks"],
            # bins=25,
            bins=bins_s,
            alpha=0.5,
            # label=picker_name[picker],
            label=label,
            edgecolor="white",
        )
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_yscale("log")
    ax[0, 2].set_yscale("log")
    ax[0, 0].set_title("P+S picks")
    ax[0, 1].set_title("P picks")
    ax[0, 2].set_title("S picks")
    ax[0, 2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    ax[0, 0].set_ylabel("Number of events")
    ax[0, 1].set_xlabel("Number of picks per event")
    fig.savefig(figure_path / f"num_picks_{folder}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_path / f"num_picks_{folder}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# %%
fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 6))
for i, folder in enumerate(folders):
    for j, picker in enumerate(pickers):
        events = pd.read_csv(f"{protocol}{bucket}/{folder}/gamma/{picker}/gamma_events.csv")
        if j == 0:
            min_ = round(min(events["sigma_time"]) ** 0.5, 1)
            max_ = round(max(events["sigma_time"]) ** 0.5, 1)
            bins = np.linspace(min_, max_, (int(max_ * 10 - min_ * 10) * 4 + 1))
        ax[i // 2, i % 2].hist(
            events["sigma_time"].apply(lambda x: x**0.5),
            #   bins=25,
            bins=bins,
            alpha=0.5,
            edgecolor="white",
            label=picker_name[picker],
        )
    ax[i // 2, i % 2].set_yscale("log")

    ax[i // 2, i % 2].text(
        0.05,
        0.95,
        f"({chr(ord('a') + i)})",
        transform=ax[i // 2, i % 2].transAxes,
        fontsize=14,
        verticalalignment="top",
    )

ax[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
fig.text(x=0.05, y=0.5, s="Number of events", ha="center", va="center", rotation="vertical", fontsize=14)
fig.text(
    x=0.5,
    y=0.05,
    s="Standard deviation of arrival-time residuals (s)",
    ha="center",
    va="center",
    fontsize=14,
)
fig.savefig(figure_path / "sigma_time.png", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / "sigma_time.pdf", dpi=300, bbox_inches="tight")


# %%
ylim = {
    "mammoth_north": [-14, 7],
    "mammoth_south": [-14, 7],
    "ridgecrest_north": [-7, 3.5],
    "ridgecrest_south": [-14, 7],
}
xlim = {
    "mammoth_north": [-5, 12],
    "mammoth_south": [-5, 20],
    "ridgecrest_north": [-5, 17],
    "ridgecrest_south": [-5, 10],
}

fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 6))

for i, folder in enumerate(folders):
    result_path = Path(f"results/stats/{folder}")

    snr_df = pd.read_csv(f"{protocol}{bucket}/training/stats/{folder}/snr.txt", header=None, names=["h5", "snr"])
    snr_df["event_id"] = snr_df["h5"].apply(lambda x: x.split("/")[-1].split(".")[0])
    # # plt.figure()
    # # plt.hist(snr_df["snr"], bins=100)
    # # plt.yscale("log")
    # # plt.xlabel("SNR (dB)")
    # # plt.ylabel("Frequency")
    # # plt.show()

    for picker in pickers:
        # detected_event = fs.glob(f"{bucket}/{folder}/gamma/{picker}/picks/*csv")
        # detected_event_id = [e.split("/")[-1].split("_")[-1].split(".")[0] for e in detected_event]
        # snr_df[f"{picker}"] = snr_df["event_id"].apply(lambda x: x in detected_event_id)

        events = pd.read_csv(f"{protocol}{bucket}/training/stats/{folder}/{picker}/time_residual.csv")

        events.loc[events["pdiff_mean"].isna(), "pdiff_mean"] = events.loc[events["pdiff_mean"].isna(), "sdiff_mean"]
        events.loc[events["sdiff_mean"].isna(), "sdiff_mean"] = events.loc[events["sdiff_mean"].isna(), "pdiff_mean"]
        events["mean"] = events[["pdiff_mean", "sdiff_mean"]].mean(axis=1)
        events = events.loc[events.groupby("event_id")["mean"].idxmin()]

        events = events.loc[
            (events["pdiff_mean"] > ylim[folder][0])
            & (events["pdiff_mean"] < ylim[folder][1])
            & (events["sdiff_mean"] > ylim[folder][0])
            & (events["sdiff_mean"] < ylim[folder][1])
        ]

        snr_df[f"{picker}"] = snr_df["event_id"].apply(lambda x: x in events["event_id"].values)

    bins = np.linspace(-10, 25, 35 + 1)
    ax[i // 2, i % 2].hist(
        snr_df["snr"],
        bins=bins,
        alpha=0.5,
        edgecolor="white",
        facecolor="gray",
        label="Dataset",
        # label=f"{len(snr_df['snr'][snr_df[picker] & (snr_df['snr'] >= xlim[folder][0]) & (snr_df['snr'] <= xlim[folder][1])])}",
    )

    for j, picker in enumerate(pickers):
        # plt.hist(snr_df["snr"][snr_df[picker]], bins=bins, label=picker, alpha=0.5, edgecolor="white")
        ax[i // 2, i % 2].hist(
            snr_df["snr"][snr_df[picker]],
            bins=bins,
            alpha=0.5,
            edgecolor="white",
            facecolor=f"C{j}",
            label=picker_name[picker],
            # label=f"{len(snr_df['snr'][snr_df[picker] & (snr_df['snr'] >= xlim[folder][0]) & (snr_df['snr'] <= xlim[folder][1])])}",
        )
    ax[i // 2, i % 2].set_yscale("log")
    ax[i // 2, i % 2].text(
        0.05,
        0.95,
        f"({chr(ord('a') + i)})",
        transform=ax[i // 2, i % 2].transAxes,
        fontsize=14,
        verticalalignment="top",
    )
    ax[i // 2, i % 2].set_xticks(bins[2::4])
    ax[i // 2, i % 2].set_xlim(xlim[folder])
    # ax[i // 2, i % 2].legend(loc="upper right")

    # manually add legend by adding text box
    patches = []
    for j, picker in enumerate(pickers):
        patch = mpatches.Patch(
            color=f"C{j}",
            label=f"{len(snr_df['snr'][snr_df[picker] & (snr_df['snr'] >= xlim[folder][0]) & (snr_df['snr'] <= xlim[folder][1])])}",
            alpha=0.5,
        )
        patches.append(patch)
    legend = ax[i // 2, i % 2].legend(handles=patches, loc="upper right")
    ax[i // 2, i % 2].add_artist(legend)


legend = ax[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
fig.text(x=0.05, y=0.5, s="Number of events", ha="center", va="center", rotation="vertical", fontsize=14)
fig.text(x=0.5, y=0.05, s="SNR (dB)", ha="center", va="center", fontsize=14)
fig.savefig(figure_path / "snr.png", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / "snr.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# ylim = {
#     "mammoth_north": [-14, 7],
#     "mammoth_south": [-14, 7],
#     "ridgecrest_north": [-7, 3.5],
#     "ridgecrest_south": [-14, 7],
# }
xlim_right = {
    "mammoth_north": 350,
    "mammoth_south": 350,
    "ridgecrest_north": 120,
    "ridgecrest_south": 350,
}

# %%
for i, folder in enumerate(folders):
    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 6), sharex=True, gridspec_kw={"hspace": 0.05})
    for j, picker in enumerate(pickers[::-1]):
        residual = pd.read_csv(f"{protocol}{bucket}/training/stats/{folder}/{picker}/time_residual.csv")
        # residual = pd.read_csv(f"results/stats/{folder}/{picker}/time_residual.csv")

        residual.loc[residual["pdiff_mean"].isna(), "pdiff_mean"] = residual.loc[
            residual["pdiff_mean"].isna(), "sdiff_mean"
        ]
        residual.loc[residual["sdiff_mean"].isna(), "sdiff_mean"] = residual.loc[
            residual["sdiff_mean"].isna(), "pdiff_mean"
        ]
        residual["mean"] = residual[["pdiff_mean", "sdiff_mean"]].mean(axis=1)
        residual = residual.loc[residual.groupby("event_id")["mean"].idxmin()]

        if folder == "ridgecrest_north":
            idx = (residual["pdiff_mean"] < -2.5) & (residual["sdiff_mean"] < -2.5)
            residual.loc[idx, "pdiff_mean"] = residual.loc[idx, "pdiff_mean"].apply(lambda x: x + 4.0)
            residual.loc[idx, "sdiff_mean"] = residual.loc[idx, "sdiff_mean"].apply(lambda x: x + 4.0)

        # index = (residual["pdiff_std"] < 10) & (residual["sdiff_std"] < 10)
        index = np.array([True] * len(residual))

        # fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))
        # ax[0, 0].scatter(residual["dist_km"][index], residual["pdiff_mean"][index], s=1, alpha=0.5, label="P")
        # ax[0, 0].scatter(residual["dist_km"][index], residual["sdiff_mean"][index], s=1, alpha=0.5, label="S")
        # ax[0, 0].set_ylim([-20, 20])

        # X = residual["dist_km"][index].values.reshape(-1, 1)
        # y1 = residual["pdiff_mean"][index].values.reshape(-1, 1)
        # y2 = residual["sdiff_mean"][index].values.reshape(-1, 1)
        # X_ = np.concatenate([X, X], axis=0)
        # y = np.concatenate([y1, y2], axis=0)
        # reg = RANSACRegressor(random_state=0).fit(X_, y)
        # dy1 = reg.predict(X).squeeze() * 0.0
        # dy2 = reg.predict(X).squeeze() * 0.0
        dy1, dy2 = 0, 0

        ax[j, 0].scatter(
            residual["dist_km"][index],
            residual["pdiff_mean"][index] - dy1,
            s=2 ** (residual["magnitude"][index].values),
            # s=3,
            alpha=0.2,
            # label="P",
            rasterized=True,
        )
        ax[j, 0].scatter(
            residual["dist_km"][index],
            residual["sdiff_mean"][index] - dy2,
            s=2 ** (residual["magnitude"][index].values),
            # s=3,
            alpha=0.2,
            # label="S",
            rasterized=True,
        )

        ax[j, 0].set_ylim(ylim[folder])
        ax[j, 0].set_xlim(right=xlim_right[folder])
        if j == 0:
            ax[j, 0].scatter([], [], s=20, c="C0", label=f"P")
            ax[j, 0].scatter([], [], s=20, c="C1", label=f"S")
            ax[j, 0].legend(loc="upper right")

        ax[j, 0].grid(linestyle="--", linewidth=0.5)
        ax[j, 0].text(
            0.02,
            0.05,
            f"({chr(ord('a') + j)}) {picker_name[picker]}",
            transform=ax[j, 0].transAxes,
            fontsize=12,
            verticalalignment="bottom",
        )

    fig.text(x=0.05, y=0.5, s="Time residual (s)", ha="center", va="center", rotation="vertical", fontsize=14)
    fig.text(x=0.5, y=0.05, s="Distance (km)", ha="center", va="center", fontsize=14)
    fig.savefig(figure_path / f"time_residual_{folder}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_path / f"time_residual_{folder}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# %%
for i, folder in enumerate(folders):
    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.05})
    for j, picker in enumerate(pickers):
        # events = pd.read_csv(f"results/stats/{folder}/{picker}/time_residual.csv")
        events = pd.read_csv(f"{protocol}{bucket}/training/stats/{folder}/{picker}/time_residual.csv")

        events.loc[events["pdiff_mean"].isna(), "pdiff_mean"] = events.loc[events["pdiff_mean"].isna(), "sdiff_mean"]
        events.loc[events["sdiff_mean"].isna(), "sdiff_mean"] = events.loc[events["sdiff_mean"].isna(), "pdiff_mean"]
        events["mean"] = events[["pdiff_mean", "sdiff_mean"]].mean(axis=1)
        events = events.loc[events.groupby("event_id")["mean"].idxmin()]

        events = events.loc[
            (events["pdiff_mean"] > ylim[folder][0])
            & (events["pdiff_mean"] < ylim[folder][1])
            & (events["sdiff_mean"] > ylim[folder][0])
            & (events["sdiff_mean"] < ylim[folder][1])
        ]

        das_info = pd.read_csv(f"{protocol}{bucket}/{folder}/das_info.csv")
        lat_0 = das_info["latitude"].mean()
        lon_0 = das_info["longitude"].mean()
        proj = pyproj.Proj(f"+proj=sterea +lon_0={lon_0} +lat_0={lat_0} +units=km")

        catalog = pd.read_csv(f"{protocol}{bucket}/{folder}/catalog_data.csv")
        catalog["x_km"], catalog["y_km"] = proj(catalog["longitude"].values, catalog["latitude"].values)
        catalog["z_km"] = catalog["depth_km"]
        catalog["dist_km"] = catalog.apply(lambda x: np.sqrt(x["x_km"] ** 2 + x["y_km"] ** 2 + x["z_km"] ** 2), axis=1)

        detected_catalog = pd.merge(events, catalog, on="event_id", how="left", suffixes=("_detected", ""))

        ax[j, 0].scatter(
            catalog["dist_km"],
            catalog["magnitude"],
            s=2 ** catalog["magnitude"],
            c="gray",
            alpha=0.2,
            rasterized=True,
        )
        ax[j, 0].scatter(
            detected_catalog["dist_km"],
            detected_catalog["magnitude"],
            s=2 ** detected_catalog["magnitude"],
            c="C3",
            alpha=0.5,
            rasterized=True,
        )

        ax[j, 0].set_xlim(right=xlim_right[folder])
        ax[j, 0].grid(linestyle="--", linewidth=0.5)
        ax[j, 0].text(
            0.02,
            0.95,
            f"({chr(ord('a') + j)}) {picker_name[picker]}",
            transform=ax[j, 0].transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        if j == 0:
            ax[j, 0].scatter([], [], s=20, c="gray", label=f"Dataset")
        ax[j, 0].scatter([], [], s=20, c="C3", label=f"Detected: {len(detected_catalog):.0f}")
        ax[j, 0].legend(loc="upper right")

    fig.text(x=0.05, y=0.5, s="Magnitude", ha="center", va="center", rotation="vertical", fontsize=14)
    fig.text(x=0.5, y=0.05, s="Distance (km)", ha="center", va="center", fontsize=14)
    fig.savefig(figure_path / f"mag_dist_{folder}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_path / f"mag_dist_{folder}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# %%
folder = "mammoth"
fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.05})
for j, picker in enumerate(pickers):
    # events = pd.read_csv(f"results/stats/{folder}/{picker}/time_residual.csv")
    events = pd.read_csv(f"{protocol}{bucket}/{folder}/gamma/{picker}/gamma_events.csv", parse_dates=["time"])
    # add timezone info
    events["time"] = events["time"].dt.tz_localize("UTC")

    das_info = pd.read_csv(f"{protocol}{bucket}/{folder}/das_info.csv")
    lat_0 = das_info["latitude"].mean()
    lon_0 = das_info["longitude"].mean()
    proj = pyproj.Proj(f"+proj=sterea +lon_0={lon_0} +lat_0={lat_0} +units=km")

    catalog = pd.read_csv(f"{protocol}{bucket}/{folder}/catalog_data.csv", parse_dates=["event_time"])
    catalog["x_km"], catalog["y_km"] = proj(catalog["longitude"].values, catalog["latitude"].values)
    catalog["z_km"] = catalog["depth_km"]
    catalog["dist_km"] = catalog.apply(lambda x: np.sqrt(x["x_km"] ** 2 + x["y_km"] ** 2 + x["z_km"] ** 2), axis=1)

    # get timestamp in seconds
    min_time = min(events.time.min(), catalog.event_time.min())
    events_time = (events.time - min_time).astype("timedelta64[s]").to_numpy()
    catalog_time = (catalog.event_time - min_time).astype("timedelta64[s]").to_numpy()
    # matrix = np.abs(events_time[:, None] - catalog_time[None, :])
    # matrix = events_time[:, None] - catalog_time[None, :]
    matrix = catalog_time[None, :] - events_time[:, None]
    matrix = (matrix > ylim["mammoth_south"][0]) & (matrix < ylim["mammoth_south"][1])
    recall = np.any(matrix, axis=0)

    # detected_catalog = pd.merge(events, catalog, on="event_id", how="left", suffixes=("_detected", ""))

    ax[j, 0].scatter(
        catalog["dist_km"],
        catalog["magnitude"],
        s=2 ** catalog["magnitude"],
        c="gray",
        alpha=0.2,
        rasterized=True,
    )
    ax[j, 0].scatter(
        catalog[recall]["dist_km"],
        catalog[recall]["magnitude"],
        s=2 ** catalog[recall]["magnitude"],
        c="C3",
        alpha=0.5,
        rasterized=True,
    )

    ax[j, 0].set_xlim(right=xlim_right["mammoth_south"])
    ax[j, 0].grid(linestyle="--", linewidth=0.5)
    ax[j, 0].text(
        0.02,
        0.95,
        f"({chr(ord('a') + j)}) {picker_name[picker]}",
        transform=ax[j, 0].transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    if j == 0:
        ax[j, 0].scatter([], [], s=20, c="gray", label=f"Dataset")
    ax[j, 0].scatter([], [], s=20, c="C3", label=f"Detected: {len(catalog[recall]):.0f}")
    ax[j, 0].legend(loc="upper right")


fig.text(x=0.05, y=0.5, s="Magnitude", ha="center", va="center", rotation="vertical", fontsize=14)
fig.text(x=0.5, y=0.05, s="Distance (km)", ha="center", va="center", fontsize=14)
fig.savefig(figure_path / f"mag_dist_{folder}.png", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / f"mag_dist_{folder}.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 6))
for i, folder in enumerate(folders):
    das_info = pd.read_csv(f"{protocol}{bucket}/{folder}/das_info.csv")
    lat_0 = das_info["latitude"].mean()
    lon_0 = das_info["longitude"].mean()
    proj = pyproj.Proj(f"+proj=sterea +lon_0={lon_0} +lat_0={lat_0} +units=km")

    catalog = pd.read_csv(f"{protocol}{bucket}/{folder}/catalog_data.csv")
    catalog["x_km"], catalog["y_km"] = proj(catalog["longitude"].values, catalog["latitude"].values)
    catalog["z_km"] = catalog["depth_km"]
    catalog["dist_km"] = catalog.apply(lambda x: np.sqrt(x["x_km"] ** 2 + x["y_km"] ** 2 + x["z_km"] ** 2), axis=1)

    bins = np.linspace(catalog["dist_km"].min(), xlim_right[folder], 21)
    hist_catalog = ax[i // 2, i % 2].hist(
        catalog["dist_km"],
        bins=bins,
        facecolor="gray",
        edgecolor="white",
        alpha=0.2,
        label="Dataset",
    )

    # manually add legend by adding text box
    patches = []

    for j, picker in enumerate(pickers):
        # events = pd.read_csv(f"results/stats/{folder}/{picker}/time_residual.csv")
        events = pd.read_csv(f"{protocol}{bucket}/training/stats/{folder}/{picker}/time_residual.csv")

        events.loc[events["pdiff_mean"].isna(), "pdiff_mean"] = events.loc[events["pdiff_mean"].isna(), "sdiff_mean"]
        events.loc[events["sdiff_mean"].isna(), "sdiff_mean"] = events.loc[events["sdiff_mean"].isna(), "pdiff_mean"]
        events["mean"] = events[["pdiff_mean", "sdiff_mean"]].mean(axis=1)
        events = events.loc[events.groupby("event_id")["mean"].idxmin()]

        events = events.loc[
            (events["pdiff_mean"] > ylim[folder][0])
            & (events["pdiff_mean"] < ylim[folder][1])
            & (events["sdiff_mean"] > ylim[folder][0])
            & (events["sdiff_mean"] < ylim[folder][1])
        ]

        detected_catalog = pd.merge(events, catalog, on="event_id", how="left", suffixes=("_detected", ""))

        hist_picker = ax[i // 2, i % 2].hist(
            detected_catalog["dist_km"],
            alpha=0.5,
            # facecolor="C3",
            facecolor=f"C{j}",
            edgecolor="white",
            bins=bins,
            label=picker_name[picker],
        )
        ax[i // 2, i % 2].set_yscale("log")

        detection_rate = hist_picker[0] / hist_catalog[0]

        # manually add legend by adding text box
        patch = mpatches.Patch(
            color=f"C{j}",
            label=f"{len(detected_catalog)}",
            alpha=0.5,
        )
        patches.append(patch)

    legend = ax[i // 2, i % 2].legend(handles=patches, loc="upper right")
    ax[i // 2, i % 2].add_artist(legend)

    ax[i // 2, i % 2].text(
        0.05,
        0.97,
        f"({chr(ord('a') + i)})",
        transform=ax[i // 2, i % 2].transAxes,
        fontsize=14,
        verticalalignment="top",
    )

legend = ax[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
fig.text(x=0.05, y=0.5, s="Number of events", ha="center", va="center", rotation="vertical", fontsize=14)
fig.text(x=0.5, y=0.05, s="Distance (km)", ha="center", va="center", fontsize=14)
fig.savefig(figure_path / f"detection.png", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / f"detection.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
folder = "mammoth"
das_info = pd.read_csv(f"{protocol}{bucket}/{folder}/das_info.csv")
lat_0 = das_info["latitude"].mean()
lon_0 = das_info["longitude"].mean()
proj = pyproj.Proj(f"+proj=sterea +lon_0={lon_0} +lat_0={lat_0} +units=km")

catalog = pd.read_csv(f"{protocol}{bucket}/{folder}/catalog_data.csv", parse_dates=["event_time"])
catalog["x_km"], catalog["y_km"] = proj(catalog["longitude"].values, catalog["latitude"].values)
catalog["z_km"] = catalog["depth_km"]
catalog["dist_km"] = catalog.apply(lambda x: np.sqrt(x["x_km"] ** 2 + x["y_km"] ** 2 + x["z_km"] ** 2), axis=1)
bins = np.linspace(catalog["dist_km"].min(), xlim_right["mammoth_south"], 31)

ax[0, 0].hist(
    catalog["dist_km"],
    alpha=0.5,
    # facecolor="C3",
    facecolor="gray",
    edgecolor="white",
    bins=bins,
    label="Dataset",
)

for j, picker in enumerate(pickers):
    # events = pd.read_csv(f"results/stats/{folder}/{picker}/time_residual.csv")
    events = pd.read_csv(f"{protocol}{bucket}/{folder}/gamma/{picker}/gamma_events.csv", parse_dates=["time"])
    # add timezone info
    events["time"] = events["time"].dt.tz_localize("UTC")

    # get timestamp in seconds
    min_time = min(events.time.min(), catalog.event_time.min())
    events_time = (events.time - min_time).astype("timedelta64[s]").to_numpy()
    catalog_time = (catalog.event_time - min_time).astype("timedelta64[s]").to_numpy()
    # matrix = np.abs(events_time[:, None] - catalog_time[None, :])
    # matrix = events_time[:, None] - catalog_time[None, :]
    matrix = catalog_time[None, :] - events_time[:, None]
    matrix = (matrix > ylim["mammoth_south"][0]) & (matrix < ylim["mammoth_south"][1])
    recall = np.any(matrix, axis=0)

    hist_picker = ax[0, 0].hist(
        catalog[recall]["dist_km"],
        alpha=0.5,
        # facecolor="C3",
        facecolor=f"C{j}",
        edgecolor="white",
        bins=bins,
        label=f"{picker_name[picker]}: {len(catalog[recall])}",
    )

ax[0, 0].set_yscale("log")
legend = ax[0, 0].legend(loc="upper right")
ax[0, 0].set_xlabel("Distance (km)")
ax[0, 0].set_ylabel("Number of events")
ax[0, 0].set_xlim(right=xlim_right["mammoth_south"])
fig.savefig(figure_path / f"detection_{folder}.png", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / f"detection_{folder}.pdf", dpi=300, bbox_inches="tight")
plt.show()


# %%
# fig, ax = plt.subplots(3, 3, squeeze=False, figsize=(15, 15))
# xlim = [-120.0, -117.5]
# ylim = [37.1, 38.7]
# zlim = [30, 0]
# catalog = pd.read_csv(f"{protocol}{bucket}/mammoth_north/catalog_data.csv")
# for i, picker in enumerate(pickers):
#     # events = pd.read_csv(f"results/gamma/{picker}/mammoth/gamma_events.csv")
#     events = pd.read_csv(f"{protocol}{bucket}/mammoth/gamma/{picker}/gamma_events.csv")

#     s = events["gamma_score"] / events["gamma_score"].max() * 2
#     ax[i, 0].scatter(catalog["longitude"], catalog["latitude"], s=1, c="gray", alpha=0.1, label=picker_name[picker])
#     ax[i, 0].scatter(events["longitude"], events["latitude"], s=s, c="red", alpha=0.2, label=picker_name[picker])

#     ax[i, 0].set_xlim(xlim)
#     ax[i, 0].set_ylim(ylim)

#     ax[i, 1].scatter(catalog["longitude"], catalog["depth_km"], s=1, c="gray", alpha=0.1, label=picker_name[picker])
#     ax[i, 1].scatter(events["longitude"], events["depth_km"], s=s * 3, c="red", alpha=0.2, label=picker_name[picker])
#     ax[i, 1].set_xlim(xlim)
#     ax[i, 1].set_ylim(zlim)

#     ax[i, 2].scatter(catalog["latitude"], catalog["depth_km"], s=1, c="gray", alpha=0.1, label=picker_name[picker])
#     ax[i, 2].scatter(events["latitude"], events["depth_km"], s=s * 3, c="red", alpha=0.2, label=picker_name[picker])
#     ax[i, 2].set_xlim(ylim)
#     ax[i, 2].set_ylim(zlim)
# %%
