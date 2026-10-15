# %%
import os
from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import fsspec
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
import seaborn as sns
from matplotlib.colors import LightSource
from tqdm import tqdm

# %%
protocol = "gs://"
bucket = "quakeflow_das"
fs = fsspec.filesystem(protocol.replace("://", ""))

picker_name = {
    "phasenet": "PhaseNet",
    "phasenet_das": "PhaseNet-DAS v1",
    "phasenet_das_v1": "PhaseNet-DAS v2",
}

figure_path = Path("paper_figures")
if not figure_path.exists():
    figure_path.mkdir()


# %%
das_mammoth_north = pd.read_csv(f"{protocol}{bucket}/mammoth_north/das_info.csv")
das_mammoth_south = pd.read_csv(f"{protocol}{bucket}/mammoth_south/das_info.csv")
catalog_mammoth_north = pd.read_csv(f"{protocol}{bucket}/mammoth_north/catalog_data.csv", index_col="event_id")
catalog_mammoth_south = pd.read_csv(f"{protocol}{bucket}/mammoth_south/catalog_data.csv", index_col="event_id")

das_ridgecrest_north = pd.read_csv(f"{protocol}{bucket}/ridgecrest_north/das_info.csv")
das_ridgecrest_south = pd.read_csv(f"{protocol}{bucket}/ridgecrest_south/das_info.csv")
catalog_ridgecrest_north = pd.read_csv(f"{protocol}{bucket}/ridgecrest_north/catalog_data.csv", index_col="event_id")
catalog_ridgecrest_south = pd.read_csv(f"{protocol}{bucket}/ridgecrest_south/catalog_data.csv", index_col="event_id")


# %%
def add_scalebar(ax, km=50, linewidth=2):
    llx0, llx1 = ax.get_xlim()
    lly0, lly1 = ax.get_ylim()
    lat_0 = (lly1 + lly0) / 2
    ll2km = np.cos(np.deg2rad(lat_0)) * 111.32

    bar_x = [llx1 - 0.1 - km / ll2km, llx1 - 0.1]
    bar_y = [lly0 + 0.1, lly0 + 0.1]
    ax.plot(bar_x, bar_y, color="k", linewidth=linewidth)
    ax.plot(
        [bar_x[0] - 0.006, bar_x[0] - 0.006], [bar_y[0] - 0.025, bar_y[0] + 0.025], color="k", linewidth=linewidth - 1
    )
    ax.plot(
        [bar_x[1] + 0.005, bar_x[1] + 0.005], [bar_y[1] - 0.025, bar_y[1] + 0.025], color="k", linewidth=linewidth - 1
    )
    ax.text(
        np.mean(bar_x),
        np.mean(bar_y) + 0.01,
        f"{km} km",
        color="k",
        fontsize=12,
        ha="center",
        va="bottom",
    )


# %%
fig, axes = plt.subplots(1, 2, squeeze=False, figsize=(10, 6))

ls = LightSource(azdeg=0, altdeg=45)

min_longitude = -119.9
max_longitude = -117.4
min_latitude = 36.7
max_latitude = 38.7
region = [min_longitude, max_longitude, min_latitude, max_latitude]
topo = pygmt.datasets.load_earth_relief(resolution="15s", region=region).to_numpy() / 1e3  # km
x = np.linspace(min_longitude, max_longitude, topo.shape[1])
y = np.linspace(min_latitude, max_latitude, topo.shape[0])
dx, dy = 1, 1
xgrid, ygrid = np.meshgrid(x, y)
axes[0, 0].pcolormesh(
    xgrid,
    ygrid,
    ls.hillshade(topo, vert_exag=10, dx=dx, dy=dy),
    vmin=-1,
    shading="gouraud",
    cmap="gray",
    alpha=1.0,
    antialiased=True,
    rasterized=True,
)
axes[0, 0].scatter(
    catalog_mammoth_south["longitude"], catalog_mammoth_south["latitude"], s=3, c="k", linewidth=0, rasterized=True
)
axes[0, 0].scatter(
    das_mammoth_north["longitude"], das_mammoth_north["latitude"], s=1, label="Mammoth north", rasterized=True
)
axes[0, 0].scatter(
    das_mammoth_south["longitude"], das_mammoth_south["latitude"], s=1, label="Mammoth south", rasterized=True
)
# axes[0].axis("scaled")
axes[0, 0].set_aspect(1.0 / np.cos(np.deg2rad((min_latitude + max_latitude) / 2)))
axes[0, 0].set_xlim([min_longitude, max_longitude])
axes[0, 0].set_ylim([min_latitude, max_latitude])
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")
axes[0, 0].tick_params(axis="x", labelrotation=15)
axes[0, 0].legend(markerscale=10, loc="lower left")
# add (a) to top upper left corner in a nice way
# add background rectangle
axes[0, 0].text(
    min_longitude + 0.07,
    max_latitude - 0.06,
    "(a)",
    fontsize=13,
    horizontalalignment="left",
    verticalalignment="top",
    bbox=dict(boxstyle='round', facecolor='white', edgecolor="none", alpha=0.7),
)
add_scalebar(axes[0, 0])

min_longitude = -119
max_longitude = -116.5
min_latitude = 34.5
max_latitude = 36.5
region = [min_longitude, max_longitude, min_latitude, max_latitude]
topo = pygmt.datasets.load_earth_relief(resolution="15s", region=region).to_numpy() / 1e3  # km
x = np.linspace(min_longitude, max_longitude, topo.shape[1])
y = np.linspace(min_latitude, max_latitude, topo.shape[0])
dx, dy = 1, 1
xgrid, ygrid = np.meshgrid(x, y)
axes[0, 1].pcolormesh(
    xgrid,
    ygrid,
    ls.hillshade(topo, vert_exag=10, dx=dx, dy=dy),
    vmin=-1,
    shading="gouraud",
    cmap="gray",
    alpha=1.0,
    antialiased=True,
    rasterized=True,
)
axes[0, 1].scatter(
    catalog_ridgecrest_north["longitude"],
    catalog_ridgecrest_north["latitude"],
    s=3,
    c="k",
    linewidth=0,
    rasterized=True,
)
axes[0, 1].scatter(
    catalog_ridgecrest_south["longitude"],
    catalog_ridgecrest_south["latitude"],
    s=1,
    c="k",
    linewidth=0,
    rasterized=True,
)
axes[0, 1].scatter(
    das_ridgecrest_north["longitude"], das_ridgecrest_north["latitude"], s=1, label="Ridgecrest north", rasterized=True
)
axes[0, 1].scatter(
    das_ridgecrest_south["longitude"], das_ridgecrest_south["latitude"], s=1, label="Ridgecrest south", rasterized=True
)
# axes[1].axis("scaled")
axes[0, 1].set_aspect(1.0 / np.cos(np.deg2rad((min_latitude + max_latitude) / 2)))
axes[0, 1].set_xlim([min_longitude, max_longitude])
axes[0, 1].set_ylim([min_latitude, max_latitude])
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")
axes[0, 1].tick_params(axis="x", labelrotation=15)
axes[0, 1].legend(markerscale=10, loc="lower left")
axes[0, 1].text(
    min_longitude + 0.07,
    max_latitude - 0.06,
    "(b)",
    fontsize=13,
    horizontalalignment="left",
    verticalalignment="top",
    bbox=dict(boxstyle='round', facecolor='white', edgecolor="none", alpha=0.7),
)
add_scalebar(axes[0, 1])

fig.tight_layout()
plt.savefig(figure_path / "das_location.png", dpi=300, bbox_inches="tight")
plt.savefig(figure_path / "das_location.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
min_longitude = -120
max_longitude = -117.5
min_latitude = 37.1
max_latitude = 38.7
min_depth = -0.1
max_depth = 21
cmap = "viridis"

xlim = [-120.0, -117.5]
ylim = [37.1, 38.7]
zlim = [30, 0]
catalog = pd.read_csv(f"{protocol}{bucket}/mammoth_north/catalog_data.csv")
pickers = ["phasenet", "phasenet_das", "phasenet_das_v1"]

for i, picker in enumerate(pickers):
    events = pd.read_csv(f"{protocol}{bucket}/mammoth/gamma/{picker}/gamma_events.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            10,
            10
            * (max_latitude - min_latitude)
            / ((max_longitude - min_longitude) * np.cos(np.deg2rad((min_latitude + max_latitude) / 2))),
        ),
        gridspec_kw={"width_ratios": [4, 1], "height_ratios": [4, 1], "wspace": 0.05, "hspace": 0.05},
        # sharex=True,
        # sharey=True,
    )

    region = [min_longitude, max_longitude, min_latitude, max_latitude]
    topo = pygmt.datasets.load_earth_relief(resolution="15s", region=region).to_numpy() / 1e3  # km
    x = np.linspace(min_longitude, max_longitude, topo.shape[1])
    y = np.linspace(min_latitude, max_latitude, topo.shape[0])
    dx, dy = 1, 1
    xgrid, ygrid = np.meshgrid(x, y)
    axes[0, 0].pcolormesh(
        xgrid,
        ygrid,
        ls.hillshade(topo, vert_exag=10, dx=dx, dy=dy),
        vmin=-1,
        shading="gouraud",
        cmap="gray",
        alpha=1.0,
        antialiased=True,
        rasterized=True,
    )

    axes[0, 0].scatter(
        catalog["longitude"],
        catalog["latitude"],
        s=1,
        c="k",
        alpha=0.5,
        # label="Catalog",
        rasterized=True,
    )

    s = 3 * events["gamma_score"] / events["gamma_score"].max()
    axes[0, 0].scatter(
        events["longitude"],
        events["latitude"],
        s=s,
        c="r",
        alpha=0.5,
        # label="PhaseNet-DAS",
        rasterized=True,
    )
    axes[0, 0].autoscale(tight=True)
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
    axes[0, 0].scatter([], [], s=3, c="k", label="Catalog")
    axes[0, 0].scatter([], [], s=3, c="r", label=f"{picker_name[picker]}")
    axes[0, 0].scatter([], [], s=3, c="C0", label="DAS cable")
    axes[0, 0].legend(markerscale=5, loc="center left")

    axes[0, 0].set_aspect(1.0 / np.cos(np.deg2rad((min_latitude + max_latitude) / 2)))
    axes[0, 0].set_xlim(xlim)
    axes[0, 0].set_ylim(ylim)
    axes[0, 0].set_ylabel("Latitude")
    axes[0, 0].xaxis.set_label_position("top")
    axes[0, 0].xaxis.tick_top()
    axes[0, 0].set_xlabel("Longitude")
    add_scalebar(axes[0, 0])

    axes[0, 1].scatter(
        catalog["depth_km"],
        catalog["latitude"],
        s=1,
        c="k",
        alpha=0.2,
        label="Catalog",
        rasterized=True,
    )
    axes[0, 1].scatter(events["depth_km"], events["latitude"], s=s, c="r", alpha=0.5, rasterized=True)
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

    axes[1, 0].scatter(
        catalog["longitude"],
        catalog["depth_km"],
        s=1,
        c="k",
        alpha=0.2,
        label="Catalog",
        rasterized=True,
    )
    axes[1, 0].scatter(events["longitude"], events["depth_km"], s=s, c="r", alpha=0.5, rasterized=True)
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
    fig.savefig(figure_path / f"catalog_mammoth_{picker}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(figure_path / f"catalog_mammoth_{picker}.png", dpi=300, bbox_inches="tight")

# %%
min_longitude = -120
max_longitude = -117.5
min_latitude = 37.1
max_latitude = 38.7
min_depth = -0.1
max_depth = 21
begin_time = pd.to_datetime("2020-11-17T00:00:00+00:00")
end_time = pd.to_datetime("2020-11-25T00:00:00+00:00")
cmap = "viridis"

xlim = [-120.0, -117.5]
ylim = [37.1, 38.7]
zlim = [30, 0]
catalog = pd.read_csv(f"{protocol}/das_mammoth/catalog_data.csv")
catalog["time"] = pd.to_datetime(catalog["event_time"], format="ISO8601")
catalog = catalog[(catalog["time"] >= begin_time) & (catalog["time"] <= end_time)]
catalog = catalog[
    (catalog["latitude"] >= min_latitude)
    & (catalog["latitude"] <= max_latitude)
    & (catalog["longitude"] >= min_longitude)
    & (catalog["longitude"] <= max_longitude)
]
pickers = ["phasenet", "phasenet_das", "phasenet_das_v1"]

fig, axes = plt.subplots(
    1,
    1,
    squeeze=False,
    figsize=(
        8,
        8
        * (max_latitude - min_latitude)
        / ((max_longitude - min_longitude) * np.cos(np.deg2rad((min_latitude + max_latitude) / 2))),
    ),
    # gridspec_kw={"width_ratios": [4, 1], "height_ratios": [4, 1], "wspace": 0.05, "hspace": 0.05},
    # sharex=True,
    # sharey=True,
)

region = [min_longitude, max_longitude, min_latitude, max_latitude]
topo = pygmt.datasets.load_earth_relief(resolution="15s", region=region).to_numpy() / 1e3  # km
x = np.linspace(min_longitude, max_longitude, topo.shape[1])
y = np.linspace(min_latitude, max_latitude, topo.shape[0])
dx, dy = 1, 1
xgrid, ygrid = np.meshgrid(x, y)
axes[0, 0].pcolormesh(
    xgrid,
    ygrid,
    ls.hillshade(topo, vert_exag=10, dx=dx, dy=dy),
    vmin=-1,
    shading="gouraud",
    cmap="gray",
    alpha=1.0,
    antialiased=True,
    rasterized=True,
)

axes[0, 0].scatter(
    das_mammoth_north["longitude"],
    das_mammoth_north["latitude"],
    s=0.5,
    c="C0",
    marker=".",
    alpha=0.5,
    rasterized=True,
)

axes[0, 0].scatter(
    das_mammoth_south["longitude"],
    das_mammoth_south["latitude"],
    s=0.5,
    c="C0",
    marker=".",
    alpha=0.5,
    rasterized=True,
    # label="DAS cable",
)

axes[0, 0].scatter(
    catalog["longitude"],
    catalog["latitude"],
    s=1 * 20,
    c="k",
    alpha=0.5,
    # label="Catalog",
    rasterized=True,
)


for i, picker in enumerate(pickers):
    events = pd.read_csv(f"{protocol}/das_mammoth/gamma/{picker}/gamma_events.csv")
    events["time"] = pd.to_datetime(events["time"], format="ISO8601")
    events["time"] = events["time"].dt.tz_localize(timezone.utc)
    events = events[(events["time"] >= begin_time) & (events["time"] <= end_time)]
    events = events[
        (events["latitude"] >= min_latitude)
        & (events["latitude"] <= max_latitude)
        & (events["longitude"] >= min_longitude)
        & (events["longitude"] <= max_longitude)
    ]

    s = 3 * events["gamma_score"] / events["gamma_score"].max() * 10

    axes[0, 0].scatter(
        events["longitude"],
        events["latitude"],
        s=s,
        c=f"C{i+1}",
        alpha=0.5,
        # label="PhaseNet-DAS",
        # label=f"{picker_name[picker]}",
        rasterized=True,
    )


axes[0, 0].autoscale(tight=True)

axes[0, 0].scatter([], [], s=3, c="k", label="Catalog")
for i, picker in enumerate(pickers):
    axes[0, 0].scatter([], [], s=3, c=f"C{i+1}", label=f"{picker_name[picker]}")
# axes[0, 0].scatter([], [], s=3, c="r", label=f"{picker_name[picker]}")
axes[0, 0].scatter([], [], s=3, c="C0", label="DAS cable")
# move legend a bit higher
axes[0, 0].legend(markerscale=5, loc="center left", bbox_to_anchor=(0.0, 0.6))


axes[0, 0].set_aspect(1.0 / np.cos(np.deg2rad((min_latitude + max_latitude) / 2)))
axes[0, 0].set_xlim(xlim)
axes[0, 0].set_ylim(ylim)
axes[0, 0].set_ylabel("Latitude")
# axes[0, 0].xaxis.set_label_position("top")
# axes[0, 0].xaxis.tick_top()
axes[0, 0].set_xlabel("Longitude")

# axes[0, 0].text(
#     min_longitude + 0.03,
#     min_latitude + 0.01,
#     "(i)",
#     fontsize=16,
#     fontweight="bold",
#     horizontalalignment="left",
#     verticalalignment="bottom",
# )

# draw a box using
min_longitude = -119.5
max_longitude = -118.4
min_latitude = 37.27
max_latitude = 38.12
axes[0, 0].plot(
    [min_longitude, max_longitude, max_longitude, min_longitude, min_longitude],
    [min_latitude, min_latitude, max_latitude, max_latitude, min_latitude],
    c="k",
    lw=1.0,
    ls="--",
)
# add text label A in the lower left corner
axes[0, 0].text(
    min_longitude + 0.03,
    min_latitude + 0.01,
    "(i)",
    fontsize=16,
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="bottom",
)
add_scalebar(axes[0, 0])

fig.savefig(figure_path / f"catalog_mammoth_continous.pdf", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / f"catalog_mammoth_continous.png", dpi=300, bbox_inches="tight")

# %%
