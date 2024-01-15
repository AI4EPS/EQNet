# %%
import os
import warnings
from glob import glob
from pathlib import Path

import fsspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from sklearn.linear_model import RANSACRegressor
from tqdm.auto import tqdm

warnings.simplefilter("error")

plt.rcParams["axes.xmargin"] = 0


# %%
protocol = "gs://"
bucket = "das_mammoth"
fs = fsspec.filesystem(protocol.replace("://", ""))

pickers = ["phasenet", "phasenet_das", "phasenet_das_v1"]
pickers = pickers[::-1]
picker_name = {
    "phasenet": "PhaseNet",
    "phasenet_das": "PhaseNet-DAS v1",
    "phasenet_das_v1": "PhaseNet-DAS v2",
}

folder = ""

figure_path = Path("paper_figures")
if not figure_path.exists():
    figure_path.mkdir(parents=True)

# %%
begin_time = pd.to_datetime("2020-11-17T00:00:00+00:00")
end_time = pd.to_datetime("2020-11-25T00:00:00+00:00")
min_longitude = -119.9
max_longitude = -117.4
min_latitude = 36.7
max_latitude = 38.7
min_score = 0.5


# %%
bins = pd.date_range(begin_time, end_time, freq="6h")

catalog = pd.read_csv(f"{protocol}{bucket}{folder}/catalog_data.csv", parse_dates=["event_time"])
catalog["event_time"] = pd.to_datetime(catalog["event_time"], format="ISO8601")
catalog["time"] = catalog["event_time"]
catalog = catalog[
    (catalog["longitude"] >= min_longitude)
    & (catalog["longitude"] <= max_longitude)
    & (catalog["latitude"] >= min_latitude)
    & (catalog["latitude"] <= max_latitude)
]
catalog = catalog[(catalog["event_time"] >= begin_time) & (catalog["event_time"] <= end_time)]

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
for j, picker in enumerate(pickers):
    events = pd.read_csv(f"{protocol}{bucket}{folder}/gamma/{picker}/gamma_events.csv", parse_dates=["time"])
    events["time"] = pd.to_datetime(events["time"]).dt.tz_localize("UTC")
    events = events[
        (events["longitude"] >= min_longitude)
        & (events["longitude"] <= max_longitude)
        & (events["latitude"] >= min_latitude)
        & (events["latitude"] <= max_latitude)
    ]
    events = events[(events["time"] >= begin_time) & (events["time"] <= end_time)]
    ax[0, 0].hist(
        events["time"],
        bins=bins,
        alpha=0.5,
        facecolor=f"C{j}",
        # facecolor="C3",
        edgecolor="white",
        linewidth=0.5,
        label=f"{picker_name[picker]}: {len(events)}",
    )

ax[0, 0].hist(
    catalog["time"], bins=bins, alpha=0.6, facecolor="gray", edgecolor="white", label=f"Catalog: {len(catalog)}"
)

legend = ax[0, 0].legend(loc="upper right")
ax[0, 0].grid(linestyle="--", linewidth=0.2)
ax[0, 0].set_ylabel("Number of events")
ax[0, 0].set_xlim(left=begin_time, right=end_time)
# set titck label 2020-11-17, 11-18, 11-19...
ax[0, 0].set_xticks(bins[::4])
xlabel = list(bins[::4].strftime("%m/%d"))
xlabel[0] = bins[0].strftime("%Y/%m/%d")
ax[0, 0].set_xticklabels(xlabel)

## add text A in the top left corner
# ax[0, 0].text(
#     0.06,
#     0.95,
#     "(i)",
#     transform=ax[0, 0].transAxes,
#     fontsize=16,
#     fontweight="bold",
#     va="top",
#     ha="center",
# )

fig.savefig(figure_path / f"catalog_hist_continous.png", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / f"catalog_hist_continous.pdf", dpi=300, bbox_inches="tight")
plt.show()


# %%
fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.05})
for j, picker in enumerate(pickers):
    if not os.path.exists(f"{picker}_picks.csv"):
        picks = []
        # pick_list = fs.glob(f"{bucket}{folder}/{picker}/picks/*.csv")
        pick_list = fs.glob(f"{bucket}{folder}/gamma/{picker}/picks/*.csv")
        for pick_file in tqdm(pick_list):
            try:
                tmp = pd.read_csv(f"{protocol}{pick_file}")
                if "station_id" in tmp.columns:
                    tmp = tmp.rename(columns={"station_id": "channel_index"})
            except Exception as e:
                print(e)
                print(pick_file)

            picks.append(tmp)
        picks = pd.concat(picks, ignore_index=True)
        # picks = picks[picks["phase_score"] >= min_score]
        picks.to_csv(f"{picker}_picks.csv", index=False)
    else:
        picks = pd.read_csv(f"{picker}_picks.csv")
    # picks = picks[picks["phase_score"] >= min_score]
    if "event_index" in picks.columns:
        picks = picks[picks["event_index"] != -1]
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    print(f"{picker}: {len(picks)}")
    # color phase_type P as blue and S as red
    color = picks["phase_type"].apply(lambda x: "C0" if x == "P" else "C3")
    ax[j, 0].scatter(
        picks["phase_time"][::100],
        picks["channel_index"][::100],
        # c=picks["phase_score"][::100],
        c="C3",
        # c=color[::100],
        s=0.2,
        alpha=0.3,
        # cmap="Oranges",
        # cmap="OrRd",
        # vmin=min_score,
        # vmax=min_score + 0.1,
        marker=",",
        rasterized=True,
        label=f"{picker_name[picker]}: {len(picks)/1e6:.1f}M",
    )
    ax[j, 0].text(
        0.04,
        0.95,
        f"({chr(97+j)})",
        transform=ax[j, 0].transAxes,
        fontsize=16,
        va="top",
        ha="center",
    )
    print("done")
    legend = ax[j, 0].legend(loc="upper right")

    ax[j, 0].set_ylabel("Channel index")
    ax[j, 0].set_xlim(left=begin_time, right=end_time)

ax[2, 0].set_xticks(bins[::4])
xlabel = list(bins[::4].strftime("%m/%d"))
xlabel[0] = bins[0].strftime("%Y/%m/%d")
ax[2, 0].set_xticklabels(xlabel)
fig.savefig(figure_path / f"picks_continous.png", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / f"picks_continous.pdf", dpi=300, bbox_inches="tight")
plt.show()


# %%
min_longitude = -119.5
max_longitude = -118.4
min_latitude = 37.27
max_latitude = 38.12

bins = pd.date_range(begin_time, end_time, freq="6h")

catalog = pd.read_csv(f"{protocol}{bucket}{folder}/catalog_data.csv", parse_dates=["event_time"])
catalog["event_time"] = pd.to_datetime(catalog["event_time"], format="ISO8601")
catalog["time"] = catalog["event_time"]
catalog = catalog[
    (catalog["longitude"] >= min_longitude)
    & (catalog["longitude"] <= max_longitude)
    & (catalog["latitude"] >= min_latitude)
    & (catalog["latitude"] <= max_latitude)
]
catalog = catalog[(catalog["event_time"] >= begin_time) & (catalog["event_time"] <= end_time)]

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
for j, picker in enumerate(pickers):
    events = pd.read_csv(f"{protocol}{bucket}{folder}/gamma/{picker}/gamma_events.csv", parse_dates=["time"])
    events["time"] = pd.to_datetime(events["time"]).dt.tz_localize("UTC")
    events = events[
        (events["longitude"] >= min_longitude)
        & (events["longitude"] <= max_longitude)
        & (events["latitude"] >= min_latitude)
        & (events["latitude"] <= max_latitude)
    ]
    events = events[(events["time"] >= begin_time) & (events["time"] <= end_time)]
    ax[0, 0].hist(
        events["time"],
        bins=bins,
        alpha=0.5,
        facecolor=f"C{j}",
        # facecolor="C3",
        linewidth=0.5,
        edgecolor="white",
        label=f"{picker_name[picker]}: {len(events)}",
    )

ax[0, 0].hist(
    catalog["time"], bins=bins, alpha=0.6, facecolor="gray", edgecolor="white", label=f"Catalog: {len(catalog)}"
)

legend = ax[0, 0].legend(loc="upper right")
ax[0, 0].grid(linestyle="--", linewidth=0.2)
ax[0, 0].set_ylabel("Number of events")
ax[0, 0].set_xlim(left=begin_time, right=end_time)
# set titck label 2020-11-17, 11-18, 11-19...
ax[0, 0].set_xticks(bins[::4])
xlabel = list(bins[::4].strftime("%m/%d"))
xlabel[0] = bins[0].strftime("%Y/%m/%d")
ax[0, 0].set_xticklabels(xlabel)

ax[0, 0].text(
    0.06,
    0.95,
    "(i)",
    transform=ax[0, 0].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="center",
)

fig.savefig(figure_path / f"catalog_hist_continous2.png", dpi=300, bbox_inches="tight")
fig.savefig(figure_path / f"catalog_hist_continous2.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
