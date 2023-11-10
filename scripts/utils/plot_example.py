# %%
import os
from pathlib import Path

import fsspec
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
from tqdm.auto import tqdm

# sns.set_context("paper")
matplotlib.use("Agg")


# %%
protocol = "gs"
bucket = "quakeflow_das"
folder = "mammoth_north"
fs = fsspec.filesystem(protocol)

# %%

# selected_event_ids = [
#     "nc73473316",
#     "nc73473331",
#     "nc73473381",
#     "nc73474001",
#     "nc71114014",
#     "nc71118574",
#     "nc71118679",
#     "nc71121589",
#     "nc71121689",
#     "nc71123139",
#     "nc71123184",
#     "nc73475186",
#     "nc73484126",
#     "nc73484266",
#     "nc73484351",
#     "nc73484856",
#     "nc73484861",
# ]

# event_ids = fs.glob(f"{protocol}://{bucket}/{folder}/data/*.h5")
# event_ids = [os.path.basename(event_id).split(".")[0] for event_id in event_ids]
# event_ids = np.unique(event_ids)
# selected_event_ids = event_ids

selected_event_ids = [
    # "nc73473316",
    "nc73473331",  # 2
    # "nc73473381",
    # "nc73474001",
    # "nc73526616",
    "nc73503520",  # 1
    # "nc73514620",
    # "nc73512635",
    # "nc73527626",
    "nc73527496",  # 3
    "nc73512120",  # 4
]

figure_path = "paper_figures/examples"
figure_path = Path(figure_path)
if not figure_path.exists():
    figure_path.mkdir(parents=True)

# %%

sampling_rate = 100
dt = 1.0 / sampling_rate
pbar = tqdm(enumerate(selected_event_ids), total=len(selected_event_ids))
for i, event_id in pbar:
    pbar.set_description(f"Plotting {event_id}")
    raw_picks = pd.read_csv(f"{protocol}://{bucket}/{folder}/phasenet/picks/{event_id}.csv")
    raw_picks["channel_index"] = raw_picks["station_id"]
    raw_picks["color"] = raw_picks["phase_type"].apply(lambda x: "r" if x == "P" else "b")

    try:
        # das_picks = pd.read_csv(f"/net/kuafu/mnt/tank/data/EventData/Mammoth_north/picks_phasenet_das/{event_id}.csv")
        das_picks = pd.read_csv(f"{protocol}://{bucket}/{folder}/phasenet_das_v1/picks/{event_id}.csv")
        das_picks = das_picks[das_picks["phase_score"] > 0.8]
        if len(das_picks[das_picks["phase_type"] == "P"]) < 500:
            continue
        if len(das_picks[das_picks["phase_type"] == "S"]) < 500:
            continue
    except:
        continue
    das_picks["color"] = das_picks["phase_type"].apply(lambda x: "r" if x == "P" else "b")

    t0 = 20
    tn = 50
    with fs.open(f"{protocol}://{bucket}/{folder}/data/{event_id}.h5", "rb") as f:
        with h5py.File(f, "r") as f:
            waveform = f["data"][:, t0 * sampling_rate : tn * sampling_rate]
    nx, nt = waveform.shape

    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    # fig, axes = plt.subplots(3, 1, figsize=(4, 5), sharex=True)

    waveform -= np.median(waveform, axis=0, keepdims=True)
    waveform -= np.median(waveform, axis=1, keepdims=True)
    waveform /= np.std(waveform, axis=1, keepdims=True)
    vmax = np.std(waveform)
    axes[0].imshow(
        waveform.T, vmax=vmax * 0.6, vmin=-vmax * 0.6, aspect="auto", cmap="seismic", extent=([0, nx, tn - t0, 0])
    )
    axes[0].set_ylim([0, tn - t0])
    axes[0].invert_yaxis()
    axes[0].grid(False)
    axes[0].set_ylabel("Time (s)")
    axes[0].text(0.02, 0.95, "(i)", transform=axes[0].transAxes, fontsize=16, va="top")

    sc = axes[1].scatter(
        raw_picks["channel_index"],
        raw_picks["phase_index"] * dt - t0,
        s=1,
        c=raw_picks["color"],
        linewidth=1,
        rasterized=True,
    )
    axes[1].set_xlim([0, nx])
    axes[1].set_ylim([0, tn - t0])
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Time (s)")
    axes[1].text(0.02, 0.95, "(ii)", transform=axes[1].transAxes, fontsize=16, va="top")

    axes[1].scatter([-1], [-1], s=14, c="r", label="P picks")  # , alpha=1.0)
    axes[1].scatter([-1], [-1], s=14, c="b", label="S picks")  # , alpha=1.0)
    axes[1].legend(loc="upper right", fontsize=14, title="PhaseNet", title_fontsize=12, alignment="left")

    axes[2].scatter(
        das_picks["channel_index"],
        das_picks["phase_index"] * dt - t0,
        s=1,
        c=das_picks["color"],
        linewidth=1,
        rasterized=True,
    )
    axes[2].set_xlim([0, nx])
    axes[2].set_ylim([0, tn - t0])
    axes[2].invert_yaxis()
    axes[2].set_ylabel("Time (s)")
    axes[2].set_xlabel("Channel index")
    axes[2].text(0.02, 0.95, "(iii)", transform=axes[2].transAxes, fontsize=16, va="top")

    axes[2].scatter([-1], [-1], s=14, c="r", label="P picks")  # , alpha=1.0)
    axes[2].scatter([-1], [-1], s=14, c="b", label="S picks")  # , alpha=1.0)
    axes[2].legend(loc="upper right", fontsize=14, title="PhaseNet-DAS", title_fontsize=12, alignment="left")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    try:
        plt.savefig(os.path.join(figure_path, f"examples/{event_id}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(figure_path, f"examples/{event_id}.svg"), dpi=300, bbox_inches="tight")
    except:
        os.mkdir(os.path.join(figure_path, f"examples"))
        plt.savefig(os.path.join(figure_path, f"examples/{event_id}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(figure_path, f"examples/{event_id}.svg"), dpi=300, bbox_inches="tight")

    # plt.show()
    # if i > 100:
    #     break
    # break

# %%
