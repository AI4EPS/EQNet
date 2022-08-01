# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from gamma.utils import association
import multiprocessing
from multiprocessing import Manager
import sys
import warnings
warnings.filterwarnings("ignore")

# %%
dataset_name = sys.argv[1]
data_path = Path(f"/net/kuafu/mnt/tank/data/EventData/{dataset_name}/")
output_path = Path(f"/net/kuafu/mnt/tank/data/EventData/{dataset_name}/picks_phasenet_filtered/")
figures_path = Path(f"/net/kuafu/mnt/tank/data/EventData/{dataset_name}/figures_phasenet_filtered/")

# data_path = Path("/net/kuafu/mnt/tank/data/EventData/Mammoth_north/")
# output_path = Path("/net/kuafu/mnt/tank/data/EventData/Mammoth_north/picks_phasenet_filtered/")
# figures_path = Path("/net/kuafu/mnt/tank/data/EventData/Mammoth_north/figures_phasenet_filtered/")

if not output_path.exists():
    output_path.mkdir(parents=True)
if not figures_path.exists():
    figures_path.mkdir(parents=True)

# %%
stations = pd.read_csv(data_path.joinpath("das_info.csv"), index_col="index")

# %%
y0 = stations["latitude"].mean()
x0 = stations["longitude"].mean()
degree2km = 111.32
stations["x_km"] = (stations["longitude"] - x0) * degree2km
stations["y_km"] = (stations["latitude"] - y0) * degree2km

## Match data format for GaMMA 
stations["x(km)"] = stations["longitude"].apply(lambda x: (x - x0)*degree2km)
stations["y(km)"] = stations["latitude"].apply(lambda x: (x - y0)*degree2km)
stations["z(km)"] = 0

stations["id"] = stations.index
stations["id"] = stations["id"].apply(lambda x: str(x))

# %%
## Setting for GaMMA
config = {
    'center': (x0, y0), 
    'xlim_degree': [x0-2, x0+2], 
    'ylim_degree': [y0-2, y0+2], 
    'degree2km': degree2km}
config["dims"] = ['x(km)', 'y(km)', 'z(km)']
config["use_dbscan"] = False
config["use_amplitude"] = False
config["x(km)"] = (np.array(config["xlim_degree"])-np.array(config["center"][0]))*config["degree2km"]
config["y(km)"] = (np.array(config["ylim_degree"])-np.array(config["center"][1]))*config["degree2km"]
config["z(km)"] = (0, 20)
config["vel"] = {"p": 6.0, "s": 6.0 / 1.73}
#config["vel"] = {"p": 5.0, "s": 5.0 / 1.73} ## Mammoth
config["method"] = "BGMM"
if config["method"] == "BGMM":
    config["oversample_factor"] = 5
if config["method"] == "GMM":
    config["oversample_factor"] = 1
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # x
    (0, None),  # t
)
config["initial_mode"] = "one_point"

# Filtering
config["min_picks_per_eq"] = 100 #len(stations)//10
config["min_p_picks_per_eq"] = 50 #len(stations)//20
config["min_s_picks_per_eq"] = 50 #len(stations)//20
config["max_sigma11"] = 0.3

for k, v in config.items():
    print(f"{k}: {v}")

# %%
def associate(picks, stations, config):

    ## match format from PhaseNet to PhaseNet-DAS
    picks["phase_type"] = picks["phase_type"].apply(lambda x: x.upper())
    picks["channel_index"] = picks["station_id"]
    picks["phase_score"] = picks["phase_prob"]
    picks = picks.join(stations, on="channel_index", how="inner")

    ## match data format for GaMMA 
    picks["id"] = picks["channel_index"].apply(lambda x: str(x))
    picks["id"] = picks["id"].apply(lambda x: str(x))

    picks["type"] = picks["phase_type"]
    picks["prob"] = picks["phase_prob"]
    picks["timestamp"] = picks["phase_time"].apply(lambda x: datetime.fromisoformat(x))

    event_idx0 = 0 ## current earthquake index
    catalogs, assignments = association(picks, stations, config, event_idx0, config["method"])

    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    picks = picks.join(assignments.set_index("pick_idx")).fillna(-1).astype({'event_idx': int})

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
        for j in range(i+1, num_events):
            if (np.abs(mean_time[i, 0] - mean_time[j, 0]) < tol) or (np.abs(mean_time[i, 1] - mean_time[j, 1]) < tol):                
                if events_.loc[event_index[i]]["prob_gamma"]/events_.loc[event_index[i]]["sigma_time"] > events_.loc[event_index[j]]["prob_gamma"]/events_.loc[event_index[j]]["sigma_time"]:
                    if event_index[j] in index_selected:
                        index_selected.remove(event_index[j])
                else:
                    if event_index[i] in index_selected:
                        index_selected.remove(event_index[i])
    return picks[picks["event_index"].isin(index_selected)]

# %%
# event_list = []
# files = data_path.rglob('picks_phasenet_raw/*.csv')
# if True:

def run(files, event_list):
    for file in files:
        picks = pd.read_csv(file)
        picks = picks[picks["phase_index"] > 10]
        picks = picks[picks["phase_prob"] > 0.5]
        
        events, picks = associate(picks, stations, config)
        for e in events:
            e["event_id"] = file.stem
            event_list.append(e)

        picks_ = picks.copy()
        picks["event_index"] = picks["event_idx"]
        picks = picks[picks["event_idx"] != -1]

        if len(picks) == 0:
            continue 
        ## filter: keep both P/S picks
        picks["event_index"] = picks["event_idx"]
        # for i, phase_type in enumerate(picks["phase_type"].unique()):
        for i, phase_type in enumerate(["P", "S"]):
            if i == 0:
                filt_picks = picks[(picks["event_index"] != -1) & (picks["phase_type"] == phase_type)][["station_id", "event_index"]]
            else:
                filt_picks = filt_picks.merge(picks[(picks["event_index"] != -1) & (picks["phase_type"] == phase_type)][["station_id", "event_index"]], how="inner", on=["station_id", "event_index"])
        picks = picks.merge(filt_picks, on=["station_id", "event_index"], how='right')
        if len(picks) == 0:
            continue
        ## filter: 
        for i, k in enumerate(picks["event_index"].unique()):
            if len(picks[(picks["event_index"] == k) & (picks["phase_type"] == "P")]) < config["min_p_picks_per_eq"]:
                picks = picks[picks["event_index"] != k]
            if len(picks[(picks["event_index"] == k) & (picks["phase_type"] == "S")]) < config["min_s_picks_per_eq"]:
                picks = picks[picks["event_index"] != k]
            if len(picks[(picks["event_index"] == k)]) < config["min_picks_per_eq"]:
                picks = picks[picks["event_index"] != k]
        ## filter 
        if len(picks) == 0:
            continue 
        picks = filter_close_false_picks(picks, events)

        if len(picks) == 0:
            continue 
        picks.sort_values(by=["channel_index", "phase_index"], inplace=True)
        picks.to_csv(output_path.joinpath(file.name), index=False, columns=["channel_index", "phase_index", "phase_time", "phase_score", "phase_type", "event_index"], float_format='%.3f')
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].scatter(picks_["channel_index"][picks_["phase_type"] == "P"], picks_["phase_index"][picks_["phase_type"] == "P"], c=picks_["event_idx"][picks_["phase_type"] == "P"].apply(lambda x: f"C{x}" if x != -1 else "k"), s=1, marker=".")
        axs[0].scatter(picks_["channel_index"][picks_["phase_type"] == "S"], picks_["phase_index"][picks_["phase_type"] == "S"], c=picks_["event_idx"][picks_["phase_type"] == "S"].apply(lambda x: f"C{x}" if x != -1 else "k"), s=1, marker="x")
        axs[0].invert_yaxis()

        picks_ = picks.copy()
        axs[1].scatter(picks_["channel_index"][picks_["phase_type"] == "P"], picks_["phase_index"][picks_["phase_type"] == "P"], c=picks_["event_idx"][picks_["phase_type"] == "P"].apply(lambda x: f"C{x}" if x != -1 else "gray"), s=1, marker=".")
        axs[1].scatter(picks_["channel_index"][picks_["phase_type"] == "S"], picks_["phase_index"][picks_["phase_type"] == "S"], c=picks_["event_idx"][picks_["phase_type"] == "S"].apply(lambda x: f"C{x}" if x != -1 else "gray"), s=1, marker="x")
        axs[1].invert_yaxis()
        plt.savefig(figures_path.joinpath(file.stem + ".jpg"), bbox_inches='tight')
        plt.close(fig)

        # plt.savefig("test.jpg")
        # plt.show()     
        # raise

# %%
if __name__ == "__main__":
    manager = Manager()
    event_list = manager.list()
    files = list(data_path.rglob('picks_phasenet_raw/*.csv'))

    jobs = []
    num_cores = multiprocessing.cpu_count()
    # num_cores = 2
    for i in range(num_cores):
        p = multiprocessing.Process(target=run, args=(files[i::num_cores], event_list))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    events = pd.DataFrame(list(event_list))
    events["event_index"] = events["event_idx"]
    events["longitude"] = events["x(km)"].apply(
        lambda x: x / config["degree2km"] + config["center"][0]
    )
    events["latitude"] = events["y(km)"].apply(
        lambda x: x / config["degree2km"] + config["center"][1]
    )
    events["depth_km"] = events["z(km)"]
    events.to_csv(data_path.joinpath("catalog_gamma.csv"), index=False, columns=["event_id", "time", "longitude", "latitude","depth_km","event_index"], float_format="%.6f")

# %%


# %%
print(output_path.joinpath("events.csv"))

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



