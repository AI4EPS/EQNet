# %%
import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict

hdf5_file = Path("/net/arius/scratch/zhuwq/EQNet_update2/datasets/NCEDC/ncedc_event_dataset_3c.h5")
picks_path = Path("../../results_ncedc_event_dataset_3c/picks_phasenet_raw/")

steps = [1, 2]

if 1 in steps:
    # %%
    # picks_path = Path("../../results_ncedc_event_dataset_3c_0.1/picks_phasenet_raw/")
    # pick_file = Path("../../results_ncedc_event_dataset_3c/picks_phasenet_raw.csv")

    # %%
    picks = sorted(list(picks_path.glob("*.csv")))
    print(f"{len(picks) = }")

    # %%
    events_empty = defaultdict(list)
    events_missed = defaultdict(list)
    events_double = defaultdict(list)

    # %%
    with h5py.File(hdf5_file, "r") as f:
        for pick in tqdm(picks):
            try:
                pred_dict = pd.read_csv(pick).to_dict(orient="list")
            except:
                events_empty[pick.stem.split("_")[0]].append(pick.name)
                continue

            label_dict = dict(f[pick.stem.replace("_", "/")].attrs)
            label_p_index = [x for (x, y) in zip(label_dict["phase_index"], label_dict["phase_type"]) if y == "P"]
            label_s_index = [x for (x, y) in zip(label_dict["phase_index"], label_dict["phase_type"]) if y == "S"]
            pred_p_index = [x for (x, y) in zip(pred_dict["phase_index"], pred_dict["phase_type"]) if y == "P"]
            pred_s_index = [x for (x, y) in zip(pred_dict["phase_index"], pred_dict["phase_type"]) if y == "S"]

            double_p_index = []
            recall_p_index = []
            for x0 in label_p_index:
                for x1 in pred_p_index:
                    if abs(x1 - x0) > 100:
                        double_p_index.append(x1)
                    else:
                        recall_p_index.append(x1)

            double_s_index = []
            recall_s_index = []
            for x0 in label_s_index:
                for x1 in pred_s_index:
                    if abs(x1 - x0) > 100:
                        double_s_index.append(x1)
                    else:
                        recall_s_index.append(x1)

            # if (len(pred_p_index) > 1) or (len(pred_s_index) > 1):
            #     events_double[pick.stem.split("_")[0]].append(pick.name)
            if (len(double_p_index) > 0) or (len(double_s_index) > 0):
                events_double[pick.stem.split("_")[0]].append(pick.name)

                # print(label_p_index, label_s_index)
                # print(pred_p_index, pred_s_index)
                # print(diff_p_index, diff_s_index)
                # raise

    # %%
    print(f"{len(events_empty) = }")
    print(f"{len(events_double) = }")

    # %%
    events_empty = dict(events_empty)
    with open("events_empty.json", "w") as fp:
        json.dump(events_empty, fp)

    events_double = dict(events_double)
    with open("events_double.json", "w") as fp:
        json.dump(events_double, fp)


if 2 in steps:
    # %%
    figure_path = Path("figures")
    if not figure_path.exists():
        figure_path.mkdir()
    if not (figure_path / "empty").exists():
        (figure_path / "empty").mkdir()
    if not (figure_path / "double").exists():
        (figure_path / "double").mkdir()

    # %%
    events_empty = json.load(open("events_empty.json", "r"))
    events_double = json.load(open("events_double.json", "r"))

    print(f"{len(events_empty) = }")
    print(f"{len(events_double) = }")

    with h5py.File(hdf5_file, "r") as f:

        for event in tqdm(events_empty):

            for pick in events_empty[event]:

                pick = Path(pick)

                data = f[pick.stem.replace("_", "/")][()]
                label_dict = dict(f[pick.stem.replace("_", "/")].attrs)
                try:
                    pred_dict = pd.read_csv(pick).to_dict(orient="list")
                except:
                    pred_dict = {"phase_index": [], "phase_type": []}

                label_p_index = [x for (x, y) in zip(label_dict["phase_index"], label_dict["phase_type"]) if y == "P"]
                label_s_index = [x for (x, y) in zip(label_dict["phase_index"], label_dict["phase_type"]) if y == "S"]
                pred_p_index = [x for (x, y) in zip(pred_dict["phase_index"], pred_dict["phase_type"]) if y == "P"]
                pred_s_index = [x for (x, y) in zip(pred_dict["phase_index"], pred_dict["phase_type"]) if y == "S"]

                fig, axs = plt.subplots(1, 1, figsize=(10, 3))
                for i in range(data.shape[0]):
                    axs.plot((data[i, :] - np.mean(data[i, :])) / np.std(data[i, :]) / 10 + i, color="k", linewidth=0.1)
                axs.set_ylim(-1, 3)

                for i in range(len(label_p_index)):
                    axs.axvline(label_p_index[i], color="C0", ymin=0, ymax=0.5)
                for i in range(len(label_s_index)):
                    axs.axvline(label_s_index[i], color="C3", ymin=0, ymax=0.5)
                for i in range(len(pred_p_index)):
                    axs.axvline(pred_p_index[i], color="C1", linestyle=":", ymin=0.5, ymax=1.0)
                for i in range(len(pred_s_index)):
                    axs.axvline(pred_s_index[i], color="C2", linestyle=":", ymin=0.5, ymax=1.0)

                fig.savefig(f"./{figure_path /'empty'/ pick.stem}.png", dpi=300)

        for event in tqdm(events_double):

            for pick in events_double[event]:

                pick = picks_path / pick

                data = f[pick.stem.replace("_", "/")][()]
                label_dict = dict(f[pick.stem.replace("_", "/")].attrs)
                try:
                    pred_dict = pd.read_csv(pick).to_dict(orient="list")
                except:
                    pred_dict = {"phase_index": [], "phase_type": []}

                label_p_index = [x for (x, y) in zip(label_dict["phase_index"], label_dict["phase_type"]) if y == "P"]
                label_s_index = [x for (x, y) in zip(label_dict["phase_index"], label_dict["phase_type"]) if y == "S"]
                pred_p_index = [x for (x, y) in zip(pred_dict["phase_index"], pred_dict["phase_type"]) if y == "P"]
                pred_s_index = [x for (x, y) in zip(pred_dict["phase_index"], pred_dict["phase_type"]) if y == "S"]

                fig, axs = plt.subplots(1, 1, figsize=(10, 3))
                for i in range(data.shape[0]):
                    axs.plot((data[i, :] - np.mean(data[i, :])) / np.std(data[i, :]) / 10 + i, color="k", linewidth=0.1)
                axs.set_ylim(-1, 3)

                for i in range(len(label_p_index)):
                    axs.axvline(label_p_index[i], color="C0", ymin=0, ymax=0.5)
                for i in range(len(label_s_index)):
                    axs.axvline(label_s_index[i], color="C3", ymin=0, ymax=0.5)
                for i in range(len(pred_p_index)):
                    axs.axvline(pred_p_index[i], color="C1", linestyle=":", ymin=0.5, ymax=1.0)
                for i in range(len(pred_s_index)):
                    axs.axvline(pred_s_index[i], color="C2", linestyle=":", ymin=0.5, ymax=1.0)

                fig.savefig(f"./{figure_path /'double'/ pick.stem}.png", dpi=300)

                # raise
                # if (len(pred_p_index) != 1) or (len(pred_s_index) != 1):

                #     fig, axs = plt.subplots(1, 1, figsize=(10, 10))
                #     for i in range(data.shape[0]):
                #         axs.plot((data[i, :] - np.mean(data[i, :]))/np.std(data[i, :])/10 + i, color="k", linewidth=0.5)
                #     axs.set_ylim(-1, 3)

                #     for i in range(len(label_p_index)):
                #         axs.axvline(label_p_index[i], color="C0", ymin=0, ymax=0.5)
                #     for i in range(len(label_s_index)):
                #         axs.axvline(label_s_index[i], color="C3", ymin=0, ymax=0.5)
                #     for i in range(len(pred_p_index)):
                #         axs.axvline(pred_p_index[i], color="C1", linestyle=":", ymin=0.5, ymax=1.0)
                #     for i in range(len(pred_s_index)):
                #         axs.axvline(pred_s_index[i], color="C2", linestyle=":", ymin=0.5, ymax=1.0)

                #     fig.savefig(f"./{figure_path / pick.stem}.png")
                #     # print(f"Saved {pick.stem}.png")

                #     # raise


# %%
# with h5py.File(hdf5_file, "r") as f:

#     for pick in tqdm(picks):

#         data = f[pick.stem.replace("_", "/")][()]
#         label_dict = dict(f[pick.stem.replace("_", "/")].attrs)
#         pred_dict = pd.read_csv(pick).to_dict(orient="list")

#         print(f"{label_dict = }")
#         print(f"{pred_dict = }")

#         label_p_index = [x for (x, y) in zip(label_dict["phase_index"], label_dict["phase_type"]) if y == "P"]
#         label_s_index = [x for (x, y) in zip(label_dict["phase_index"], label_dict["phase_type"]) if y == "S"]
#         pred_p_index = [x for (x, y) in zip(pred_dict["phase_index"], pred_dict["phase_type"]) if y == "P"]
#         pred_s_index = [x for (x, y) in zip(pred_dict["phase_index"], pred_dict["phase_type"]) if y == "S"]

#         print(len(label_p_index), len(label_s_index), len(pred_p_index), len(pred_s_index))
#         if (len(pred_p_index) != 1) or (len(pred_s_index) != 1):

#             fig, axs = plt.subplots(1, 1, figsize=(10, 10))
#             for i in range(data.shape[0]):
#                 axs.plot((data[i, :] - np.mean(data[i, :]))/np.std(data[i, :])/10 + i, color="k", linewidth=0.5)
#             axs.set_ylim(-1, 3)

#             for i in range(len(label_p_index)):
#                 axs.axvline(label_p_index[i], color="C0", ymin=0, ymax=0.5)
#             for i in range(len(label_s_index)):
#                 axs.axvline(label_s_index[i], color="C3", ymin=0, ymax=0.5)
#             for i in range(len(pred_p_index)):
#                 axs.axvline(pred_p_index[i], color="C1", linestyle=":", ymin=0.5, ymax=1.0)
#             for i in range(len(pred_s_index)):
#                 axs.axvline(pred_s_index[i], color="C2", linestyle=":", ymin=0.5, ymax=1.0)

#             fig.savefig(f"./{figure_path / pick.stem}.png")
#             # print(f"Saved {pick.stem}.png")

#             # raise
# %%
