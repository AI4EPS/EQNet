# %%
import os
from functools import partial
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# %%
pick_path = "../../results_ps_test/picks_phasenet_plus"

# %%
event_ids = glob(f"{pick_path}/*")

# %%
picks_list = []
for event_id in tqdm(event_ids):
    station_ids = glob(f"{event_id}/*.csv")
    for station_id in station_ids:
        if os.stat(station_id).st_size == 0:
            print(f"Empty file: {station_id}")
            continue
        picks = pd.read_csv(station_id, parse_dates=["phase_time"])
        picks["event_index"] = event_id.split("/")[-1]
        picks_list.append(picks)
picks = pd.concat(picks_list, ignore_index=True)
picks["phase_time"] = pd.to_datetime(picks["phase_time"])

# %%
plt.figure()
plt.hist(picks["phase_score"], bins=np.linspace(0.3, 1, 70 // 2 + 1))
# %%
plt.figure()
idx = picks["phase_type"] == "P"
plt.hist(picks[idx]["phase_polarity"], bins=np.linspace(-1, 1, 200 // 5 + 1))
# %%
# %%
plt.figure()
idx = picks["phase_type"] == "S"
plt.hist(picks[idx]["phase_polarity"], bins=np.linspace(-1, 1, 200 // 5 + 1))
plt.yscale("log")

# %%
# plot P and S phase freqency in bar plot
num_p = len(picks[picks["phase_type"] == "P"])
num_s = len(picks[picks["phase_type"] == "S"])
plt.figure()
plt.bar(["P", "S"], [num_p, num_s])

# %%
label_list = []
with h5py.File("/nfs/quakeflow_dataset/NC/quakeflow_nc/waveform_test.h5", "r") as fp:
    event_ids = list(fp.keys())
    for event_id in tqdm(event_ids):
        event_attrs = dict(fp[event_id].attrs)
        station_ids = list(fp[event_id].keys())
        # for station_id in station_ids:
        #     label.append(fp[event_id][station_id]["label"][()])
        for station_id in station_ids:
            station_attrs = dict(fp[event_id][station_id].attrs)
            num_picks = len(station_attrs["phase_time"])
            labels = {
                "event_index": [event_id] * num_picks,
                "station_id": [station_id] * num_picks,
                "phase_time": list(station_attrs["phase_time"]),
                "phase_type": list(station_attrs["phase_type"]),
                "phase_polarity": list(station_attrs["phase_polarity"]),
            }
            label_list.append(pd.DataFrame(labels))
    labels = pd.concat(label_list, ignore_index=True)
    labels["phase_time"] = pd.to_datetime(labels["phase_time"])
    # labels["phase_polarity"] = labels["phase_polarity"].map({"U": 1, "D": -1, "N": 0})


# %%
# plot P and S phase freqency in bar plot
num_p = len(picks[picks["phase_type"] == "P"])
num_s = len(picks[picks["phase_type"] == "S"])
plt.figure()
plt.bar(["P", "S"], [num_p, num_s])

num_p = len(labels[labels["phase_type"] == "P"])
num_s = len(labels[labels["phase_type"] == "S"])
plt.bar(["P_", "S_"], [num_p, num_s])


# %%
def calc_metrics(picks, labels, polarity_threshold, score_threshold, time_tolerance, phase_type, param="phase_time"):
    picks_ = picks[
        (picks["phase_score"] > score_threshold)
        & (picks["phase_type"] == phase_type)
        # & (abs(picks["phase_polarity"]) >= polarity_threshold)
        # & (picks["phase_polarity"] >= polarity_threshold)
        # & (picks["phase_polarity"] <= -polarity_threshold)
    ]
    labels_ = labels[
        (labels["phase_type"] == phase_type)
        #  & (labels["phase_polarity"] != "N")
        # & (labels["phase_polarity"] == "U")
        # & (labels["phase_polarity"] == "D")
    ]
    picks_.loc[:, "phase_polarity"] = picks_["phase_polarity"].map(
        lambda x: "D" if x < -polarity_threshold else "N" if x < polarity_threshold else "U"
    )
    # labels_.loc[:, "phase_polarity"] = labels_["phase_polarity"].map(
    #     lambda x: "D" if x < -polarity_tolerance else "N" if x < polarity_tolerance else "U"
    # )

    merged = pd.merge(labels_, picks_, on=["station_id", "event_index"], suffixes=("_true", "_pred"))

    if len(merged) == 0:
        print("No match found")
        return None

    def check_phase_time(row, tolerance=time_tolerance):
        return (
            abs((row[f"phase_time_true"] - row[f"phase_time_pred"]).total_seconds()) <= tolerance
            and row["phase_type_true"] == row["phase_type_pred"]
        )

    def check_phase_polarity(row, tolerance=time_tolerance):
        return (
            abs((row[f"phase_time_true"] - row[f"phase_time_pred"]).total_seconds()) <= tolerance
            and row[f"phase_polarity_true"] == row[f"phase_polarity_pred"]
            and row["phase_type_true"] == row["phase_type_pred"]
        )

    if param == "phase_time":
        merged["correct"] = merged.apply(check_phase_time, axis=1)
    elif param == "phase_polarity":
        merged["correct"] = merged.apply(check_phase_polarity, axis=1)
    else:
        raise ValueError("Invalid check")

    true_positives = merged["correct"].sum()
    false_positives = len(picks_) - true_positives
    false_negatives = len(labels_) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    TPR = true_positives / len(labels_) if len(labels_) > 0 else 0
    FPR = false_positives / len(picks_) if len(picks_) > 0 else 0

    print(f"Polarity threshold: {polarity_threshold}")
    print(f"Score threshold: {score_threshold}")
    print(f"Time tolerance: {time_tolerance}")
    print(f"Parameter: {param}")
    print(f"Phase: {phase_type}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"TPR: {TPR:.4f}")
    print(f"FPR: {FPR:.4f}")

    return {
        "phase_type": phase_type,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TPR": TPR,
        "FPR": FPR,
        "param": param,
    }


polarity_threshold = 0.5  # [0.7, 1] or [-1, -0.7]
score_threshold = 0.5
time_tolerance = 0.1
phase_type = "P"
param = "phase_time"
# param = "phase_polarity"

metrics = calc_metrics(picks, labels, polarity_threshold, score_threshold, time_tolerance, phase_type, param)

# %%
metrics_list = []
for phase_type in ["P", "S"]:
    for score_threshold in tqdm(np.linspace(0.1, 1, 20)):
        metrics = calc_metrics(picks, labels, polarity_threshold, score_threshold, time_tolerance, phase_type, param)
        if metrics is not None:
            metrics_list.append(metrics)

metrics = pd.DataFrame(metrics_list)

# %%
plt.figure()
idx = metrics["phase_type"] == "P"
plt.plot(metrics[idx]["recall"], metrics[idx]["precision"], label="P")
idx_f1 = np.argmax(metrics[idx]["f1"])
plt.scatter(metrics[idx].iloc[idx_f1]["recall"], metrics[idx].iloc[idx_f1]["precision"], color="red")
plt.legend()
plt.xlabel("Recall")
plt.xlabel("Precision")
plt.xlim([0.5, 1.0])
plt.ylim([0.5, 1.0])
plt.figure()
idx = metrics["phase_type"] == "S"
plt.plot(metrics[idx]["recall"], metrics[idx]["precision"], label="S")
idx_f1 = np.argmax(metrics[idx]["f1"])
plt.scatter(metrics[idx].iloc[idx_f1]["recall"], metrics[idx].iloc[idx_f1]["precision"], color="red")
plt.legend()
plt.xlabel("Recall")
plt.xlabel("Precision")
plt.xlim([0.5, 1.0])
plt.ylim([0.5, 1.0])
# %%
plt.figure()
idx = metrics["phase_type"] == "P"
plt.plot([1.0] + list(metrics[idx]["FPR"]) + [0.0], [1.0] + list(metrics[idx]["TPR"]) + [0.0], label="P")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.figure()
idx = metrics["phase_type"] == "S"
plt.plot([1.0] + list(metrics[idx]["FPR"]) + [0.0], [1.0] + list(metrics[idx]["TPR"]) + [0.0], label="S")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

# %%
