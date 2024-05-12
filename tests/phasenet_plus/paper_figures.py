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
def calc_metrics(picks, labels, polarity_threshold, score_threshold, time_tolerance, phase_type, param="phase_time"):
    picks_ = picks[(picks["phase_score"] > score_threshold) & (picks["phase_type"] == phase_type)].copy()
    labels_ = labels[(labels["phase_type"] == phase_type)].copy()
    picks_.drop(columns=["phase_type"], inplace=True)
    labels_.drop(columns=["phase_type"], inplace=True)
    if "phase_polarity" in picks_.columns:
        picks_.loc[:, "phase_polarity"] = picks_["phase_polarity"].map(
            lambda x: "D" if x < -polarity_threshold else "N" if x < polarity_threshold else "U"
        )

    merged = pd.merge(labels_, picks_, how="left", on=["station_id", "event_index"], suffixes=("_true", "_pred"))

    if len(merged) == 0:
        print("No match found")
        return None

    def check_phase_time(row, tolerance=time_tolerance):
        return abs((row[f"phase_time_true"] - row[f"phase_time_pred"]).total_seconds()) <= tolerance

    def check_phase_polarity(row, tolerance=time_tolerance):
        return (
            abs((row[f"phase_time_true"] - row[f"phase_time_pred"]).total_seconds()) <= tolerance
            and row[f"phase_polarity_true"] == row[f"phase_polarity_pred"]
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

    return {
        "phase_type": phase_type,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TPR": TPR,
        "FPR": FPR,
        "param": param,
    }


def calc_error(picks, labels, score_threshold, phase_type):
    picks_ = picks[(picks["phase_score"] > score_threshold) & (picks["phase_type"] == phase_type)].copy()
    labels_ = labels[(labels["phase_type"] == phase_type)].copy()
    picks_.drop(columns=["phase_type"], inplace=True)
    labels_.drop(columns=["phase_type"], inplace=True)

    merged = pd.merge(labels_, picks_, how="left", on=["station_id", "event_index"], suffixes=("_true", "_pred"))
    merged["time_error"] = (merged["phase_time_pred"] - merged["phase_time_true"]).dt.total_seconds()

    return merged


# %%
def filter_duplicates(df):

    df.sort_values(
        ["event_index", "station_id", "phase_type", "phase_index"], ascending=[True, True, True, True], inplace=True
    )

    merged_rows = []
    previous_event_index = None
    previous_station_id = None
    previous_phase_type = None
    previous_phase_index = None
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if (
            (previous_event_index is not None)
            and (previous_station_id is not None)
            and (previous_phase_type is not None)
            and (previous_phase_index is not None)
            and (row["event_index"] == previous_event_index)
            and (row["station_id"] == previous_station_id)
            and (row["phase_type"] == previous_phase_type)
            and (abs(row["phase_index"] - previous_phase_index) <= 10)
        ):
            # merged_rows[-1] = row ## keep the last one
            pass  ## keep the first one
        else:
            merged_rows.append(row)
        previous_event_index = row["event_index"]
        previous_station_id = row["station_id"]
        previous_phase_type = row["phase_type"]
        previous_phase_index = row["phase_index"]
    print(f"Original: {len(df)}, Filtered: {len(merged_rows)}")
    df = pd.DataFrame(merged_rows).reset_index(drop=True)

    return df


# %%
def read_labels(file):
    label_list = []
    with h5py.File(file, "r") as fp:
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

    return labels


# %%
if __name__ == "__main__":

    # %% Load origin PhaseNet model
    reload = False
    region = "NC"
    phasenet_csv = f"phasenet_picks_{region}.csv"
    if reload or not os.path.exists(phasenet_csv):
        if region == "NC":
            phasenet_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/models/phasenet_picks.csv"
            )
        elif region == "SC":
            phasenet_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_sc/resolve/main/models/phasenet_picks.csv"
            )
        # phasenet_picks = pd.read_csv("phasenet_origin/picks_sc.csv")
        # phasenet_picks[["event_index", "station_id"]] = phasenet_picks["file_name"].str.split("/", expand=True)
        # phasenet_picks.drop(columns=["file_name", "begin_time"], inplace=True)
        # phasenet_picks.rename(columns={"event_index": "event_id"}, inplace=True)
        # phasenet_picks["phase_time"] = pd.to_datetime(phasenet_picks["phase_time"])
        # phasenet_picks["phase_time"] = phasenet_picks["phase_time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        # phasenet_picks.to_csv(
        #     "phasenet_picks.csv",
        #     columns=["event_id", "station_id", "phase_time", "phase_score", "phase_type"],
        #     index=False,
        # )
    else:
        phasenet_picks = pd.read_csv(phasenet_csv)
    phasenet_picks["event_index"] = phasenet_picks["event_id"]
    phasenet_picks["phase_time"] = pd.to_datetime(phasenet_picks["phase_time"])

    # %%
    phasenet_pt_csv = f"phasenet_pt_picks_{region}.csv"
    if reload or not os.path.exists(phasenet_pt_csv):
        if region == "NC":
            phasenet_pt_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/models/phasenet_pt_picks.csv"
            )
        elif region == "SC":
            phasenet_pt_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_sc/resolve/main/models/phasenet_pt_picks.csv"
            )
        # pick_path = "../../results_phasenet_quakeflow_sc/picks_phasenet"
        # event_ids = glob(f"{pick_path}/*")
        # picks_list = []
        # for event_id in tqdm(event_ids):
        #     station_ids = glob(f"{event_id}/*.csv")
        #     for station_id in station_ids:
        #         if os.stat(station_id).st_size == 0:
        #             # print(f"Empty file: {station_id}")
        #             continue
        #         phasenet_pt_picks = pd.read_csv(station_id)
        #         phasenet_pt_picks["event_index"] = event_id.split("/")[-1]
        #         picks_list.append(phasenet_pt_picks)
        # phasenet_pt_picks = pd.concat(picks_list, ignore_index=True)
        # phasenet_pt_picks["phase_time"] = pd.to_datetime(phasenet_pt_picks["phase_time"])
        # phasenet_pt_picks.drop(columns=["dt_s"], inplace=True)
        # phasenet_pt_picks = filter_duplicates(phasenet_pt_picks)
        # phasenet_pt_picks.drop(columns=["phase_index"], inplace=True)
        # phasenet_pt_picks.rename(columns={"event_index": "event_id"}, inplace=True)
        # phasenet_pt_picks.to_csv(
        #     phasenet_pt_csv,
        #     columns=["event_id", "station_id", "phase_time", "phase_score", "phase_type"],
        #     index=False,
        # )
    else:
        phasenet_pt_picks = pd.read_csv(phasenet_pt_csv, parse_dates=["phase_time"])

    phasenet_pt_picks["event_index"] = phasenet_pt_picks["event_id"]
    phasenet_pt_picks["phase_time"] = pd.to_datetime(phasenet_pt_picks["phase_time"])

    # %% Load PhaseNet+ model
    phasenet_plus_csv = f"phasenet_plus_picks_{region}.csv"
    if reload or not os.path.exists(phasenet_plus_csv):
        if region == "NC":
            phasenet_plus_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/models/phasenet_plus_picks.csv"
            )
        elif region == "SC":
            phasenet_plus_picks = pd.read_csv(
                "https://huggingface.co/datasets/AI4EPS/quakeflow_sc/resolve/main/models/phasenet_plus_picks.csv"
            )
        # pick_path = "../../results_ps_test_sc/picks_phasenet_plus"
        # event_ids = glob(f"{pick_path}/*")
        # picks_list = []
        # for event_id in tqdm(event_ids):
        #     station_ids = glob(f"{event_id}/*.csv")
        #     for station_id in station_ids:
        #         if os.stat(station_id).st_size == 0:
        #             # print(f"Empty file: {station_id}")
        #             continue
        #         phasenet_plus_picks = pd.read_csv(station_id)
        #         phasenet_plus_picks["event_index"] = event_id.split("/")[-1]
        #         picks_list.append(phasenet_plus_picks)
        # phasenet_plus_picks = pd.concat(picks_list, ignore_index=True)
        # phasenet_plus_picks["phase_time"] = pd.to_datetime(phasenet_plus_picks["phase_time"])
        # phasenet_plus_picks.drop(columns=["dt_s"], inplace=True)
        # phasenet_plus_picks = filter_duplicates(phasenet_plus_picks)
        # phasenet_plus_picks.drop(columns=["phase_index"], inplace=True)
        # phasenet_plus_picks.rename(columns={"event_index": "event_id"}, inplace=True)
        # phasenet_plus_picks.to_csv(
        #     phasenet_plus_csv,
        #     columns=["event_id", "station_id", "phase_time", "phase_score", "phase_type", "phase_polarity"],
        #     index=False,
        # )
    else:
        phasenet_plus_picks = pd.read_csv(phasenet_plus_csv, parse_dates=["phase_time"])

    phasenet_plus_picks["event_index"] = phasenet_plus_picks["event_id"]
    phasenet_plus_picks["phase_time"] = pd.to_datetime(phasenet_plus_picks["phase_time"])

    # %% Load labels
    label_csv = f"labels_{region}.csv"
    if not os.path.exists(label_csv):
        if region == "NC":
            labels = read_labels(file="/nfs/quakeflow_dataset/NC/quakeflow_nc/waveform_test.h5")
        elif region == "SC":
            labels = read_labels(file="/nfs/quakeflow_dataset/SC/quakeflow_sc/waveform_test.h5")
        labels.to_csv(label_csv, index=False)
    else:
        labels = pd.read_csv(label_csv, parse_dates=["phase_time"])

    # %%
    plt.figure()
    idx = phasenet_plus_picks["phase_type"] == "P"
    plt.hist(
        phasenet_plus_picks[idx]["phase_score"], bins=np.linspace(0.3, 1, 70 // 2 + 1), alpha=0.5, label="PhaseNet+"
    )
    idx = phasenet_picks["phase_type"] == "P"
    plt.hist(phasenet_picks[idx]["phase_score"], bins=np.linspace(0.3, 1, 70 // 2 + 1), alpha=0.5, label="PhaseNet")
    plt.legend()
    idx = phasenet_pt_picks["phase_type"] == "P"
    plt.hist(
        phasenet_pt_picks[idx]["phase_score"], bins=np.linspace(0.3, 1, 70 // 2 + 1), alpha=0.5, label="PhaseNet (PT)"
    )
    plt.legend()

    plt.figure()
    idx = phasenet_plus_picks["phase_type"] == "S"
    plt.hist(
        phasenet_plus_picks[idx]["phase_score"], bins=np.linspace(0.3, 1, 70 // 2 + 1), alpha=0.5, label="PhaseNet+"
    )
    idx = phasenet_picks["phase_type"] == "S"
    plt.hist(phasenet_picks[idx]["phase_score"], bins=np.linspace(0.3, 1, 70 // 2 + 1), alpha=0.5, label="PhaseNet")
    plt.legend()
    idx = phasenet_pt_picks["phase_type"] == "S"
    plt.hist(
        phasenet_pt_picks[idx]["phase_score"], bins=np.linspace(0.3, 1, 70 // 2 + 1), alpha=0.5, label="PhaseNet (PT)"
    )
    plt.legend()

    # %%
    plt.figure()
    idx = phasenet_plus_picks["phase_type"] == "P"
    plt.hist(phasenet_plus_picks[idx]["phase_polarity"], bins=np.linspace(-1, 1, 200 // 5 + 1))

    # %%
    plt.figure()
    idx = phasenet_plus_picks["phase_type"] == "S"
    plt.hist(phasenet_plus_picks[idx]["phase_polarity"], bins=np.linspace(-1, 1, 200 // 5 + 1))
    plt.yscale("log")

    # %%
    polarity_threshold = 0.5
    plt.figure()
    num_u = len(labels[labels["phase_polarity"] == "U"])
    num_n = len(labels[labels["phase_polarity"] == "N"])
    num_d = len(labels[labels["phase_polarity"] == "D"])
    plt.bar(["U", "N", "D"], [num_u, num_n, num_d], alpha=0.5, label="Labels")

    num_u = len(phasenet_plus_picks[phasenet_plus_picks["phase_polarity"] > polarity_threshold])
    num_n = len(phasenet_plus_picks[np.abs(phasenet_plus_picks["phase_polarity"].values) <= polarity_threshold])
    num_d = len(phasenet_plus_picks[phasenet_plus_picks["phase_polarity"] < -polarity_threshold])
    plt.bar(["U", "N", "D"], [num_u, num_n, num_d], alpha=0.5, label="PhaseNet+")
    plt.legend()

    # %%
    plt.figure()
    num_p_labels = len(labels[labels["phase_type"] == "P"])
    num_s_labels = len(labels[labels["phase_type"] == "S"])
    num_p_phasenet_plus = len(phasenet_plus_picks[phasenet_plus_picks["phase_type"] == "P"])
    num_s_phasenet_plus = len(phasenet_plus_picks[phasenet_plus_picks["phase_type"] == "S"])
    num_p_phasenet = len(phasenet_picks[phasenet_picks["phase_type"] == "P"])
    num_s_phasenet = len(phasenet_picks[phasenet_picks["phase_type"] == "S"])
    num_p_phasenet_pt = len(phasenet_pt_picks[phasenet_pt_picks["phase_type"] == "P"])
    num_s_phasenet_pt = len(phasenet_pt_picks[phasenet_pt_picks["phase_type"] == "S"])
    bar_width = 0.2
    index = np.arange(2)
    plt.bar(index, [num_p_labels, num_s_labels], bar_width, label="Labels")
    plt.bar(index + bar_width, [num_p_phasenet_plus, num_s_phasenet_plus], bar_width, label="PhaseNet+")
    plt.bar(index + 2 * bar_width, [num_p_phasenet, num_s_phasenet], bar_width, label="PhaseNet")
    plt.bar(index + 3 * bar_width, [num_p_phasenet_pt, num_s_phasenet_pt], bar_width, label="PhaseNet (PT)")
    plt.xticks(index + 1.5 * bar_width, ["P", "S"])
    plt.legend(loc="lower center")

    # %%
    polarity_threshold = 0.5  # [0.7, 1] or [-1, -0.7]
    score_threshold = 0.5
    time_tolerance = 0.1
    param = "phase_time"
    # param = "phase_polarity"

    # %%
    for phase_type in ["P", "S"]:
        plt.figure()
        # for picks, name in zip([phasenet_plus_picks, phasenet_picks], ["PhaseNet+", "PhaseNet"]):
        for picks, name in zip(
            [phasenet_plus_picks, phasenet_picks, phasenet_pt_picks], ["PhaseNet+", "PhaseNet", "PhaseNet (PT)"]
        ):

            merged = calc_error(picks, labels, score_threshold, phase_type)

            plt.hist(
                merged["time_error"],
                bins=np.linspace(-0.5, 0.5, 51),
                edgecolor="k",
                # facecolor="b",
                alpha=0.5,
                label=name,
            )
        plt.legend()
        plt.show()

    # %%
    metrics_summary = []
    for phase_type in ["P", "S"]:
        # for picks, name in zip([phasenet_plus_picks, phasenet_picks], ["PhaseNet+", "PhaseNet"]):
        for picks, name in zip(
            [phasenet_plus_picks, phasenet_picks, phasenet_pt_picks], ["PhaseNet+", "PhaseNet", "PhaseNet (PT)"]
        ):
            metrics_list = []
            for score_threshold in tqdm(np.linspace(0.1, 1, 20)):
                metrics = calc_metrics(
                    picks, labels, polarity_threshold, score_threshold, time_tolerance, phase_type, param
                )
                if metrics is not None:
                    metrics_list.append(metrics)

            metrics = pd.DataFrame(metrics_list)
            metrics["model"] = name
            metrics_summary.append(metrics)

    metrics_summary = pd.concat(metrics_summary, ignore_index=True)

    # %%
    plt.figure()
    idx = (metrics_summary["phase_type"] == "P") & (metrics_summary["model"] == "PhaseNet+")
    plt.plot(metrics_summary[idx]["recall"], metrics_summary[idx]["precision"], label="PhaseNet+")
    idx_f1 = np.argmax(metrics_summary[idx]["f1"])
    print(
        metrics_summary[idx].iloc[idx_f1]["recall"],
        metrics_summary[idx].iloc[idx_f1]["precision"],
        metrics_summary[idx].iloc[idx_f1]["f1"],
    )
    plt.scatter(metrics_summary[idx].iloc[idx_f1]["recall"], metrics_summary[idx].iloc[idx_f1]["precision"])

    idx = (metrics_summary["phase_type"] == "P") & (metrics_summary["model"] == "PhaseNet")
    plt.plot(metrics_summary[idx]["recall"], metrics_summary[idx]["precision"], label="PhaseNet")
    idx_f1 = np.argmax(metrics_summary[idx]["f1"])
    print(
        metrics_summary[idx].iloc[idx_f1]["recall"],
        metrics_summary[idx].iloc[idx_f1]["precision"],
        metrics_summary[idx].iloc[idx_f1]["f1"],
    )
    plt.scatter(metrics_summary[idx].iloc[idx_f1]["recall"], metrics_summary[idx].iloc[idx_f1]["precision"])

    idx = (metrics_summary["phase_type"] == "P") & (metrics_summary["model"] == "PhaseNet (PT)")
    plt.plot(metrics_summary[idx]["recall"], metrics_summary[idx]["precision"], label="PhaseNet (PT)")
    idx_f1 = np.argmax(metrics_summary[idx]["f1"])
    print(
        metrics_summary[idx].iloc[idx_f1]["recall"],
        metrics_summary[idx].iloc[idx_f1]["precision"],
        metrics_summary[idx].iloc[idx_f1]["f1"],
    )
    plt.scatter(metrics_summary[idx].iloc[idx_f1]["recall"], metrics_summary[idx].iloc[idx_f1]["precision"])

    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.5, 1.0])
    plt.ylim([0.5, 1.0])
    plt.figure()

    idx = (metrics_summary["phase_type"] == "S") & (metrics_summary["model"] == "PhaseNet+")
    plt.plot(metrics_summary[idx]["recall"], metrics_summary[idx]["precision"], label="PhaseNet+")
    idx_f1 = np.argmax(metrics_summary[idx]["f1"])
    print(
        metrics_summary[idx].iloc[idx_f1]["recall"],
        metrics_summary[idx].iloc[idx_f1]["precision"],
        metrics_summary[idx].iloc[idx_f1]["f1"],
    )
    plt.scatter(metrics_summary[idx].iloc[idx_f1]["recall"], metrics_summary[idx].iloc[idx_f1]["precision"])

    idx = (metrics_summary["phase_type"] == "S") & (metrics_summary["model"] == "PhaseNet")
    plt.plot(metrics_summary[idx]["recall"], metrics_summary[idx]["precision"], label="PhaseNet")
    idx_f1 = np.argmax(metrics_summary[idx]["f1"])
    print(
        metrics_summary[idx].iloc[idx_f1]["recall"],
        metrics_summary[idx].iloc[idx_f1]["precision"],
        metrics_summary[idx].iloc[idx_f1]["f1"],
    )
    plt.scatter(metrics_summary[idx].iloc[idx_f1]["recall"], metrics_summary[idx].iloc[idx_f1]["precision"])

    idx = (metrics_summary["phase_type"] == "S") & (metrics_summary["model"] == "PhaseNet (PT)")
    plt.plot(metrics_summary[idx]["recall"], metrics_summary[idx]["precision"], label="PhaseNet (PT)")
    idx_f1 = np.argmax(metrics_summary[idx]["f1"])
    print(
        metrics_summary[idx].iloc[idx_f1]["recall"],
        metrics_summary[idx].iloc[idx_f1]["precision"],
        metrics_summary[idx].iloc[idx_f1]["f1"],
    )
    plt.scatter(metrics_summary[idx].iloc[idx_f1]["recall"], metrics_summary[idx].iloc[idx_f1]["precision"])

    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.5, 1.0])
    plt.ylim([0.5, 1.0])
    # %%
    plt.figure()
    idx = (metrics_summary["phase_type"] == "P") & (metrics_summary["model"] == "PhaseNet+")
    plt.plot(
        [1.0] + list(metrics_summary[idx]["FPR"]) + [0.0],
        [1.0] + list(metrics_summary[idx]["TPR"]) + [0.0],
        label="PhaseNet+",
    )
    idx = (metrics_summary["phase_type"] == "P") & (metrics_summary["model"] == "PhaseNet")
    plt.plot(
        [1.0] + list(metrics_summary[idx]["FPR"]) + [0.0],
        [1.0] + list(metrics_summary[idx]["TPR"]) + [0.0],
        label="PhaseNet",
    )
    idx = (metrics_summary["phase_type"] == "P") & (metrics_summary["model"] == "PhaseNet (PT)")
    plt.plot(
        [1.0] + list(metrics_summary[idx]["FPR"]) + [0.0],
        [1.0] + list(metrics_summary[idx]["TPR"]) + [0.0],
        label="PhaseNet (PT)",
    )
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.figure()
    idx = (metrics_summary["phase_type"] == "S") & (metrics_summary["model"] == "PhaseNet+")
    plt.plot(
        [1.0] + list(metrics_summary[idx]["FPR"]) + [0.0],
        [1.0] + list(metrics_summary[idx]["TPR"]) + [0.0],
        label="PhaseNet+",
    )
    idx = (metrics_summary["phase_type"] == "S") & (metrics_summary["model"] == "PhaseNet")
    plt.plot(
        [1.0] + list(metrics_summary[idx]["FPR"]) + [0.0],
        [1.0] + list(metrics_summary[idx]["TPR"]) + [0.0],
        label="PhaseNet",
    )
    idx = (metrics_summary["phase_type"] == "S") & (metrics_summary["model"] == "PhaseNet (PT)")
    plt.plot(
        [1.0] + list(metrics_summary[idx]["FPR"]) + [0.0],
        [1.0] + list(metrics_summary[idx]["TPR"]) + [0.0],
        label="PhaseNet (PT)",
    )
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # %%
    phase_type = "P"
    polarity_threshold = 0.5
    score_threshold = 0.5
    picks = phasenet_plus_picks
    metrics = calc_metrics(picks, labels, polarity_threshold, score_threshold, time_tolerance, phase_type, "phase_time")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    metrics = calc_metrics(
        picks, labels, polarity_threshold, score_threshold, time_tolerance, phase_type, "phase_polarity"
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")

# %%
