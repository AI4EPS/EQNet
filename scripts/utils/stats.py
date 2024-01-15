# %%
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp

# %%
# root_path = Path("results")
# paths = ["training_v0", "training_v1", "training_v2"]
root_path = Path("results/gamma")
paths = ["phasenet", "phasenet_das", "phasenet_das_v1"]
# paths = ["phasenet_das_v1"]

# %%
folders = ["ridgecrest_north", "ridgecrest_south", "mammoth_north", "mammoth_south"]
# folders = folders[::-1]
# folders = ["ridgecrest_north"]


# %%
# def count_picks(csv_list):
#     associated_picks = 0
#     num_picks = 0
#     for csv in tqdm(csv_list):
#         df = pd.read_csv(csv)
#         # Sort the DataFrame
#         df.sort_values(["station_id", "phase_index", "event_index"], ascending=[True, True, True], inplace=True)

#         # Define a custom function to merge rows
#         def merge_rows(group):
#             merged_rows = []
#             previous_index = None

#             for i, row in group.iterrows():
#                 if previous_index is not None and abs(row["phase_index"] - previous_index) <= 2:
#                     # if row["event_index"] != -1:
#                     merged_rows[-1] = row
#                 else:
#                     merged_rows.append(row)

#                 previous_index = row["phase_index"]

#             return pd.DataFrame(merged_rows)

#         # Group by station_id and apply the custom function
#         df = df.groupby("station_id", group_keys=False).apply(merge_rows).reset_index(drop=True)

#         num_picks += len(df)
#         associated_picks += len(df[df["event_index"] != -1])
#     return num_picks, associated_picks


def process(csv, num_picks, num_associated, stats, lock):
    df = pd.read_csv(csv)

    # Sort the DataFrame
    df.sort_values(["station_id", "phase_index", "event_index"], ascending=[True, True, False], inplace=True)

    # Define a custom function to merge rows
    def merge_rows(group):
        merged_rows = []
        previous_index = None

        for i, row in group.iterrows():
            if (previous_index is not None) and (abs(row["phase_index"] - previous_index) <= 10):
                if row["event_index"] != -1:  # event_index may not be ascending
                    merged_rows[-1] = row
            else:
                merged_rows.append(row)

            previous_index = row["phase_index"]

        return pd.DataFrame(merged_rows)

    # Group by station_id and apply the custom function
    df = df.groupby("station_id", group_keys=False).apply(merge_rows).reset_index(drop=True)

    with lock:
        num_picks.value += len(df)
        num_associated.value += len(df[df["event_index"] != -1])
        stats.append(
            [
                len(df),
                len(df[df["event_index"] != -1]),
                len(df[df["phase_type"] == "P"]),
                len(df[(df["phase_type"] == "P") & (df["event_index"] != -1)]),
                len(df[df["phase_type"] == "S"]),
                len(df[(df["phase_type"] == "S") & (df["event_index"] != -1)]),
            ]
        )


def count_picks(csv_list):
    manager = mp.Manager()
    num_picks = manager.Value("i", 0, lock=True)
    num_associated = manager.Value("i", 0, lock=True)
    stats = manager.list()
    lock = manager.Lock()
    num_cpu = mp.cpu_count()
    pbar = tqdm(total=len(csv_list))
    with mp.Pool(num_cpu) as pool:
        for csv in csv_list:
            pool.apply_async(
                process, args=(csv, num_picks, num_associated, stats, lock), callback=lambda x: pbar.update()
            )
        pool.close()
        pool.join()
    pbar.close()
    return num_picks.value, num_associated.value, list(stats)


# %%
with open("stats.txt", "w") as f:
    f.write(f"path,folder,num_csv,picks,associated_num,associated_ratio\n")
    for folder in folders:
        for path in paths:
            csv_list = list((root_path / path / folder / "picks").glob("*.csv"))
            num_csv = len(csv_list)
            num_picks, num_associated, stats = count_picks(csv_list)
            print(
                f"{path} {folder}: {num_csv} csv, {num_picks} picks, {num_associated} associated picks, {num_associated/num_picks*100:.2f}% associated"
            )
            f.write(f"{path},{folder},{num_csv},{num_picks},{num_associated},{num_associated/num_picks*100:.2f}\n")
            with open(f"stats_{path}_{folder}.txt", "w") as f2:
                f2.write(f"num_picks,num_associated,num_picks_P,num_associated_P,num_picks_S,num_associated_S\n")
                for num_pick, num_associated, num_pick_P, num_associated_P, num_pick_S, num_associated_S in stats:
                    f2.write(
                        f"{num_pick},{num_associated},{num_pick_P},{num_associated_P},{num_pick_S},{num_associated_S}\n"
                    )
# %%
