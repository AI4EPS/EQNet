# %%
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json
import multiprocessing as mp
import fsspec
from pathlib import Path

# %%
result_path = Path("/home/weiqiang/data/quakeflow_nc/data")
if not result_path.exists():
    result_path.mkdir()


# # %%
# with open("event_ids.txt") as f:
#     event_ids = f.read().splitlines()

# # %%
# hdf5 = "gs://quakeflow_ncedc/ncedc_event_dataset_polarity.h5"
# with fsspec.open(hdf5, "rb") as fs:
#     with h5py.File(fs, "r") as fp:
#         for event_id in event_ids:
#             for station in fp[event_id].keys():
#                 print(fp[event_id][station].shape)
#                 raise

# %%
# hdf5 = "gs://quakeflow_ncedc/ncedc_event_dataset_polarity.h5"
# hdf5 = "ncedc_event_dataset_polarity.h5"
hdf5 = "/home/weiqiang/data/waveform.h5"
if not Path("event_ids.txt").exists():
    with fsspec.open(hdf5, "rb") as fs:
        with h5py.File(fs, "r") as fp:
            event_ids = list(fp.keys())

    # %%
    with open("event_ids.txt", "w") as f:
        f.write("\n".join(event_ids))


# %%
def save_by_year(year, hdf5):
    year1, year2 = year
    if (year2 - year1) == 1:
        name = f"NC{year1}.h5"
    else:
        name = f"NC{year1}-{year2-1}.h5"
    with fsspec.open(hdf5, "rb") as fs:
        with h5py.File(fs, "r") as fp:
            with h5py.File(result_path / name, "w") as fout:
                select_events = 0
                # for event_id in fp.keys():
                for event_id in event_ids:
                    # print(event_id)
                    y = int(fp[event_id].attrs["event_time"][:4])
                    if y >= year1 and y < year2:
                        fp.copy(fp[event_id], fout, name=event_id)
                        select_events += 1
                print(f"{name}, select_events: {select_events}")


# %%
if __name__ == "__main__":
    # %%
    with open("event_ids.txt") as f:
        event_ids = f.read().splitlines()

    years = [
        [1970, 1990],
        [1990, 1995],
        [1995, 2000],
        [2000, 2005],
        [2005, 2010],
    ] + [[x, x + 1] for x in range(2010, 2023)]
    print(years)
    num_cpu = len(years)
    with mp.Pool(num_cpu) as pool:
        pool.starmap(save_by_year, [(year, hdf5) for year in years])
# %%
