# %%
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import multiprocessing as mp

plt.rcParams["figure.facecolor"] = "white"

# %%
h5_in = "/home/weiqiang/data/ncedc_event_dataset_polarity.h5"
h5_out = "/home/weiqiang/data/ncedc_event_dataset.h5"

num_per_file = 10000

num = 0
with h5py.File(h5_in, "r") as fp_in:
    with h5py.File(h5_out, "w") as fp_out:
        for event_id in tqdm(sorted(list(fp_in.keys()))):
            event = fp_in[event_id]

            for station_id, station in event.items():
                waveform = station[:, :].T  # 3, nt
                waveform = waveform.astype(np.float32)

                if event_id not in fp_out:
                    group_ = fp_out.create_group(event_id)
                    for key, value in event.attrs.items():
                        group_.attrs[key] = value

                if station_id not in fp_out[event_id]:
                    dataset_ = fp_out[event_id].create_dataset(station_id, data=waveform)
                    for key, value in station.attrs.items():
                        dataset_.attrs[key] = value
                else:
                    print("Duplicate station", event_id, station_id)

            num += 1

            # if num >= 2000:
            #     break
print(f"Total number of samples: {num}")
